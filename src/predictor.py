"""
The trained predictor, as a deployable artifact rather than an experiment.

Milestone 2 established the shape: a per-layer ridge probe on PCA-reduced hidden
states, predicting which experts layer L+1 will select from layer L's residual
stream, reaching 59.7% recall@8 on held-out registers against a 39.5% bar.

This packages that for the prefetch engine, where two properties matter that did
not matter in the experiment:

  cheap   the predictor runs 47 times per token, inside the critical path it is
          trying to speed up. Per layer it is one 256x128 matmul on a 256-vector
          -- ~33K multiply-adds -- against the 2.92 MiB fetch it decides. If the
          predictor cost as much as a fetch there would be no point.
  fixed   PCA basis, per-layer weights and normalisation are all frozen at fit
          time and saved together, so inference cannot silently disagree with
          training about scaling. Getting that wrong would show up as a mildly
          disappointing hit rate rather than an error, which is the worst kind
          of bug to have here.
"""
import numpy as np

N_EXPERT = 128


class ExpertPredictor:
    """Per-layer ridge probes over a shared PCA basis."""

    def __init__(self, mean, comp, scale, weights):
        self.mean = mean            # (d,)      PCA centring, from train rows
        self.comp = comp            # (d, k)    principal components
        self.scale = scale          # (k,)      per-component std, train rows
        self.weights = weights      # {layer: (k, n_expert)}

    # -- fitting -----------------------------------------------------------
    @classmethod
    def fit(cls, H, E, pairs, train_mask, n_comp=256, lam=100.0):
        """H: (n, d) hidden states. E: (n, 8) expert ids.
        pairs: (src_idx, dst_idx) arrays -- h at src predicts experts at dst.
        train_mask: over `pairs`, which rows may be used for fitting.
        """
        src, dst = pairs
        rows = np.unique(src[train_mask])
        mean = H[rows].mean(0)
        Xc = H[rows] - mean
        C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
        w, V = np.linalg.eigh(C.astype(np.float64))
        comp = V[:, ::-1][:, :n_comp].astype(np.float32)
        proj = (H[rows] - mean) @ comp
        scale = proj.std(0) + 1e-6
        return mean, comp, scale

    @classmethod
    def fit_layer(cls, X, Y, lam=100.0):
        """Dual-form ridge: samples (~thousands) far outnumber nothing here, but
        the feature count (256) is small enough that either form works. Primal is
        used because it is the one that produces the (k, n_expert) matrix the
        engine wants to keep."""
        G = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float32)
        return np.linalg.solve(G, X.T @ Y).astype(np.float32)

    # -- inference ---------------------------------------------------------
    def project(self, h):
        """Hidden state(s) -> PCA features. Accepts (d,) or (n, d)."""
        return ((h - self.mean) @ self.comp) / self.scale

    def top_p(self, layer, h_proj, p=8):
        """Predicted experts for layer+1, given layer's projected hidden state."""
        W = self.weights.get(layer)
        if W is None:
            return np.empty(0, dtype=np.int64)
        s = h_proj @ W
        return np.argpartition(-s, p)[:p]

    def scores(self, layer, h_proj):
        W = self.weights.get(layer)
        return None if W is None else h_proj @ W

    # -- persistence -------------------------------------------------------
    def save(self, path):
        np.savez_compressed(
            path, mean=self.mean, comp=self.comp, scale=self.scale,
            layers=np.array(sorted(self.weights)),
            stack=np.stack([self.weights[l] for l in sorted(self.weights)]))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        weights = {int(l): w for l, w in zip(d["layers"], d["stack"])}
        return cls(d["mean"], d["comp"], d["scale"], weights)

    def n_params(self):
        return (self.comp.size + sum(w.size for w in self.weights.values()))
