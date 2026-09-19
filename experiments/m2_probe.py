"""
Milestone 2, step 2: the same question, scored honestly.

Milestone 1's 51.3% was a ceiling, not an estimate: the ridge strength and the
ensemble weight were both chosen by their score on the held-out rows. This fixes
that and adds the nonlinearity milestone 1 never tried.

  split        by REGISTER, three ways -- 5 registers train, 1 validate, 2 test.
               Registers rather than prompts because rows inside one prompt share
               a context and leak; registers rather than a random split for the
               same reason one level up. Every hyperparameter is chosen on
               validation and the test registers are scored exactly once.
  features     PCA to N_COMP dims, fitted on TRAIN ROWS ONLY. 2048 features
               against ~3.5K rows per layer is the overparameterised regime that
               made milestone 1's shared probe collapse; it also makes 47 MLPs
               tractable on CPU.
  predictors   per-layer ridge (closed form), per-layer MLP (one hidden layer),
               the repeat prior, and ensembles of each probe with that prior.

The bar is naive-repeat, which is not prefetchable and therefore not a candidate
implementation -- it is here because it is the number every trained predictor in
AI2 failed to beat, and the only reason this project continued past milestone 1
is that a *prefetchable* predictor got past it.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA", os.path.join(HERE, "m2_hidden.npz"))
OUT = os.path.join(HERE, "m2_probe_result.json")
N_COMP = int(os.environ.get("N_COMP", "256"))
LAMBDAS = [0.1, 1, 10, 100, 1000]
WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
HIDDEN = int(os.environ.get("HIDDEN", "256"))
EPOCHS = int(os.environ.get("EPOCHS", "150"))
N_EXPERT, K = 128, 8


def recall_at_k(scores, truth, k=K):
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    return np.mean([len(set(t) & set(p)) for t, p in zip(truth, top)]) / k


def zscore_rows(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)


def train_mlp(Xtr, Ytr, Xva, Xte, seed=0):
    """One hidden layer, plain SGD in numpy. Returns (val_scores, test_scores)."""
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    W1 = rng.normal(0, (2.0 / d) ** 0.5, (d, HIDDEN)).astype(np.float32)
    b1 = np.zeros(HIDDEN, dtype=np.float32)
    W2 = rng.normal(0, (2.0 / HIDDEN) ** 0.5, (HIDDEN, N_EXPERT)).astype(np.float32)
    b2 = np.zeros(N_EXPERT, dtype=np.float32)
    Y = Ytr / Ytr.sum(1, keepdims=True)
    lr, n = 0.5, len(Xtr)
    for _ in range(EPOCHS):
        Z = Xtr @ W1 + b1
        A = np.maximum(Z, 0)
        logits = A @ W2 + b2
        logits -= logits.max(1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(1, keepdims=True)
        dL = (P - Y) / n
        gW2, gb2 = A.T @ dL, dL.sum(0)
        dA = dL @ W2.T
        dZ = dA * (Z > 0)
        gW1, gb1 = Xtr.T @ dZ, dZ.sum(0)
        W2 -= lr * gW2; b2 -= lr * gb2; W1 -= lr * gW1; b1 -= lr * gb1

    def fwd(X):
        return np.maximum(X @ W1 + b1, 0) @ W2 + b2

    return fwd(Xva), fwd(Xte)


def main():
    if not os.path.exists(DATA):
        sys.exit(f"missing {DATA} -- run m2_capture.py first")
    d = np.load(DATA, allow_pickle=True)
    H = d["hidden"].astype(np.float32)
    E = d["experts"].astype(np.int64)
    keys, reg = d["keys"], d["register"].astype(str)
    pid, tok, lay = keys[:, 0], keys[:, 1], keys[:, 2]
    idx = {(p, t, l): i for i, (p, t, l) in enumerate(zip(pid, tok, lay))}
    n_layers = int(lay.max()) + 1

    regs = sorted(set(reg))
    train_r, val_r, test_r = regs[:5], regs[5:6], regs[6:]
    print(f"{len(H):,} rows, {n_layers} layers, registers {regs}", file=sys.stderr)
    print(f"train={train_r}  val={val_r}  test={test_r}", file=sys.stderr)

    # PCA on train rows only
    tr_mask = np.isin(reg, train_r)
    mu = H[tr_mask].mean(0)
    Xc = H[tr_mask] - mu
    # randomised-free: covariance eigendecomposition at 2048x2048 is fine here
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w, V = np.linalg.eigh(C.astype(np.float64))
    comp = V[:, ::-1][:, :N_COMP].astype(np.float32)
    var = float(w[::-1][:N_COMP].sum() / w.sum())
    print(f"PCA {H.shape[1]} -> {N_COMP} dims, {100 * var:.1f}% variance retained",
          file=sys.stderr)
    Hp = (H - mu) @ comp
    sd = Hp[tr_mask].std(0) + 1e-6
    Hp /= sd

    per_layer, agg = [], {k: [] for k in
                          ("ridge", "mlp", "repeat", "ridge_ens", "mlp_ens", "random")}
    rng = np.random.default_rng(1)

    for L in range(n_layers - 1):
        src, dst = [], []
        for (p, t, l), i in idx.items():
            if l == L and (p, t, L + 1) in idx:
                src.append(i)
                dst.append(idx[(p, t, L + 1)])
        if len(src) < 200:
            continue
        src, dst = np.array(src), np.array(dst)
        r = reg[src]
        m_tr, m_va, m_te = np.isin(r, train_r), np.isin(r, val_r), np.isin(r, test_r)
        if min(m_tr.sum(), m_va.sum(), m_te.sum()) < 40:
            continue

        X = Hp[src]
        Y = np.zeros((len(dst), N_EXPERT), dtype=np.float32)
        Y[np.arange(len(dst))[:, None], E[dst]] = 1.0
        Xtr, Ytr = X[m_tr], Y[m_tr]
        Eva, Ete = E[dst][m_va], E[dst][m_te]

        # repeat prior, as multi-hot over E_{L+1}(t-1), for every split
        def prior_for(mask):
            P = np.zeros((mask.sum(), N_EXPERT), dtype=np.float32)
            for m, row in enumerate(np.where(mask)[0]):
                p, t = keys[src[row]][:2]
                j = idx.get((p, t - 1, L + 1))
                if j is not None:
                    P[m, E[j]] = 1.0
            return P
        Pva, Pte = prior_for(m_va), prior_for(m_te)

        # --- ridge, lambda chosen on validation
        G = Xtr @ Xtr.T
        best_lam, best_va = None, -1.0
        for lam in LAMBDAS:
            A = np.linalg.solve(G + lam * np.eye(len(G), dtype=np.float32), Ytr)
            v = recall_at_k(X[m_va] @ (Xtr.T @ A), Eva)
            if v > best_va:
                best_lam, best_va = lam, v
        A = np.linalg.solve(G + best_lam * np.eye(len(G), dtype=np.float32), Ytr)
        Wd = Xtr.T @ A
        ridge_va, ridge_te = X[m_va] @ Wd, X[m_te] @ Wd

        # --- mlp
        mlp_va, mlp_te = train_mlp(Xtr, Ytr, X[m_va], X[m_te])

        # --- ensemble weights chosen on validation, applied to test
        def ens(sv, st):
            bw = max(WEIGHTS, key=lambda w: recall_at_k(zscore_rows(sv) + w * Pva, Eva))
            return recall_at_k(zscore_rows(st) + bw * Pte, Ete), bw

        ridge_ens, w_r = ens(ridge_va, ridge_te)
        mlp_ens, w_m = ens(mlp_va, mlp_te)

        row = {
            "layer": L,
            "ridge": recall_at_k(ridge_te, Ete),
            "mlp": recall_at_k(mlp_te, Ete),
            "repeat": recall_at_k(Pte + 1e-6 * rng.normal(size=Pte.shape), Ete),
            "ridge_ens": ridge_ens,
            "mlp_ens": mlp_ens,
            "random": recall_at_k(rng.normal(size=(len(Ete), N_EXPERT)), Ete),
            "lam": best_lam, "w_ridge": w_r, "w_mlp": w_m, "n_test": int(m_te.sum()),
        }
        per_layer.append(row)
        for k in agg:
            agg[k].append(row[k])
        if L % 12 == 0:
            print(f"  layer {L:2d}: ridge {100 * row['ridge']:.1f} mlp {100 * row['mlp']:.1f} "
                  f"repeat {100 * row['repeat']:.1f} ens {100 * row['mlp_ens']:.1f}",
                  file=sys.stderr, flush=True)

    print(f"\n=== MILESTONE 2: HONEST SPLIT ({len(per_layer)} layer transitions, "
          f"test registers {test_r}) ===\n")
    print("%-34s %10s   %s" % ("predictor", "recall@8", "prefetchable?"))
    order = [("MLP probe + repeat prior", "mlp_ens", "YES"),
             ("ridge probe + repeat prior", "ridge_ens", "YES"),
             ("naive-repeat (the bar)", "repeat", "no"),
             ("MLP probe on h_L alone", "mlp", "YES"),
             ("ridge probe on h_L alone", "ridge", "YES"),
             ("random", "random", "--")]
    for label, k, pf in order:
        print("%-34s %9.1f%%   %s" % (label, 100 * np.mean(agg[k]), pf))

    bar = np.mean(agg["repeat"])
    best_k = max(("mlp_ens", "ridge_ens"), key=lambda k: np.mean(agg[k]))
    gain = np.mean(agg[best_k]) - bar
    print(f"\nBest prefetchable predictor beats the naive-repeat bar by "
          f"{100 * gain:+.1f} points ({100 * np.mean(agg[best_k]):.1f}% vs {100 * bar:.1f}%).")
    wins = sum(1 for r in per_layer if r[best_k] > r["repeat"])
    print(f"It wins on {wins} of {len(per_layer)} layer transitions.")

    json.dump({"means": {k: float(np.mean(v)) for k, v in agg.items()},
               "test_registers": test_r, "val_register": val_r, "n_comp": N_COMP,
               "pca_variance": var, "per_layer": per_layer},
              open(OUT, "w"), indent=1, default=float)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
