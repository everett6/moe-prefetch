"""
A2: give the probe inputs it does not have, and measure what each one is worth.

A1 established that nonlinearity is not the bottleneck -- a properly trained MLP
ties linear ridge (53.2 vs 54.1) rather than beating it. So what is missing is
signal, not capacity. This adds signal.

The plan named `ffn_moe_logits-L` first. On reflection it is the weakest of the
candidates and it is not captured here, for a reason worth stating: the router's
logits at layer L are `W_L . RMSNorm(h_L)` -- a linear projection of the very
vector the probe already takes as input. A linear model can already represent
almost all of it. The only part it cannot is the normalisation, which is tested
below as `h_norm` without needing a re-capture.

The inputs that carry genuinely new information are ones already on disk:

  h            PCA(h_L), 256 dims. The milestone 2 baseline.
  h_norm       PCA(h_L / ||h_L||). What RMSNorm does before the gate sees it;
               strips the magnitude the router discards anyway.
  prev         multi-hot of E_{L+1}(t-1), 128 dims -- the previous token's
               experts at the layer being predicted. This is the naive-repeat
               signal, but as a FEATURE the model can weight per layer and mix
               with the hidden state, instead of a scalar-weighted prior bolted
               on after the fact.
  cur          multi-hot of E_L(t), 128 dims -- the experts this layer just
               chose, known before L+1 runs. Alone it was worth 6.5% (milestone
               1); the question is whether it adds anything *given* h_L.
  delta        PCA(h_L(t) - h_L(t-1)), 256 dims -- where the residual stream is
               moving, not just where it is.

Each is tested as an addition to `h`, then the best combination, all with ridge
(A1 showed the model class does not matter here) and the same register-level
split: 5 train, 1 validate, 2 test. The repeat prior is still applied on top, so
the numbers are comparable with A1's 60.5%.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA", os.path.join(HERE, "m2_hidden.npz"))
OUT = os.path.join(HERE, "a2_richer_inputs_result.json")
N_COMP = int(os.environ.get("N_COMP", "256"))
LAMBDAS = [1, 10, 100, 1000, 10000]
WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
N_EXPERT, K = 128, 8


def recall_at_k(scores, truth, k=K):
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    return np.mean([len(set(t) & set(p)) for t, p in zip(truth, top)]) / k


def zs(a):
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)


def pca(H, rows, n_comp):
    mu = H[rows].mean(0)
    Xc = H[rows] - mu
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w, V = np.linalg.eigh(C.astype(np.float64))
    comp = V[:, ::-1][:, :n_comp].astype(np.float32)
    P = (H - mu) @ comp
    return P / (P[rows].std(0) + 1e-6)


def main():
    d = np.load(DATA, allow_pickle=True)
    H = d["hidden"].astype(np.float32)
    E = d["experts"].astype(np.int64)
    keys, reg = d["keys"], d["register"].astype(str)
    idx = {(p, t, l): i for i, (p, t, l) in enumerate(zip(keys[:, 0], keys[:, 1], keys[:, 2]))}
    n_layers = int(keys[:, 2].max()) + 1
    regs = sorted(set(reg))
    train_r, val_r, test_r = regs[:5], regs[5:6], regs[6:]
    tr_rows = np.isin(reg, train_r)
    print(f"{len(H):,} rows; train {train_r} val {val_r} test {test_r}", file=sys.stderr)

    Ph = pca(H, tr_rows, N_COMP)
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-6)
    Pn = pca(Hn, tr_rows, N_COMP)
    # trajectory: h_L(t) - h_L(t-1), zero where there is no previous token
    D = np.zeros_like(H)
    for (p, t, l), i in idx.items():
        j = idx.get((p, t - 1, l))
        if j is not None:
            D[i] = H[i] - H[j]
    Pd = pca(D, tr_rows, N_COMP)
    print("built h, h_norm, delta feature spaces", file=sys.stderr)

    SETS = ["h", "h+prev", "h+cur", "h+delta", "h_norm", "h+prev+cur",
            "h+prev+cur+delta", "h_norm+prev+cur"]
    agg = {s: [] for s in SETS}
    agg["repeat"] = []

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

        Y = np.zeros((len(dst), N_EXPERT), dtype=np.float32)
        Y[np.arange(len(dst))[:, None], E[dst]] = 1.0
        Eva, Ete = E[dst][m_va], E[dst][m_te]

        # feature blocks, aligned on `src`
        prev = np.zeros((len(src), N_EXPERT), dtype=np.float32)
        for m, i in enumerate(src):
            p, t = keys[i][:2]
            j = idx.get((p, t - 1, L + 1))
            if j is not None:
                prev[m, E[j]] = 1.0
        cur = np.zeros((len(src), N_EXPERT), dtype=np.float32)
        cur[np.arange(len(src))[:, None], E[src]] = 1.0

        blocks = {"h": Ph[src], "h_norm": Pn[src], "prev": prev, "cur": cur,
                  "delta": Pd[src]}

        def run(names):
            X = np.concatenate([blocks[n] for n in names], axis=1)
            Xtr, Ytr = X[m_tr], Y[m_tr]
            G = Xtr.T @ Xtr
            best_v, Wd = -1.0, None
            for lam in LAMBDAS:
                W = np.linalg.solve(G + lam * np.eye(X.shape[1], dtype=np.float32),
                                    Xtr.T @ Ytr)
                v = recall_at_k(X[m_va] @ W, Eva)
                if v > best_v:
                    best_v, Wd = v, W
            sva, ste = zs(X[m_va] @ Wd), zs(X[m_te] @ Wd)
            bw = max(WEIGHTS, key=lambda w: recall_at_k(sva + w * prev[m_va], Eva))
            return recall_at_k(ste + bw * prev[m_te], Ete)

        for s in SETS:
            agg[s].append(run(s.split("+")))
        agg["repeat"].append(recall_at_k(
            prev[m_te] + 1e-6 * np.random.randn(m_te.sum(), N_EXPERT).astype(np.float32), Ete))
        if L % 12 == 0:
            print(f"  layer {L:2d}: h {100 * agg['h'][-1]:.1f}  "
                  f"h+prev+cur {100 * agg['h+prev+cur'][-1]:.1f}", file=sys.stderr, flush=True)

    base = np.mean(agg["h"])
    print(f"\n=== A2: RICHER INPUTS ({len(agg['h'])} layers, test {test_r}; "
          f"all with the repeat prior on top) ===\n")
    print("%-22s %10s %10s" % ("features", "recall@8", "vs h"))
    for s in SETS:
        v = np.mean(agg[s])
        print("%-22s %9.1f%% %9.1f" % (s, 100 * v, 100 * (v - base)))
    print("%-22s %9.1f%% %9.1f" % ("naive-repeat", 100 * np.mean(agg["repeat"]),
                                   100 * (np.mean(agg["repeat"]) - base)))

    best_set = max(SETS, key=lambda s: np.mean(agg[s]))
    a1_best = 0.605
    print(f"\nBest: {best_set} at {100 * np.mean(agg[best_set]):.1f}% "
          f"({100 * (np.mean(agg[best_set]) - a1_best):+.1f} vs A1's 60.5%)")

    json.dump({"means": {k: float(np.mean(v)) for k, v in agg.items()},
               "best_set": best_set, "a1_best": a1_best}, open(OUT, "w"),
              indent=1, default=float)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
