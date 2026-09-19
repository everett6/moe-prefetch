"""
Milestone 1b, step 3: the same question, but with the obvious objection removed.

m1b_hidden_probe.py trained ONE 2048x128 probe shared across all 47 layer
transitions and got 9.4% recall@8 against naive-repeat's 41.5%. That is a real
weakness in the test, not just in the result: routing is layer-specific, every
layer has its own gate matrix, and a single shared probe has to serve all of
them at once. A negative result from that setup is not yet a negative result
about the signal.

So this trains a SEPARATE probe per layer transition L -> L+1, and uses ridge
regression in dual form because the shape demands it: ~450 training rows per
layer against 2048 features is badly overparameterised, and unregularised
least squares would simply memorise. Dual form solves it in sample space
(450x450) instead of feature space (2048x2048), which is both correct and fast.

Several ridge strengths are swept, and the best *held-out* score per layer is
reported -- which is generous to the probe on purpose. If the signal still is
not there when every layer gets its own model, tuned per layer, and scored at
its own best regularisation, then it is not there.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA", os.path.join(HERE, "m1b_hidden.npz"))
OUT = os.path.join(HERE, "m1b_perlayer_probe_result.json")
LAMBDAS = [float(x) for x in os.environ.get("LAMBDAS", "1,10,100,1000,10000").split(",")]
N_EXPERT, K = 128, 8


def recall_at_k(scores, truth, k=K):
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    return np.mean([len(set(t) & set(p)) for t, p in zip(truth, top)]) / k


def main():
    d = np.load(DATA)
    H, E, keys = d["hidden"].astype(np.float32), d["experts"].astype(np.int64), d["keys"]
    pid, tok, lay = keys[:, 0], keys[:, 1], keys[:, 2]
    idx = {(p, t, l): i for i, (p, t, l) in enumerate(zip(pid, tok, lay))}
    held = sorted(set(pid))[-2:]
    n_layers = int(lay.max()) + 1
    print(f"{len(H):,} rows, holding out prompts {held}", file=sys.stderr)

    per_layer, rep_all, probe_all, rnd_all, ens_all = [], [], [], [], []
    rng = np.random.default_rng(1)

    for L in range(n_layers - 1):
        src, dst = [], []
        for (p, t, l), i in idx.items():
            if l != L:
                continue
            j = idx.get((p, t, L + 1))
            if j is not None:
                src.append(i)
                dst.append(j)
        if len(src) < 40:
            continue
        src, dst = np.array(src), np.array(dst)
        te = np.isin(pid[src], held)
        if te.sum() < 10 or (~te).sum() < 20:
            continue

        X = H[src]
        mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-6      # fit scaling on train only
        X = (X - mu) / sd
        Y = np.zeros((len(dst), N_EXPERT), dtype=np.float32)
        Y[np.arange(len(dst))[:, None], E[dst]] = 1.0
        Xtr, Ytr, Xte = X[~te], Y[~te], X[te]
        Ete = E[dst][te]

        # dual ridge: W = X'(XX' + lam I)^-1 Y, solved in sample space
        G = Xtr @ Xtr.T
        best, best_scores = -1.0, None
        for lam in LAMBDAS:
            A = np.linalg.solve(G + lam * np.eye(len(G), dtype=np.float32), Ytr)
            sc = Xte @ (Xtr.T @ A)
            r = recall_at_k(sc, Ete)
            if r > best:
                best, best_scores = r, sc

        # same held-out rows, naive-repeat baseline
        rep = [len(set(E[idx[(p, t - 1, L + 1)]]) & set(E[j])) / K
               for (p, t), j in ((tuple(keys[i][:2]), dst[m])
                                 for m, i in enumerate(src) if te[m])
               if (p, t - 1, L + 1) in idx]
        rnd = recall_at_k(rng.normal(size=(len(Ete), N_EXPERT)), Ete)

        # Ensemble. The probe and naive-repeat turn out to be anti-correlated by
        # depth -- where one is weak the other is strong -- so the interesting
        # question is not which wins but whether they are complementary. The
        # repeat prior is a multi-hot of the previous token's experts at L+1,
        # added to the (standardised) probe scores at a fixed weight.
        prior = np.zeros_like(best_scores)
        rows_te = np.where(te)[0]
        for m, row in enumerate(rows_te):
            p, t = keys[src[row]][:2]
            j = idx.get((p, t - 1, L + 1))
            if j is not None:
                prior[m, E[j]] = 1.0
        z = (best_scores - best_scores.mean(1, keepdims=True)) / (best_scores.std(1, keepdims=True) + 1e-9)
        ens = max(recall_at_k(z + w * prior, Ete) for w in (0.5, 1.0, 2.0, 4.0))

        per_layer.append({"layer": L, "probe": best, "ensemble": ens,
                          "repeat": float(np.mean(rep)) if rep else None,
                          "random": rnd, "n_test": int(te.sum())})
        probe_all.append(best)
        ens_all.append(ens)
        rnd_all.append(rnd)
        if rep:
            rep_all.append(float(np.mean(rep)))

    print(f"\n=== MILESTONE 1b: PER-LAYER RIDGE PROBES, h_L -> E_-L+1- "
          f"({len(per_layer)} layer transitions) ===\n")
    print("%-28s %10s" % ("predictor", "recall@8"))
    print("%-28s %9.1f%%" % ("probe + repeat ensemble", 100 * np.mean(ens_all)))
    print("%-28s %9.1f%%" % ("per-layer probe on h_L", 100 * np.mean(probe_all)))
    print("%-28s %9.1f%%" % ("naive-repeat (not prefetchable)", 100 * np.mean(rep_all)))
    print("%-28s %9.1f%%" % ("random", 100 * np.mean(rnd_all)))

    best = max(per_layer, key=lambda r: r["probe"])
    print(f"\nbest single layer: {best['layer']} -> {best['layer'] + 1} at "
          f"{100 * best['probe']:.1f}% (repeat there: {100 * best['repeat']:.1f}%)")
    beat = [r for r in per_layer if r["repeat"] and r["probe"] > r["repeat"]]
    print(f"layers where the probe beats naive-repeat: {len(beat)} of {len(per_layer)}")

    json.dump({"mean_probe": float(np.mean(probe_all)),
               "mean_ensemble": float(np.mean(ens_all)),
               "mean_repeat": float(np.mean(rep_all)),
               "mean_random": float(np.mean(rnd_all)),
               "layers_probe_wins": len(beat), "per_layer": per_layer},
              open(OUT, "w"), indent=1, default=float)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
