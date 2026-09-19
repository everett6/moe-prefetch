"""
B1/B2: does letting the predictor learn from its own mistakes actually help?

This is the half of the original brief that was never built. Everything so far
fits the probe once, offline, on five registers and freezes it. But the predictor
is wrong about ~28% of layers on every token, and the truth arrives microseconds
later when the layer's router runs. That is a free, perfectly labelled training
signal being discarded 47 times per token.

Two update rules, because they cost very different amounts:

  RLS   recursive least squares -- the exact incremental form of the ridge
        regression already being used, so the online model is identical to
        refitting on all data seen so far. O(d^2) per update: with d = 512
        features that is ~262K multiply-adds per layer, ~12M per token.
  LMS   plain gradient step, W += eta * x (y - W'x)'. O(d * 128) = 65K per
        layer, ~3M per token. Approximate, ~4x cheaper.

The honest test is **distribution shift**, because that is the only place online
adaptation can win. A probe fitted on five registers and run on `science` and
`technical` is seeing genuinely new material; if learning on the fly does not
help there, it will not help anywhere. Both rules start from the same frozen
probe and see the held-out stream in token order, exactly once -- prediction
always precedes the update for that row, so nothing is scored on data it has
already learned from.

Reported as a curve over the stream as well as an average, because "improves as
it goes" is the actual claim being tested, and an average can hide it.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA", os.path.join(HERE, "m2_hidden.npz"))
OUT = os.path.join(HERE, "b1_online_probe_result.json")
N_COMP = int(os.environ.get("N_COMP", "256"))
LAM = float(os.environ.get("LAM", "100"))
ETA = float(os.environ.get("ETA", "0.02"))
N_EXPERT, K = 128, 8


def recall(scores, truth, k=K):
    top = np.argpartition(-scores, k)[:k]
    return len(set(top) & set(truth)) / k


def main():
    d = np.load(DATA, allow_pickle=True)
    H = d["hidden"].astype(np.float32)
    E = d["experts"].astype(np.int64)
    keys, reg = d["keys"], d["register"].astype(str)
    idx = {(p, t, l): i for i, (p, t, l) in enumerate(zip(keys[:, 0], keys[:, 1], keys[:, 2]))}
    n_layers = int(keys[:, 2].max()) + 1
    regs = sorted(set(reg))
    train_r, test_r = regs[:5], regs[6:]
    tr_rows = np.isin(reg, train_r)
    print(f"train {train_r} -> online on {test_r}", file=sys.stderr)

    # A2 feature space: normalised hidden state, PCA fitted on train rows only
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-6)
    mu = Hn[tr_rows].mean(0)
    C = ((Hn[tr_rows] - mu).T @ (Hn[tr_rows] - mu)) / max(tr_rows.sum() - 1, 1)
    w, V = np.linalg.eigh(C.astype(np.float64))
    comp = V[:, ::-1][:, :N_COMP].astype(np.float32)
    P = (Hn - mu) @ comp
    P /= (P[tr_rows].std(0) + 1e-6)

    def featrow(i, L):
        prev = np.zeros(N_EXPERT, dtype=np.float32)
        p, t = keys[i][:2]
        j = idx.get((p, t - 1, L + 1))
        if j is not None:
            prev[E[j]] = 1.0
        cur = np.zeros(N_EXPERT, dtype=np.float32)
        cur[E[i]] = 1.0
        return np.concatenate([P[i], prev, cur])

    D = N_COMP + 2 * N_EXPERT

    # --- fit the frozen starting probe on the training registers
    W0, P0 = {}, {}
    for L in range(n_layers - 1):
        rows = [i for (p, t, l), i in idx.items()
                if l == L and (p, t, L + 1) in idx and reg[i] in train_r]
        if len(rows) < 50:
            continue
        X = np.stack([featrow(i, L) for i in rows])
        Y = np.zeros((len(rows), N_EXPERT), dtype=np.float32)
        Y[np.arange(len(rows))[:, None], E[[idx[(keys[i][0], keys[i][1], L + 1)] for i in rows]]] = 1.0
        G = X.T @ X + LAM * np.eye(D, dtype=np.float32)
        W0[L] = np.linalg.solve(G, X.T @ Y)
        P0[L] = np.linalg.inv(G).astype(np.float32)      # RLS covariance, seeded
    print(f"frozen probe fitted for {len(W0)} layers ({D} features)", file=sys.stderr)

    # --- stream the held-out registers in token order
    stream = []
    for i in np.where(np.isin(reg, test_r))[0]:
        p, t, l = keys[i]
        if (p, t, l + 1) in idx and l in W0:
            stream.append((int(p), int(t), int(l), i, idx[(p, t, l + 1)]))
    stream.sort(key=lambda r: (r[0], r[1], r[2]))
    print(f"{len(stream):,} held-out (token, layer) steps", file=sys.stderr)

    modes = {"frozen": None, "lms": "lms", "rls": "rls"}
    results = {}
    for name, rule in modes.items():
        W = {L: W0[L].copy() for L in W0}
        Pc = {L: P0[L].copy() for L in P0} if rule == "rls" else None
        scores, t0 = [], time.perf_counter()
        for (p, t, L, i, j) in stream:
            x = featrow(i, L)
            s = x @ W[L]
            scores.append(recall(s, E[j]))
            if rule is None:
                continue
            y = np.zeros(N_EXPERT, dtype=np.float32)
            y[E[j]] = 1.0
            err = y - s
            if rule == "lms":
                W[L] += ETA * np.outer(x, err)
            else:
                Px = Pc[L] @ x
                k = Px / (1.0 + float(x @ Px))
                W[L] += np.outer(k, err)
                Pc[L] -= np.outer(k, Px)
        el = time.perf_counter() - t0
        sc = np.array(scores)
        n = len(sc)
        results[name] = {
            "mean": float(sc.mean()),
            "first_quarter": float(sc[: n // 4].mean()),
            "last_quarter": float(sc[-(n // 4):].mean()),
            "us_per_step": 1e6 * el / n,
        }
        print(f"  {name:<7} mean {100 * sc.mean():.1f}%  "
              f"first-quarter {100 * sc[:n // 4].mean():.1f}%  "
              f"last-quarter {100 * sc[-(n // 4):].mean():.1f}%  "
              f"{1e6 * el / n:.0f} us/step", file=sys.stderr, flush=True)

    print(f"\n=== B2: ONLINE ADAPTATION ON HELD-OUT REGISTERS {test_r} "
          f"({len(stream):,} steps) ===\n")
    print("%-10s %9s %14s %14s %12s" % ("rule", "recall@8", "first quarter",
                                        "last quarter", "cost/step"))
    for name, r in results.items():
        print("%-10s %8.1f%% %13.1f%% %13.1f%% %10.0f us"
              % (name, 100 * r["mean"], 100 * r["first_quarter"],
                 100 * r["last_quarter"], r["us_per_step"]))

    fz = results["frozen"]
    print(f"\nFrozen drift over the stream: {100 * (fz['last_quarter'] - fz['first_quarter']):+.1f} "
          "points (this is the workload changing, not learning)")
    for name in ("lms", "rls"):
        r = results[name]
        print(f"{name}: {100 * (r['mean'] - fz['mean']):+.1f} points on average, "
              f"{100 * (r['last_quarter'] - fz['last_quarter']):+.1f} by the last quarter")

    best = max(("lms", "rls"), key=lambda k: results[k]["mean"])
    gain = results[best]["mean"] - fz["mean"]
    print(f"\nVerdict: online adaptation is worth {100 * gain:+.1f} points "
          f"({best}). " + ("Phase B is alive." if gain > 0.005 else
                           "Not enough to justify the per-token cost -- Phase B stops here."))

    json.dump({"test_registers": test_r, "steps": len(stream), "d": D,
               "lam": LAM, "eta": ETA, "results": results},
              open(OUT, "w"), indent=1, default=float)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
