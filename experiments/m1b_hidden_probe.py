"""
Milestone 1b, step 2: can the hidden state at layer L predict layer L+1's experts?

This is the decisive question for the whole repo. m1_prediction_signals.py showed
the two cheap signals are each useless for a different reason:

  same-layer, previous token   45.8%   informative, but needs h_L(t+1) -- which
                                       IS the computation being prefetched for
  cross-layer, same token       6.5%   prefetchable, but at the 6.2% random floor

The one route left is the one Pre-gated MoE takes: predict E_{L+1} from the
hidden state h_L rather than from the 8 expert IDs. h_L is 2048 floats and is in
hand a full layer early, so if the signal is there, it is both informative *and*
prefetchable. If it is not there, predictive prefetching is closed on this model
and only cross-token LRU (i.e. what llama.cpp PR #27861 already does) remains.

Three predictors, on 26,667 captured (token, layer) rows from Q4_K_M:

  repeat-same-layer   E_L(t) -> E_L(t+1). The 45.8% bar. NOT prefetchable;
                      carried here only so every number sits in one table.
  linear probe        h_L -> 128 logits -> top-8, trained. The real question.
  random              the 6.2% floor.

Split is BY PROMPT, not by row: rows from one prompt share a context and leak
into each other badly. Two prompts are held out entirely.

Trained with numpy only -- multinomial logistic regression by gradient descent
on a 2048x128 weight matrix. Torch is available but unnecessary at this size,
and keeping the dependency out means this runs anywhere the capture does.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA", os.path.join(HERE, "m1b_hidden.npz"))
OUT = os.path.join(HERE, "m1b_hidden_probe_result.json")
EPOCHS = int(os.environ.get("EPOCHS", "60"))
LR = float(os.environ.get("LR", "0.5"))
N_EXPERT, K = 128, 8


def recall_at_k(scores, truth, k=K):
    """scores: (n, 128) -- fraction of the k true experts inside the top-k."""
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    hits = np.array([len(set(t) & set(p)) for t, p in zip(truth, top)])
    return hits.mean() / k


def main():
    if not os.path.exists(DATA):
        print(f"missing {DATA} -- run m1b_capture_hidden.py first", file=sys.stderr)
        sys.exit(2)
    d = np.load(DATA)
    H, E, K_ = d["hidden"].astype(np.float32), d["experts"].astype(np.int64), d["keys"]
    pid, tok, lay = K_[:, 0], K_[:, 1], K_[:, 2]
    n_layers = int(lay.max()) + 1
    print(f"{len(H):,} rows, {len(set(pid))} prompts, {n_layers} layers", file=sys.stderr)

    # index by (prompt, token, layer) so we can line up L -> L+1 and t -> t+1
    idx = {(p, t, l): i for i, (p, t, l) in enumerate(zip(pid, tok, lay))}

    # --- pairs for the cross-layer probe: h at (p,t,L) predicts experts at (p,t,L+1)
    src, dst = [], []
    for (p, t, l), i in idx.items():
        j = idx.get((p, t, l + 1))
        if j is not None:
            src.append(i)
            dst.append(j)
    src, dst = np.array(src), np.array(dst)
    held = sorted(set(pid))[-2:]                      # two prompts held out entirely
    is_test = np.isin(pid[src], held)
    print(f"{len(src):,} cross-layer pairs; holding out prompts {held} "
          f"({is_test.sum():,} test)", file=sys.stderr)

    X = H[src]
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)           # standardise on all rows
    Y = np.zeros((len(dst), N_EXPERT), dtype=np.float32)
    Y[np.arange(len(dst))[:, None], E[dst]] = 1.0 / K   # multi-hot, normalised
    Xtr, Ytr, Xte, Yte = X[~is_test], Y[~is_test], X[is_test], Y[is_test]
    Ete = E[dst][is_test]

    # --- linear probe, plain gradient descent on cross-entropy
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.01, (X.shape[1], N_EXPERT)).astype(np.float32)
    b = np.zeros(N_EXPERT, dtype=np.float32)
    for ep in range(EPOCHS):
        logits = Xtr @ W + b
        logits -= logits.max(1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(1, keepdims=True)
        G = (P - Ytr) / len(Xtr)
        W -= LR * (Xtr.T @ G)
        b -= LR * G.sum(0)
        if ep % 20 == 0 or ep == EPOCHS - 1:
            loss = -(Ytr * np.log(P + 1e-9)).sum(1).mean()
            print(f"  epoch {ep:3d} loss {loss:.4f}", file=sys.stderr, flush=True)
    probe = recall_at_k(Xte @ W + b, Ete)

    # --- baselines, on exactly the same held-out rows
    # naive-repeat, same layer, previous token: informative but not prefetchable
    rep_hits = []
    for i in np.where(is_test)[0]:
        p, t, l = K_[src[i]]
        prev = idx.get((p, t - 1, l + 1))
        if prev is not None:
            rep_hits.append(len(set(E[prev]) & set(E[dst[i]])) / K)
    repeat = float(np.mean(rep_hits)) if rep_hits else float("nan")

    # cross-layer expert IDs, the signal m1_prediction_signals.py measured
    xl = np.mean([len(set(E[src[i]]) & set(E[dst[i]])) / K for i in np.where(is_test)[0]])

    rand_scores = np.random.default_rng(1).normal(size=(len(Ete), N_EXPERT))
    rnd = recall_at_k(rand_scores, Ete)

    print("\n=== MILESTONE 1b: DOES h_L PREDICT E_{L+1}? "
          f"({len(Ete):,} held-out rows, 2 unseen prompts) ===\n")
    print("%-40s %10s   %s" % ("predictor", "recall@8", "prefetchable?"))
    rows = [
        ("linear probe on h_L (2048 -> 128)", probe, "YES -- h_L is one layer early"),
        ("naive-repeat, same layer, prev token", repeat, "no -- needs h_L(t+1)"),
        ("cross-layer expert IDs", xl, "yes, but no signal"),
        ("random", rnd, "-- floor"),
    ]
    for label, v, note in rows:
        print("%-40s %9.1f%%   %s" % (label, 100 * v, note))

    verdict = ("BEATS the naive-repeat bar -- predictive prefetching is viable"
               if probe > repeat else
               "does NOT beat naive-repeat -- no trained predictor is justified here")
    print(f"\nVerdict: the hidden-state probe {verdict}.")
    json.dump({"probe": probe, "repeat": repeat, "cross_layer_ids": float(xl),
               "random": rnd, "n_test": int(len(Ete)), "held_out_prompts": held},
              open(OUT, "w"), indent=1, default=float)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
