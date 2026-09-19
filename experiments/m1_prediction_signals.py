"""
Milestone 1a: which prediction signal actually has lookahead?

This exists because the first draft of this repo's README proposed predicting
the next TOKEN and running it through the real router to get exact expert IDs.
That has a dependency flaw worth stating plainly, because it invalidates the
mechanism rather than merely weakening it:

  A layer's router takes that layer's hidden state, h_L(t). To know which
  experts token t+1 needs at layer L, you need h_L(t+1) -- which requires
  having already run layers 0..L-1 on token t+1. Knowing the token's *identity*
  early does not give you its hidden states. So "predict the token, then ask the
  router" cannot run ahead of the computation it is meant to prefetch for.

What is left, and what the literature actually does (Pre-gated MoE; FATE), is
prediction ACROSS LAYERS WITHIN a token: while layer L is computing, predict
what layer L+1 will select, and fetch it. That direction has genuine slack --
one layer of compute, ~0.27 ms at Q4_K_M's 77.9 tok/s, against a measured
0.255 ms/layer transfer -- and it needs no future hidden state, only the
current one.

So before building anything this measures the three candidate signals against
each other on AI2's existing 255,360-row Q4_K_M trace, using recall@8: of the 8
experts layer L+1 truly selects, how many does the prediction contain?

  same-layer, previous token    E_L(t)    -> E_L(t+1)    "naive-repeat", the
                                                         47.8% bar every trained
                                                         predictor in AI2 lost to
  cross-layer, same token       E_L(t)    -> E_{L+1}(t)  the prefetchable one
  cross-layer, previous token   E_{L+1}(t)-> E_{L+1}(t+1) same-layer again, but
                                                         reported per layer-pair
                                                         so the two are comparable
  random                        8 of 128 at random       the 6.2% floor

If cross-layer overlap is near the random floor, prefetching one layer ahead has
nothing to predict from and this project needs a different mechanism (a trained
cross-layer predictor, as in Pre-gated MoE, which requires touching the model).
If it is well above the floor, milestone 1b trains one.
"""
import json
import os
import random
import statistics
import sys
from collections import defaultdict

TRACE = os.environ.get("TRACE", "/home/everett/AI2/experiments/expert_trace.jsonl")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m1_prediction_signals_result.json")


def load(path):
    """-> {(prompt_id, token_pos): {layer: [expert ids]}}"""
    seqs = defaultdict(dict)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            seqs[(r["prompt_id"], r["token_pos"])][r["layer"]] = r["experts"]
    return seqs


def recall(pred, truth):
    return len(set(pred) & set(truth)) / len(truth) if truth else 0.0


def main():
    if not os.path.exists(TRACE):
        print(f"trace not found: {TRACE}", file=sys.stderr)
        sys.exit(2)
    seqs = load(TRACE)
    keys = sorted(seqs)
    print(f"loaded {len(keys):,} (prompt, token) positions from {os.path.basename(TRACE)}",
          file=sys.stderr)

    layers = sorted({l for v in seqs.values() for l in v})
    n_layers, k = len(layers), len(next(iter(seqs.values()))[layers[0]])
    n_experts = 1 + max(e for v in seqs.values() for ids in v.values() for e in ids)
    print(f"{n_layers} layers, top-{k} of {n_experts} experts", file=sys.stderr)

    scores = {"same_layer_prev_token": [], "cross_layer_same_token": [], "random": []}
    per_layer = defaultdict(lambda: {"cross": [], "same": []})
    rng = random.Random(0)

    for (pid, pos) in keys:
        cur = seqs[(pid, pos)]
        nxt = seqs.get((pid, pos + 1))
        for L in layers:
            if L not in cur:
                continue
            # cross-layer, same token: does layer L predict layer L+1?
            if (L + 1) in cur:
                r = recall(cur[L], cur[L + 1])
                scores["cross_layer_same_token"].append(r)
                per_layer[L]["cross"].append(r)
            # same-layer, next token: the naive-repeat bar
            if nxt and L in nxt:
                r = recall(cur[L], nxt[L])
                scores["same_layer_prev_token"].append(r)
                per_layer[L]["same"].append(r)
                scores["random"].append(recall(rng.sample(range(n_experts), k), nxt[L]))

    print("\n=== MILESTONE 1a: WHICH SIGNAL HAS LOOKAHEAD? "
          f"(Q4_K_M, {len(scores['same_layer_prev_token']):,} evaluated pairs) ===\n")
    print("%-34s %10s   %s" % ("signal", "recall@8", "can it prefetch?"))
    rows = [
        ("same-layer, previous token", "same_layer_prev_token",
         "NO -- needs h_L(t+1), which is the computation itself"),
        ("cross-layer, same token", "cross_layer_same_token",
         "YES -- h_L is in hand one layer early"),
        ("random (8 of %d)" % n_experts, "random", "-- floor"),
    ]
    out = {}
    for label, key, note in rows:
        v = statistics.mean(scores[key]) if scores[key] else 0.0
        out[key] = v
        print("%-34s %9.1f%%   %s" % (label, 100 * v, note))

    print("\nper-layer cross-layer overlap (the prefetchable signal):")
    xs = [(L, statistics.mean(d["cross"])) for L, d in sorted(per_layer.items()) if d["cross"]]
    for L, v in xs[:4] + [("...", None)] + xs[-4:]:
        print(f"   layer {L:>3} -> {L if L == '...' else L + 1:<3}  " +
              ("" if v is None else f"{100 * v:.1f}%"))
    if xs:
        best = max(xs, key=lambda t: t[1])
        out["cross_layer_best"] = {"layer": best[0], "recall": best[1]}
        print(f"\nbest layer pair: {best[0]} -> {best[0] + 1} at {100 * best[1]:.1f}%")

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
