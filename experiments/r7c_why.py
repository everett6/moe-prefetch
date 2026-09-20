"""
R7c: why a 99.7%-precision prefetcher is worth nothing.

R7b is the awkward result. A confidence-gated policy reaches 99.7% precision and
buys +0.0 tok/s, against an oracle worth +11.0 on the same trace. Precision that
high is not a broken model, so the explanation has to be about WHICH experts it
is right about.

The hypothesis: the model is confident exactly where confidence is worthless.
Its strongest feature is the previous token's experts, and an expert the last
token used is still in the cache -- so its confident predictions are for things
already resident, which cost nothing to "prefetch" and save nothing. The experts
that actually miss are the novel ones, and novelty is precisely what a
repeat-driven model cannot see coming.

If that is right, then splitting the true next-token experts by residency should
show a large gap in both recall and assigned probability. If it is wrong -- if
the model predicts non-resident experts nearly as well -- then R7b's result has
some other cause and the hypothesis should be dropped.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import models_deep as M  # noqa: E402
from train_deep import Index  # noqa: E402
from prefetch_env import PrefetchEnv  # noqa: E402
from r5b_phase import phase_mask  # noqa: E402

ART = os.path.join(ROOT, "artifacts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(ROOT, "data", "index-v4-replay.npz"))
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus-v4-replay"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ix = Index(args.index)
    d = ix.d
    rows = ix.rows({"train": 0, "val": 1, "test": 2}[args.split])
    rows = rows[np.lexsort((d["layer"][rows], d["pos"][rows], d["prompt"][rows]))]
    dec = phase_mask(ix, args.corpus)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = M.build(ck["cfg"]["arch"], ix.n_layers, r=ck["cfg"].get("r", 64),
                    hidden=ck["cfg"].get("hidden", 128),
                    context=ck["cfg"].get("context", True)).to(device).eval()
    model.load_state_dict(ck["state"])

    sc = np.zeros((len(rows), M.N_EXPERT), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(rows), 16384):
            idx = rows[s:s + 16384]
            b, _, _ = ix.batch(idx, device)
            sc[s:s + len(idx)] = torch.sigmoid(model(b)[0]).cpu().numpy()

    env = PrefetchEnv(capacity=66, n_layers=ix.n_layers)
    agg = {"resident": {"n": 0, "in_top8": 0, "prob": 0.0},
           "absent": {"n": 0, "in_top8": 0, "prob": 0.0}}
    last = None
    for n, r in enumerate(rows):
        L = int(d["layer"][r]); nxt = L + 1
        key = (int(d["prompt"][r]), int(d["pos"][r]))
        if key != last:
            if last is not None:
                env.step_boundary()
            last = key
        if dec[r] and len(env.resident[nxt]) >= 0.9 * 66:
            p = sc[n]
            top8 = set(int(e) for e in np.argpartition(-p, 8)[:8])
            for e in d["target"][r]:
                e = int(e)
                if e < 0:
                    continue
                g = "resident" if e in env.resident[nxt] else "absent"
                agg[g]["n"] += 1
                agg[g]["in_top8"] += 1 if e in top8 else 0
                agg[g]["prob"] += float(p[e])
        cur = [int(e) for e in d["cur"][r] if e >= 0]
        env.observe(L, cur)
        env.admit_demand(L, cur)

    print("\n=== R7c: WHICH EXPERTS IS THE MODEL RIGHT ABOUT? ===\n")
    print("true next-token experts, warm decode rows, split by cache residency\n")
    print(f"{'':12}{'count':>12}{'share':>9}{'recall@8':>11}{'mean prob':>12}")
    print("-" * 56)
    out = {}
    tot = sum(agg[g]["n"] for g in agg) or 1
    for g in ("resident", "absent"):
        a = agg[g]
        n = max(a["n"], 1)
        out[g] = {"count": a["n"], "share": a["n"] / tot,
                  "recall8": a["in_top8"] / n, "mean_prob": a["prob"] / n}
        print(f"{g:12}{a['n']:>12,}{100*a['n']/tot:>8.1f}%"
              f"{100*out[g]['recall8']:>10.1f}%{out[g]['mean_prob']:>12.3f}")

    r, b = out["resident"], out["absent"]
    print(f"\nrecall gap: {100*(r['recall8']-b['recall8']):+.1f} points, "
          f"probability gap: {r['mean_prob']-b['mean_prob']:+.3f}")
    print("\nThe experts that are already cached cost nothing to fetch and save")
    print("nothing to predict. The absent ones are the entire opportunity.")
    if b["recall8"] < 0.5 * r["recall8"]:
        print("\nHypothesis holds: the model is confident precisely where")
        print("confidence is worthless, and blind where it would pay.")
    else:
        print("\nHypothesis does NOT hold. The model sees novelty perfectly well")
        print(f"-- {100*b['recall8']:.0f}% recall on absent experts against 6.25%")
        print("for chance. It is weaker there than on resident experts, but not")
        print("blind. R7b's result has another cause, and r7b now measures it:")
        print("a CORRECT prefetch still evicts, and in a 93%-effective cache the")
        print("evicted expert is usually needed again. The break-even formula")
        print("prices only the wrong prefetches, so it understates the real bar.")
    json.dump(out, open(os.path.join(ART, "r7c_why.json"), "w"), indent=1)
    print("\nwrote artifacts/r7c_why.json")


if __name__ == "__main__":
    main()
