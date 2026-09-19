"""
E5: the stopping criterion -- what would a PERFECT predictor be worth?

Before concluding that more data and more training cannot help, it is worth
knowing the ceiling they are aiming at. The fresh model reaches 61% precision on
non-resident experts against a 75% break-even, so the obvious question is how
much throughput is waiting on the other side of that gap.

This replays the same traces through the same engine model with an ORACLE: at
each layer it prefetches exactly the experts the next token will use there, and
nothing else. Precision 100%, recall 100%, zero wasted uploads. No predictor can
beat it, so whatever it is worth is the entire remaining prize for prediction --
and if that prize is small, the plateau is real rather than a failure of effort.

Reported at the measured upload cost and at reduced ones, so the two levers
(better prediction, cheaper uploads) can be compared on the same axis.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from prefetch_env import PrefetchEnv, fit_cost_model  # noqa: E402
from train_deep import Index  # noqa: E402


def replay(ix, rows, depth, cost, oracle_ids=None, capacity=66, max_inserts=2,
           pool_extra=3, max_tokens=1500):
    d = ix.d
    env = PrefetchEnv(capacity=capacity, max_inserts=max_inserts,
                      pool_extra=pool_extra, cost=cost, n_layers=ix.n_layers)
    last, ntok = None, 0
    for r in rows:
        L = int(d["layer"][r])
        key = (int(d["prompt"][r]), int(d["pos"][r]))
        if key != last:
            if last is not None:
                env.step_boundary()
                ntok += 1
                if ntok >= max_tokens:
                    break
            last = key
        if depth > 0 and oracle_ids is not None:
            want = oracle_ids.get((key[0], key[1] + 1, L + 1))
            if want is not None:
                env.prefetch(L + 1, [int(e) for e in want if e >= 0], depth,
                             min_keep=max_inserts)
        cur = [int(e) for e in d["cur"][r] if e >= 0]
        env.observe(L, cur)
        env.admit_demand(L, cur)
    tok, hit, up = env.tok_s(max(ntok, 1))
    return {"tok_s": tok, "hit": hit, "uploads_per_token": up, "n_tokens": ntok}


def main():
    # Parameterised so the ceiling can be recomputed on the real-prompt corpus.
    # The v3 answer was "-0.4 tok/s for a perfect predictor"; whether that holds
    # on real traffic is a question about the data, not about the arithmetic.
    index = os.environ.get("INDEX", os.path.join(ROOT, "data", "index-v3.npz"))
    if len(sys.argv) > 1:
        index = sys.argv[1]
    print(f"index: {index}")
    ix = Index(index)
    d = ix.d
    rows = ix.rows(1)
    rows = rows[np.lexsort((d["layer"][rows], d["pos"][rows], d["prompt"][rows]))]
    # (prompt, pos, layer) -> experts that layer uses on that token
    oracle = {(int(d["prompt"][r]), int(d["pos"][r]), int(d["layer"][r])): d["cur"][r]
              for r in rows}
    cost = fit_cost_model()
    C0 = cost["C_ms_per_upload"]

    print("\n=== E5: THE CEILING -- a perfect predictor, same engine ===\n")
    print("%14s %8s %11s %11s %11s %13s"
          % ("upload cost", "depth 0", "oracle d1", "oracle d2", "oracle d3", "best gain"))
    out = []
    for scale in (1.0, 0.5, 0.25, 0.0):
        c = dict(cost)
        c["C_ms_per_upload"] = C0 * scale
        base = replay(ix, rows, 0, c)
        oracles = {p: replay(ix, rows, p, c, oracle) for p in (1, 2, 3)}
        best = max(oracles.values(), key=lambda v: v["tok_s"])
        gain = best["tok_s"] - base["tok_s"]
        out.append({"upload_us": 1000 * C0 * scale, "depth0": base["tok_s"],
                    "oracle": {str(p): v["tok_s"] for p, v in oracles.items()},
                    "best_gain": gain})
        print("%12.1f us %8.1f %11.1f %11.1f %11.1f %+12.1f"
              % (1000 * C0 * scale, base["tok_s"], oracles[1]["tok_s"],
                 oracles[2]["tok_s"], oracles[3]["tok_s"], gain))

    at_measured = out[0]
    print(f"\nAt the measured upload cost, a PERFECT predictor is worth "
          f"{at_measured['best_gain']:+.1f} tok/s.")
    print("Our model reaches 61% precision against a 75% break-even; the gap above")
    print("is the entire prize for closing that, and it bounds what more data or a")
    print("better architecture could return.")
    json.dump({"cost_model": cost, "rows": out},
              open(os.path.join(ROOT, "artifacts", "oracle_ceiling.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
