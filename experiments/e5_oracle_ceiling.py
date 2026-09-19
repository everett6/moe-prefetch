"""
E5: the stopping criterion -- what would a PERFECT predictor be worth?

Before concluding that more data and more training cannot help, it is worth
knowing the ceiling they are aiming at: how much throughput a PERFECT predictor
would buy, in this engine, at the measured upload cost.

This prints only that ceiling. An earlier version also printed the then-current
model's precision as a hardcoded "61%", which went stale the moment a new model
was trained and would have been read as a live measurement. The precision of
whichever model is current comes from e2_precision.py, which measures it.

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
           pool_extra=3, max_tokens=int(os.environ.get("MAX_TOKENS", "1500")),
           prefetch_at=None):
    """`prefetch_at` gates where the oracle is ALLOWED to prefetch, without
    removing any rows from the replay.

    That distinction is the whole point. The predictor would only run at decode,
    so charging it for prefill uploads overstates its cost -- but simply dropping
    the prefill rows also drops the cache WARMING they do, and decode then starts
    against an empty cache that the real engine never sees. Gating the prefetch
    while replaying every row keeps the cache honest and the accounting fair."""
    d = ix.d
    env = PrefetchEnv(capacity=capacity, max_inserts=max_inserts,
                      pool_extra=pool_extra, cost=cost, n_layers=ix.n_layers)
    last, ntok = None, 0
    # Per-phase accounting. tok_s() pools every replayed position, so on a real
    # corpus -- where 80% of positions are prefill -- a decode-only gain is
    # diluted about fivefold and reads as nothing. Decode throughput is the
    # metric actually reported, so the decode steps are costed separately.
    dstats = {"hit": 0, "lookup": 0, "uploads": 0}
    dtok = 0
    prev = dict(env.stats)
    for r in rows:
        L = int(d["layer"][r])
        key = (int(d["prompt"][r]), int(d["pos"][r]))
        if key != last:
            if last is not None:
                env.step_boundary()
                ntok += 1
                if prefetch_at is None or prefetch_at.get(last, True):
                    for k in dstats:
                        dstats[k] += env.stats[k] - prev[k]
                    dtok += 1
                prev = dict(env.stats)
                if ntok >= max_tokens:
                    break
            last = key
        may_prefetch = prefetch_at is None or prefetch_at.get(key, True)
        if depth > 0 and oracle_ids is not None and may_prefetch:
            want = oracle_ids.get((key[0], key[1] + 1, L + 1))
            if want is not None:
                env.prefetch(L + 1, [int(e) for e in want if e >= 0], depth,
                             min_keep=max_inserts)
        cur = [int(e) for e in d["cur"][r] if e >= 0]
        env.observe(L, cur)
        env.admit_demand(L, cur)
    tok, hit, up = env.tok_s(max(ntok, 1))
    dhit = dstats["hit"] / max(dstats["lookup"], 1)
    dmissed = ix.n_layers * 8 * (1 - dhit)
    dup = dstats["uploads"] / max(dtok, 1)
    dms = (cost["A_ms"] + cost["B_ms_per_miss"] * dmissed
           + cost["C_ms_per_upload"] * dup)
    return {"tok_s": tok, "hit": hit, "uploads_per_token": up, "n_tokens": ntok,
            "decode_tok_s": 1000.0 / max(dms, 1e-6), "decode_hit": dhit,
            "decode_tokens": dtok, "decode_uploads_per_token": dup}


def main():
    # Parameterised so the ceiling can be recomputed on the real-prompt corpus.
    # The v3 answer was "-0.4 tok/s for a perfect predictor"; whether that holds
    # on real traffic is a question about the data, not about the arithmetic.
    index = os.environ.get("INDEX", os.path.join(ROOT, "data", "index-v3.npz"))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        index = args[0]
    decode_only = "--decode-only" in sys.argv
    print(f"index: {index}")
    ix = Index(index)
    d = ix.d
    rows = ix.rows(1)
    rows = rows[np.lexsort((d["layer"][rows], d["pos"][rows], d["prompt"][rows]))]
    # (prompt, pos, layer) -> experts that layer uses on that token
    oracle = {(int(d["prompt"][r]), int(d["pos"][r]), int(d["layer"][r])): d["cur"][r]
              for r in rows}
    prefetch_at = None
    if decode_only:
        # At any prefill batch of 32+ positions every layer uses all 128 experts
        # (artifacts/r6_prefill_union.json), so a prefetcher there has nothing to
        # choose between. Gate it off for prefill positions; the rows still
        # replay, so the cache still warms exactly as it does in the engine.
        sys.path.insert(0, HERE)
        from r5b_phase import phase_mask
        corpus = os.environ.get("REPLAY_CORPUS",
                                os.path.join(ROOT, "data", "corpus-v4-replay"))
        dec = phase_mask(ix, corpus)
        prefetch_at = {}
        for r in rows:
            prefetch_at[(int(d["prompt"][r]), int(d["pos"][r]))] = bool(dec[r])
        n_dec = sum(prefetch_at.values())
        print(f"decode-only prefetch: allowed at {n_dec:,} of "
              f"{len(prefetch_at):,} token positions (all rows still replayed)")

    cost = fit_cost_model()
    C0 = cost["C_ms_per_upload"]

    print("\n=== E5: THE CEILING -- a perfect predictor, same engine ===\n")
    print("%14s %8s %11s %11s %11s %13s"
          % ("upload cost", "depth 0", "oracle d1", "oracle d2", "oracle d3", "best gain"))
    out = []
    for scale in (1.0, 0.5, 0.25, 0.0):
        c = dict(cost)
        c["C_ms_per_upload"] = C0 * scale
        # prefetch_at must be passed to the BASELINE too. It gates prefetching,
        # which is a no-op at depth 0, but it also selects which tokens the
        # per-phase accounting attributes to decode. Without it the baseline is
        # costed over every token and the oracle over decode tokens only, which
        # compares two different things and manufactured a +92.7 tok/s "gain".
        base = replay(ix, rows, 0, c, prefetch_at=prefetch_at)
        oracles = {p: replay(ix, rows, p, c, oracle, prefetch_at=prefetch_at)
                   for p in (1, 2, 3)}
        best = max(oracles.values(), key=lambda v: v["tok_s"])
        gain = best["tok_s"] - base["tok_s"]
        dbest = max(oracles.values(), key=lambda v: v["decode_tok_s"])
        dgain = dbest["decode_tok_s"] - base["decode_tok_s"]
        out.append({"upload_us": 1000 * C0 * scale, "depth0": base["tok_s"],
                    "oracle": {str(p): v["tok_s"] for p, v in oracles.items()},
                    "best_gain": gain,
                    "decode_depth0": base["decode_tok_s"],
                    "decode_oracle": {str(p): v["decode_tok_s"]
                                      for p, v in oracles.items()},
                    "decode_best_gain": dgain,
                    "decode_tokens": base["decode_tokens"]})
        print("%12.1f us %8.1f %11.1f %11.1f %11.1f %+12.1f"
              % (1000 * C0 * scale, base["tok_s"], oracles[1]["tok_s"],
                 oracles[2]["tok_s"], oracles[3]["tok_s"], gain))

    print("\n   (decode tokens only -- the metric the benchmark reports)")
    print("%14s %8s %11s %11s %11s %13s"
          % ("upload cost", "depth 0", "oracle d1", "oracle d2", "oracle d3",
             "best gain"))
    for r in out:
        print("%11.1f us %8.1f %11.1f %11.1f %11.1f %13.1f"
              % (r["upload_us"], r["decode_depth0"], r["decode_oracle"]["1"],
                 r["decode_oracle"]["2"], r["decode_oracle"]["3"],
                 r["decode_best_gain"]))

    at_measured = out[0]
    print(f"\nAt the measured upload cost, a PERFECT predictor is worth "
          f"{at_measured['best_gain']:+.1f} tok/s over the whole trace, "
          f"{at_measured['decode_best_gain']:+.1f} tok/s on decode tokens "
          f"(from {at_measured['decode_tokens']:,} decode tokens).")
    print("The gap above is the entire prize for a better predictor, and it")
    print("bounds what more data or a better architecture could return.")
    # Named for what was measured. An earlier version wrote a fixed filename and
    # a v4 run silently destroyed the committed v3 result.
    tag = os.path.basename(index).replace(".npz", "")
    suffix = "-decode" if decode_only else ""
    json.dump({"cost_model": cost, "rows": out, "index": index,
               "decode_only": decode_only},
              open(os.path.join(ROOT, "artifacts",
                                f"oracle_ceiling_{tag}{suffix}.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
