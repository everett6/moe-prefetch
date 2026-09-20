"""
P1: is there room for a better prefetcher, and where is it?

R7b closed the previous design: a confidence-gated prefetcher loses throughput at
every precision the 75% break-even calls profitable, because a CORRECT prefetch
still evicts, and at 93.3% residency the evicted expert is usually needed again.
Eviction damage measured about three times the direct cost of being wrong.

That diagnosis implies where a better design would have to act, and this
measures whether the room is really there before anything is built. Four arms,
same engine, same real decode traces:

  LRU,    no prefetch    the engine as it ships
  LRU,    oracle prefetch  E5's ceiling: perfect admission, dumb eviction
  Belady, no prefetch    perfect eviction, no admission help at all
  Belady, oracle prefetch  both perfect -- the true ceiling of the whole design

Belady evicts the resident expert whose next use is furthest away. It is not
implementable (it reads the future) and that is the point: it bounds what any
eviction policy, learned or otherwise, could return.

The comparison that decides the next step is Belady-no-prefetch against
LRU-no-prefetch. If perfect eviction alone is worth a lot, a model that predicts
TIME TO NEXT USE is the thing to build, and admission is secondary. If it is
worth nothing, then the cache is already extracting what the trace allows and no
prefetcher of any kind will help.
"""
import bisect
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from train_deep import Index  # noqa: E402
from prefetch_env import PrefetchEnv, fit_cost_model  # noqa: E402
from r5b_phase import phase_mask  # noqa: E402

ART = os.path.join(ROOT, "artifacts")


def build_future(ix, rows):
    """For each layer, expert -> sorted token indices at which it is used.

    Token index, not row index: eviction competes over the lifetime of a token,
    and a layer is visited once per token."""
    d = ix.d
    fut = {}
    tok_of = {}
    t = -1
    last = None
    for r in rows:
        key = (int(d["prompt"][r]), int(d["pos"][r]))
        if key != last:
            t += 1
            last = key
        tok_of[r] = t
        L = int(d["layer"][r])
        for e in d["cur"][r]:
            e = int(e)
            if e >= 0:
                fut.setdefault((L, e), []).append(t)
    return fut, tok_of


def run(ix, rows, cost, tok_of, fut=None, oracle=None, decode_ok=None,
        capacity=66, max_inserts=2):
    d = ix.d
    now = {"t": 0}

    victim_fn = None
    if fut is not None:
        def victim_fn(layer, resident, recency):
            # Belady: evict whatever is needed furthest in the future. An expert
            # with no future use at all is the ideal victim.
            best, best_t = None, -1
            for e in resident:
                uses = fut.get((layer, e))
                nxt = None
                if uses:
                    i = bisect.bisect_right(uses, now["t"])
                    if i < len(uses):
                        nxt = uses[i]
                if nxt is None:
                    return e
                if nxt > best_t:
                    best, best_t = e, nxt
            return best if best is not None else next(iter(resident))

    env = PrefetchEnv(capacity=capacity, max_inserts=max_inserts, cost=cost,
                      victim_fn=victim_fn, n_layers=ix.n_layers)
    dstats = {"hit": 0, "lookup": 0, "uploads": 0}
    dtok, prev = 0, dict(env.stats)
    last = None
    for r in rows:
        L = int(d["layer"][r]); nxt = L + 1
        key = (int(d["prompt"][r]), int(d["pos"][r]))
        if key != last:
            if last is not None:
                env.step_boundary()
                if decode_ok is None or decode_ok.get(last, True):
                    for k in dstats:
                        dstats[k] += env.stats[k] - prev[k]
                    dtok += 1
                prev = dict(env.stats)
            last = key
        now["t"] = tok_of[r]
        if oracle is not None and (decode_ok is None or decode_ok.get(key, True)):
            w = oracle.get((key[0], key[1] + 1, nxt))
            if w is not None:
                want = [int(e) for e in w if e >= 0]
                if want:
                    env.prefetch(nxt, want, len(want), min_keep=max_inserts)
        cur = [int(e) for e in d["cur"][r] if e >= 0]
        env.observe(L, cur)
        env.admit_demand(L, cur)
    hit = dstats["hit"] / max(dstats["lookup"], 1)
    missed = ix.n_layers * 8 * (1 - hit)
    up = dstats["uploads"] / max(dtok, 1)
    ms = (cost["A_ms"] + cost["B_ms_per_miss"] * missed
          + cost["C_ms_per_upload"] * up)
    return {"tok_s": 1000.0 / max(ms, 1e-6), "hit": hit,
            "uploads_per_token": up, "tokens": dtok}


def main():
    index = os.path.join(ROOT, "data", "index-v4-replay.npz")
    corpus = os.path.join(ROOT, "data", "corpus-v4-replay")
    ix = Index(index)
    d = ix.d
    rows = ix.rows(1)
    rows = rows[np.lexsort((d["layer"][rows], d["pos"][rows], d["prompt"][rows]))]
    dec = phase_mask(ix, corpus)
    decode_ok = {(int(d["prompt"][r]), int(d["pos"][r])): bool(dec[r]) for r in rows}
    fut, tok_of = build_future(ix, rows)
    oracle = {(int(d["prompt"][r]), int(d["pos"][r]), int(d["layer"][r])): d["cur"][r]
              for r in rows}
    cost = fit_cost_model()

    arms = {
        "LRU, no prefetch":        dict(),
        "LRU, oracle prefetch":    dict(oracle=oracle),
        "Belady, no prefetch":     dict(fut=fut),
        "Belady, oracle prefetch": dict(fut=fut, oracle=oracle),
    }
    print("\n=== P1: WHERE IS THE REMAINING ROOM? ===\n")
    print(f"{'arm':26}{'decode tok/s':>14}{'vs LRU':>9}{'hit':>9}{'uploads/tok':>13}")
    print("-" * 71)
    out, base = {}, None
    for name, kw in arms.items():
        r = run(ix, rows, cost, tok_of, decode_ok=decode_ok, **kw)
        if base is None:
            base = r["tok_s"]
        r["gain"] = r["tok_s"] - base
        out[name] = r
        print(f"{name:26}{r['tok_s']:>14.1f}{r['gain']:>+9.1f}"
              f"{100*r['hit']:>8.1f}%{r['uploads_per_token']:>13.1f}")

    ev = out["Belady, no prefetch"]["gain"]
    ad = out["LRU, oracle prefetch"]["gain"]
    both = out["Belady, oracle prefetch"]["gain"]
    print(f"\nperfect eviction alone:  {ev:+.1f} tok/s")
    print(f"perfect admission alone: {ad:+.1f} tok/s")
    print(f"both together:           {both:+.1f} tok/s")
    if ev > ad:
        print("\nEviction is the bigger lever. A model that predicts TIME TO NEXT")
        print("USE is the thing to build; admission is secondary.")
    elif ev <= 0.5:
        print("\nPerfect eviction is worth nothing: LRU is already extracting")
        print("what this trace allows, and no eviction model can help.")
    else:
        print("\nAdmission is the bigger lever, but eviction is not free either.")
    json.dump(out, open(os.path.join(ART, "p1_eviction_headroom.json"), "w"),
              indent=1, default=float)
    print("\nwrote artifacts/p1_eviction_headroom.json")


if __name__ == "__main__":
    main()
