"""
P2: how much of Belady's +53.3 tok/s does a policy you can actually implement get?

P1 showed perfect eviction is worth +53.3 tok/s against LRU, five times the
admission ceiling, and that it works by cutting uploads per token from 20.5 to
8.0. Belady reads the future, so it is a bound, not a policy.

Before training a model to approximate it, the cheap options get measured, because
a counter you can add to the C++ cache in twenty lines beats a neural network
that needs a feature pipeline, a training set and an inference budget inside the
critical path. Rule 5: prefer the simple improvement that works.

  LRU        what ships now: evict least recently used
  LFU        evict least frequently used, counted over the whole run
  LFU-W      frequency over a sliding window, so a burst long past stops
             protecting an expert forever
  LRU2       evict by SECOND most recent use -- one touch no longer saves you,
             which is the classic fix for scan pollution
  score      w*normalised-frequency + (1-w)*normalised-recency

The routing here has a strong frequency structure -- the commonest expert is
used 1.5x as often as the rarest across the corpus, but within one layer over a
short window the spread is far wider -- so frequency is the obvious signal to
try first.
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
from p1_eviction_headroom import build_future  # noqa: E402

ART = os.path.join(ROOT, "artifacts")


def make_policy(kind, state, fut=None, now=None, window=64, w=0.5):
    """Victim choosers. `state` carries the counters the experiment maintains."""
    if kind == "LRU":
        return None

    if kind == "belady":
        def f(layer, resident, recency):
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
        return f

    if kind == "LFU":
        def f(layer, resident, recency):
            c = state["count"][layer]
            return min(resident, key=lambda e: (c.get(e, 0), recency.get(e, 0)))
        return f

    if kind == "LFU-W":
        def f(layer, resident, recency):
            c = state["win"][layer]
            return min(resident, key=lambda e: (len(c.get(e, ())), recency.get(e, 0)))
        return f

    if kind == "LRU2":
        def f(layer, resident, recency):
            h = state["hist"][layer]
            def key(e):
                u = h.get(e, ())
                # no second use yet: treat as infinitely old, which is what
                # makes this resist a single touch protecting a cold expert
                return u[-2] if len(u) >= 2 else -1
            return min(resident, key=lambda e: (key(e), recency.get(e, 0)))
        return f

    if kind == "score":
        def f(layer, resident, recency):
            c = state["win"][layer]
            mx = max((len(c.get(e, ())) for e in resident), default=1) or 1
            t = now["t"] or 1
            def key(e):
                freq = len(c.get(e, ())) / mx
                rec = recency.get(e, 0) / max(t, 1)
                return w * freq + (1 - w) * rec
            return min(resident, key=key)
        return f
    raise ValueError(kind)


def run(ix, rows, cost, tok_of, kind, fut=None, decode_ok=None,
        capacity=66, max_inserts=2, window=64, w=0.5):
    d = ix.d
    now = {"t": 0}
    state = {"count": [dict() for _ in range(ix.n_layers)],
             "win": [dict() for _ in range(ix.n_layers)],
             "hist": [dict() for _ in range(ix.n_layers)]}
    vf = make_policy(kind, state, fut=fut, now=now, window=window, w=w)
    env = PrefetchEnv(capacity=capacity, max_inserts=max_inserts, cost=cost,
                      victim_fn=vf, n_layers=ix.n_layers)
    dstats = {"hit": 0, "lookup": 0, "uploads": 0}
    dtok, prev, last = 0, dict(env.stats), None
    for r in rows:
        L = int(d["layer"][r])
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
        cur = [int(e) for e in d["cur"][r] if e >= 0]
        env.observe(L, cur)
        env.admit_demand(L, cur)
        # counters update AFTER the decision, from what the engine just saw --
        # everything here is observable at runtime, unlike `fut`
        t = now["t"]
        for e in cur:
            state["count"][L][e] = state["count"][L].get(e, 0) + 1
            u = state["hist"][L].setdefault(e, [])
            u.append(t)
            wl = state["win"][L].setdefault(e, [])
            wl.append(t)
            while wl and wl[0] < t - window:
                wl.pop(0)
    hit = dstats["hit"] / max(dstats["lookup"], 1)
    missed = ix.n_layers * 8 * (1 - hit)
    up = dstats["uploads"] / max(dtok, 1)
    ms = (cost["A_ms"] + cost["B_ms_per_miss"] * missed
          + cost["C_ms_per_upload"] * up)
    return {"tok_s": 1000.0 / max(ms, 1e-6), "hit": hit, "uploads_per_token": up}


def main():
    ix = Index(os.path.join(ROOT, "data", "index-v4-replay.npz"))
    d = ix.d
    rows = ix.rows(1)
    rows = rows[np.lexsort((d["layer"][rows], d["pos"][rows], d["prompt"][rows]))]
    dec = phase_mask(ix, os.path.join(ROOT, "data", "corpus-v4-replay"))
    decode_ok = {(int(d["prompt"][r]), int(d["pos"][r])): bool(dec[r]) for r in rows}
    fut, tok_of = build_future(ix, rows)
    cost = fit_cost_model()

    print("\n=== P2: IMPLEMENTABLE EVICTION POLICIES ===\n")
    print(f"{'policy':16}{'decode tok/s':>14}{'vs LRU':>9}{'hit':>9}"
          f"{'uploads/tok':>13}{'of Belady':>11}")
    print("-" * 72)
    out = {}
    base = run(ix, rows, cost, tok_of, "LRU", decode_ok=decode_ok)
    bel = run(ix, rows, cost, tok_of, "belady", fut=fut, decode_ok=decode_ok)
    span = bel["tok_s"] - base["tok_s"]
    trials = [("LRU", {}), ("LFU", {}), ("LFU-W", {"window": 64}),
              ("LFU-W256", {"window": 256}), ("LRU2", {}),
              ("score w=.5", {"w": 0.5}), ("score w=.8", {"w": 0.8})]
    for name, kw in trials:
        kind = name.split()[0].replace("LFU-W256", "LFU-W")
        win = kw.pop("window", 256 if "256" in name else 64)
        r = run(ix, rows, cost, tok_of, kind, decode_ok=decode_ok, window=win, **kw)
        r["gain"] = r["tok_s"] - base["tok_s"]
        r["of_belady"] = r["gain"] / span if span else 0.0
        out[name] = r
        print(f"{name:16}{r['tok_s']:>14.1f}{r['gain']:>+9.1f}{100*r['hit']:>8.1f}%"
              f"{r['uploads_per_token']:>13.1f}{100*r['of_belady']:>10.0f}%")
    out["belady"] = dict(bel, gain=span, of_belady=1.0)
    print(f"{'Belady (bound)':16}{bel['tok_s']:>14.1f}{span:>+9.1f}"
          f"{100*bel['hit']:>8.1f}%{bel['uploads_per_token']:>13.1f}{100:>10.0f}%")

    best = max((k for k in out if k != "belady"), key=lambda k: out[k]["tok_s"])
    print(f"\nbest implementable: {best} at {out[best]['gain']:+.1f} tok/s, "
          f"{100*out[best]['of_belady']:.0f}% of the Belady bound")
    json.dump(out, open(os.path.join(ART, "p2_eviction_policies.json"), "w"),
              indent=1, default=float)
    print("wrote artifacts/p2_eviction_policies.json")


if __name__ == "__main__":
    main()
