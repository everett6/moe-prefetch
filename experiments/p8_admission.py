"""
P8: should every miss be admitted?

The engine admits on the FIRST miss: an expert that is routed to and is not in
VRAM is uploaded (up to max_inserts per layer per step) and displaces the LRU
victim. P1 showed what that costs -- Belady reaches a better hit rate with 61%
fewer uploads (8.0 against 20.5 per token), so most of what LRU uploads is
wasted motion: an expert that is used once is paid for (C = 142 us), evicts an
expert that was about to be used again, and is itself evicted unused.

The fitted cost model gives the rule directly. A miss costs B = 47 us of CPU
compute; an upload costs C = 142 us. An admission therefore has to buy about
C/B = 3 future hits before it has paid for itself, before counting whatever
the victim would have returned. "Admit on first miss" assumes every missed
expert will; the cold tail does not.

Both reference engines gate this and this engine does not:

  colibri   a speculative load may evict a resident only if the candidate is
            historically HOTTER than the victim (25% + 4 hysteresis)
  ds4       "look-ahead is not evidence of reuse": a prefetched slot enters at
            the bottom of the recency order and is promoted only by a real hit

P2 found frequency-based EVICTION harmful here (LFU -8.5 tok/s), so nothing in
this file is assumed to work. The policies are measured against the same real
decode traces, with LRU eviction held fixed so only admission varies:

  first-miss        the engine today
  k-in-W            admit on the k-th miss of an expert within the last W tokens
                    (the 2Q / ARC ghost-list rule)
  hotter-than-LRU   colibri's guard: a decayed use count for every expert, and
                    the candidate is admitted only if it beats the LRU victim's
                    by the same 25% + 4 hysteresis

CAVEAT, the same one every simulated number here carries: Phase 0 was never
done and PrefetchEnv is not trace-validated against the engine. Phase 2 is the
one data point on how far to trust it -- predicted +4.5, measured +3.7 -- so
read the RANKING of policies, not the tok/s. Nothing here is a result until
r3_real_bench.py (WITH_ADMIT) says so on the real engine.
"""
import json
import os
import sys
from collections import defaultdict, deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from train_deep import Index  # noqa: E402
from prefetch_env import PrefetchEnv, fit_cost_model  # noqa: E402
from r5b_phase import phase_mask  # noqa: E402

ART = os.path.join(ROOT, "artifacts")


class FirstMiss:
    name = "first-miss (engine today)"

    def reset(self, n_layers):
        pass

    def use(self, layer, ids, t):
        pass

    def admit(self, env, layer, misses, t):
        return misses


class KInWindow:
    """Admit on the k-th miss of an expert within the last `window` tokens."""

    def __init__(self, k, window):
        self.k, self.window = k, window
        self.name = f"{k}-in-{window}"

    def reset(self, n_layers):
        self.ghost = [defaultdict(deque) for _ in range(n_layers)]

    def use(self, layer, ids, t):
        pass

    def admit(self, env, layer, misses, t):
        out = []
        g = self.ghost[layer]
        for e in misses:
            q = g[e]
            q.append(t)
            while q and q[0] <= t - self.window:
                q.popleft()
            if len(q) >= self.k:
                out.append(e)
        return out


class HotterThanVictim:
    """colibri's guard. Heat counts every use (hit or miss) and halves every
    `half` tokens; a miss is admitted only if its heat beats the LRU victim's
    by 25% + `floor`. A layer with spare slots always admits."""

    def __init__(self, half=256, floor=1.0):
        self.half, self.floor = half, floor
        self.name = f"hotter-than-LRU (half {half}, +{floor:g})"

    def reset(self, n_layers):
        self.heat = [defaultdict(float) for _ in range(n_layers)]
        self.last_decay = 0

    def use(self, layer, ids, t):
        if t - self.last_decay >= self.half:
            for h in self.heat:
                for e in list(h):
                    h[e] *= 0.5
                    if h[e] < 0.05:
                        del h[e]
            self.last_decay = t
        h = self.heat[layer]
        for e in ids:
            h[e] += 1.0

    def admit(self, env, layer, misses, t):
        if len(env.resident[layer]) < env.cap[layer]:
            return misses
        rec = env.recency[layer]
        if not rec:
            return misses
        victim = min(rec, key=rec.get)
        h = self.heat[layer]
        vs = h.get(victim, 0.0)
        return [e for e in misses if h.get(e, 0.0) > vs * 1.25 + self.floor]


def replay(ix, rows, decode_ok, cost, policy, capacity, max_inserts=2):
    d = ix.d
    env = PrefetchEnv(capacity=capacity, max_inserts=max_inserts, cost=cost,
                      n_layers=ix.n_layers)
    policy.reset(ix.n_layers)
    dstats = {"hit": 0, "lookup": 0, "uploads": 0}
    dtok, prev, last, t = 0, dict(env.stats), None, 0
    for r in rows:
        L = int(d["layer"][r])
        key = (int(d["prompt"][r]), int(d["pos"][r]))
        if key != last:
            if last is not None:
                env.step_boundary()
                if decode_ok.get(last, True):
                    for k in dstats:
                        dstats[k] += env.stats[k] - prev[k]
                    dtok += 1
                prev = dict(env.stats)
                t += 1
            last = key
        cur = [int(e) for e in d["cur"][r] if e >= 0]
        misses = [e for e in cur if e not in env.resident[L]]
        env.observe(L, cur)
        policy.use(L, cur, t)
        env.admit_demand(L, policy.admit(env, L, misses, t))
    env.step_boundary()
    if decode_ok.get(last, True):
        for k in dstats:
            dstats[k] += env.stats[k] - prev[k]
        dtok += 1
    hit = dstats["hit"] / max(dstats["lookup"], 1)
    missed = ix.n_layers * 8 * (1 - hit)
    up = dstats["uploads"] / max(dtok, 1)
    ms = cost["A_ms"] + cost["B_ms_per_miss"] * missed + cost["C_ms_per_upload"] * up
    return {"tok_s": 1000.0 / max(ms, 1e-6), "hit": hit, "missed_per_token": missed,
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
    cost = fit_cost_model()

    policies = [FirstMiss()]
    for k, w in ((2, 4), (2, 8), (2, 16), (2, 32), (2, 64), (2, 256), (3, 16), (3, 64)):
        policies.append(KInWindow(k, w))
    for half, floor in ((64, 1.0), (256, 1.0), (256, 4.0), (1024, 1.0)):
        policies.append(HotterThanVictim(half, floor))

    out = {}
    for cap in (56, 72):
        print(f"\n=== capacity {cap} per layer, LRU eviction, decode rows only ===")
        print(f"{'admission policy':34}{'tok/s':>8}{'vs first':>10}{'hit':>8}"
              f"{'miss/tok':>10}{'upl/tok':>9}")
        print("-" * 79)
        base = None
        for p in policies:
            r = replay(ix, rows, decode_ok, cost, p, cap)
            if base is None:
                base = r["tok_s"]
            r["gain"] = r["tok_s"] - base
            out[f"{cap}/{p.name}"] = r
            print(f"{p.name:34}{r['tok_s']:8.1f}{r['gain']:+10.1f}{100*r['hit']:7.1f}%"
                  f"{r['missed_per_token']:10.1f}{r['uploads_per_token']:9.1f}", flush=True)

    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "p8_admission.json"), "w") as f:
        json.dump({"results": out,
                   "caveat": "PrefetchEnv is not trace-validated (Phase 0 never done). "
                             "Ranking only; the engine A/B is the result."}, f, indent=1)
    print("\nwrote artifacts/p8_admission.json")
    print("Simulated. Ranking only -- measure with WITH_ADMIT=1 r3_real_bench.py.")


if __name__ == "__main__":
    main()
