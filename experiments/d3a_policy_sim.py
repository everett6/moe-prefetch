"""
D3a: simulate PR #27861's ACTUAL admission policy, not the idealised one.

m3_simulate_speedup.py modelled prefetching the way the design wanted it to
work: predict layer L+1's experts while layer L computes, and have them resident
by the time L+1 runs. D1 then measured the real implementation at 112.8 tok/s
against m3's predicted 113.0, which validated the cost model -- but it validated
the *LRU* row, and LRU needs no prefetch at all.

Reading `llama-moecache.cpp` for the D3 hook shows the predictor cannot ride on
that policy as written, for a reason that has nothing to do with prediction
quality:

  moe_obs_cb()          runs during graph execution, records misses into `pending`
  llama_moe_cache_step() runs AFTER decode, publishes uploads finished since the
                        last step, then schedules <= max_inserts new ones
  worker thread         does the copies off the decode thread

So an expert that misses during token t is scheduled at the end of token t and
published at the end of token t+1 -- resident from token **t+2**. Admission is
per-token, not per-layer, and by the time it happens the cache already knows the
true expert set for the whole token that just ran.

That is the problem. A predictor of E_{L+1}(t) is competing against ground truth
that arrives before the upload is even scheduled. Whatever its recall, it cannot
beat knowing the answer.

So the question D3 actually has to answer is not "is the predictor good enough",
it is "**is there anything left for any predictor to win here**". This simulates
the real policy faithfully -- 2-token latency, max_inserts budget, reverse-order
pending drain, LRU eviction skipping in-flight slots -- and runs five candidate
sources through it, including a next-token oracle that no predictor can beat.

If the oracle is close to demand-driven LRU, the answer is that the graph
surgery is the prerequisite and the predictor is not the bottleneck. That is a
result, and it is much cheaper to find here than in C++.
"""
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get("DATA", os.path.join(HERE, "m2_hidden.npz"))
OUT = os.path.join(HERE, "d3a_policy_sim_result.json")
MODEL = os.environ.get("MODEL", os.path.join(ROOT, "models", "predictor-full.bin"))

CAPACITY = int(os.environ.get("CAPACITY", "66"))
MAX_INSERTS = [int(x) for x in os.environ.get("MAX_INSERTS", "2,4,8").split(",")]
TOP_P = int(os.environ.get("TOP_P", "8"))
N_EXPERT, K = 128, 8
SLOPE_MS, FLOOR_MS = 0.3208, 5.876
CPU_EXPERT_MS = SLOPE_MS / K

sys.path.insert(0, HERE)
from d2_export import read_binary, BLOCK_PCA, BLOCK_PREV, BLOCK_CUR  # noqa: E402


class Layer:
    """One layer's slice of the cache, mirroring layer_state in llama-moecache.cpp."""

    __slots__ = ("n_slots", "slot_expert", "expert_slot", "last_use", "in_flight", "pending")

    def __init__(self, n_slots, n_expert):
        self.n_slots = n_slots
        self.slot_expert = [-1] * n_slots
        self.expert_slot = [-1] * n_expert
        self.last_use = [0] * n_slots
        self.in_flight = [False] * n_slots
        self.pending = []

    def resident(self, e):
        return self.expert_slot[e] >= 0

    def touch(self, e, clock):
        self.last_use[self.expert_slot[e]] = clock

    def pick_victim(self, protected=()):
        """Empty non-in-flight slot if any, else LRU non-in-flight. -1 if none.

        `protected` holds experts something believes will be needed soon. They are
        skipped unless every candidate slot is protected, in which case the LRU
        rule applies as before -- a protection set must never be able to wedge
        the cache."""
        slot, best = -1, None
        fallback, fbest = -1, None
        for s in range(self.n_slots):
            if self.in_flight[s]:
                continue
            e = self.slot_expert[s]
            if e < 0:
                return s
            if fbest is None or self.last_use[s] < fbest:
                fbest, fallback = self.last_use[s], s
            if e in protected:
                continue
            if best is None or self.last_use[s] < best:
                best, slot = self.last_use[s], s
        return slot if slot >= 0 else fallback

    def evict_and_reserve(self, slot):
        victim = self.slot_expert[slot]
        if victim >= 0:
            self.expert_slot[victim] = -1
            self.slot_expert[slot] = -1
        self.in_flight[slot] = True

    def publish(self, expert, slot, clock):
        self.slot_expert[slot] = expert
        self.expert_slot[expert] = slot
        self.last_use[slot] = clock
        self.in_flight[slot] = False


def simulate(tokens, n_layers, capacity, max_inserts, candidates_fn, protect_fn=None,
             early_issue=False):
    """Faithful replay of llama-moecache.cpp's observe -> step -> worker pipeline.

    candidates_fn(t, L, layer) -> ranked expert ids to consider admitting, best
    first. protect_fn(t, L) -> experts eviction should spare. The budget, the
    publish latency and the LRU rule are the real ones.

    early_issue changes WHEN the copy is handed to the worker, and nothing else.
    As shipped, a miss at token t is scheduled at the end of t and published at
    the end of t+1 -- resident from t+2. The copy is 54 us and a token is 8.9 ms,
    so the worker is idle for almost all of that wait. Issuing from the observe
    callback instead, while the rest of the token is still running, lets the same
    job be published at the end of token t: resident from t+1.

    This is the one latency reduction that needs no mid-graph table mutation, so
    it is the one that is safe. The tables still change only inside step().
    """
    layers = [Layer(capacity, N_EXPERT) for _ in range(n_layers)]
    clock = 0
    inflight_jobs = []          # scheduled at the previous step, published at this one
    hits = total = full = seen = uploads = 0

    for t, tok in enumerate(tokens):
        # ---- graph execution: observe every routed expert
        for L in sorted(tok["layers"]):
            lay = layers[L]
            ids = tok["layers"][L]
            miss = 0
            for e in ids:
                e = int(e)
                total += 1
                if lay.resident(e):
                    hits += 1
                    clock += 1
                    lay.touch(e, clock)
                else:
                    miss += 1
                    if e not in lay.pending:
                        lay.pending.append(e)
            seen += 1
            if miss == 0:
                full += 1

        # ---- llama_moe_cache_step(): publish first, then schedule
        for (L, e, s) in inflight_jobs:
            clock += 1
            layers[L].publish(e, s, clock)
        inflight_jobs = []

        for L in range(n_layers):
            lay = layers[L]
            cands = candidates_fn(t, L, lay)
            if not cands:
                lay.pending.clear()
                continue
            budget = max_inserts
            for e in cands:
                if budget <= 0:
                    break
                e = int(e)
                if lay.resident(e) or any(j[0] == L and j[1] == e for j in inflight_jobs):
                    continue
                slot = lay.pick_victim(protect_fn(t, L) if protect_fn else ())
                if slot < 0:
                    break
                lay.evict_and_reserve(slot)
                inflight_jobs.append((L, e, slot))
                uploads += 1
                budget -= 1
            lay.pending.clear()

        if early_issue:                 # the copies were already in flight: publish now
            for (L, e, s_) in inflight_jobs:
                clock += 1
                layers[L].publish(e, s_, clock)
            inflight_jobs = []

    fl = full / max(seen, 1)
    eh = hits / max(total, 1)
    pess = 1000 / (FLOOR_MS + (1 - fl) * n_layers * SLOPE_MS)
    opt = 1000 / (FLOOR_MS + (1 - eh) * n_layers * K * CPU_EXPERT_MS)
    return {"expert_hit": eh, "full_layer_rate": fl, "uploads_per_token": uploads / len(tokens),
            "pessimistic_tok_s": pess, "optimistic_tok_s": opt}


def main():
    d = np.load(DATA, allow_pickle=True)
    H = d["hidden"].astype(np.float32)
    E = d["experts"].astype(np.int64)
    keys, reg = d["keys"], d["register"].astype(str)
    n_layers = int(keys[:, 2].max()) + 1
    regs = sorted(set(reg))
    test_r = regs[6:]

    by_tok = {}
    for i in np.where(np.isin(reg, test_r))[0]:
        p, t, l = keys[i]
        by_tok.setdefault((int(p), int(t)), {"layers": {}, "row": {}})
        by_tok[(int(p), int(t))]["layers"][int(l)] = E[i]
        by_tok[(int(p), int(t))]["row"][int(l)] = i
    order = sorted(by_tok)
    tokens = [by_tok[k] for k in order]
    print(f"{len(tokens):,} test tokens x {n_layers} layers, capacity {CAPACITY}",
          file=sys.stderr)

    # ---- the exported predictor, scoring every (token, layer) up front
    m = read_binary(MODEL)
    print(f"predictor {os.path.basename(MODEL)}: d={m['feat_dim']}, "
          f"{len(m['W'])} layers", file=sys.stderr)
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-6)
    P = ((Hn - m["mean"]) @ m["comp"]) / m["scale"] if BLOCK_PCA in m["blocks"] else None

    # pred[t][L] = top-P predicted experts for layer L of token t, from layer L-1
    pred = [dict() for _ in tokens]
    prev_ids = {}                       # layer -> previous token's expert ids
    for t, tok in enumerate(tokens):
        for L in range(1, n_layers):
            src_L = L - 1
            if src_L not in m["W"] or src_L not in tok["row"]:
                continue
            x = np.zeros(m["feat_dim"], dtype=np.float32)
            for (b, off, size) in m["table"]:
                if b == BLOCK_PCA:
                    x[off:off + size] = P[tok["row"][src_L]]
                elif b == BLOCK_PREV:
                    if L in prev_ids:
                        x[off + prev_ids[L]] = 1.0
                elif b == BLOCK_CUR:
                    x[off + E[tok["row"][src_L]]] = 1.0
            s = x @ m["W"][src_L]
            s = (s - s.mean()) / (s.std() + 1e-9)           # the exported prior
            if m["prior"][src_L] and L in prev_ids:
                s[prev_ids[L]] += m["prior"][src_L]
            pred[t][L] = np.argpartition(-s, TOP_P)[:TOP_P]
        for L in tok["layers"]:
            prev_ids[L] = E[tok["row"][L]]

    # ---- candidate sources
    def demand(t, L, lay):
        return list(reversed(lay.pending))          # exactly what the PR does

    def predict_only(t, L, lay):
        return list(pred[t].get(L, []))

    def demand_then_predict(t, L, lay):
        out = list(reversed(lay.pending))
        out += [int(e) for e in pred[t].get(L, []) if e not in out]
        return out

    def oracle_next(t, L, lay):
        """The experts the NEXT token will use here: the ceiling on any predictor
        under step-boundary admission, because nothing can know more than this."""
        nxt = tokens[t + 1]["layers"].get(L) if t + 1 < len(tokens) else None
        return [] if nxt is None else [int(e) for e in nxt]

    def oracle_next_then_demand(t, L, lay):
        out = oracle_next(t, L, lay)
        out += [e for e in reversed(lay.pending) if e not in out]
        return out

    def next_set(t, L):
        nxt = tokens[t + 1]["layers"].get(L) if t + 1 < len(tokens) else None
        return set() if nxt is None else {int(e) for e in nxt}

    def protect_pred(t, L):
        """Spare whatever the predictor scored highly for this layer. Costs no
        bandwidth -- it only changes which slot a copy lands in."""
        return set(int(e) for e in pred[t].get(L, ()))

    def filtered_demand(t, L, lay):
        """Admit a miss only if the predictor also ranks it. D1 measured that
        extra upload bandwidth is actively harmful (inserts 8 and 16 both fell
        to 32.6 tok/s, below no cache at all), so declining a fetch is a lever
        in the direction that machine rewards."""
        top = set(int(e) for e in pred[t].get(L, ()))
        keep = [e for e in reversed(lay.pending) if e in top]
        return keep

    SOURCES = [("demand (as implemented)", demand, None),
               ("demand, early issue", demand, None),
               ("demand+predict, early", demand_then_predict, None),
               ("predict only", predict_only, None),
               ("demand + predict", demand_then_predict, None),
               ("demand, predict-protect", demand, protect_pred),
               ("predict-filtered admit", filtered_demand, None),
               ("oracle: next token", oracle_next, None),
               ("oracle: next + demand", oracle_next_then_demand, None),
               ("oracle: protect only", demand, next_set),
               ("oracle: admit + protect", oracle_next_then_demand, next_set)]

    rows = []
    for mi in MAX_INSERTS:
        for name, fn, pfn in SOURCES:
            r = simulate(tokens, n_layers, CAPACITY, mi, fn, pfn,
                         early_issue=name.endswith("early") or "early issue" in name)
            r.update(max_inserts=mi, source=name)
            rows.append(r)
            print(f"  inserts {mi} {name:<24} all-8 {100 * r['full_layer_rate']:5.1f}%  "
                  f"{r['pessimistic_tok_s']:5.1f}-{r['optimistic_tok_s']:5.1f} tok/s",
                  file=sys.stderr, flush=True)

    print(f"\n=== D3a: PR #27861's REAL ADMISSION POLICY (capacity {CAPACITY}, "
          f"{len(tokens):,} held-out tokens) ===\n")
    print("%8s %-24s %11s %15s %12s %17s"
          % ("inserts", "candidate source", "expert hit", "all-8 resident",
             "uploads/tok", "tok/s (pess-opt)"))
    for r in rows:
        print("%8d %-24s %10.1f%% %14.1f%% %11.2f %8.1f - %-7.1f"
              % (r["max_inserts"], r["source"], 100 * r["expert_hit"],
                 100 * r["full_layer_rate"], r["uploads_per_token"],
                 r["pessimistic_tok_s"], r["optimistic_tok_s"]))

    base = next(r for r in rows if r["max_inserts"] == 2 and r["source"].startswith("demand ("))
    pr = next(r for r in rows if r["max_inserts"] == 2 and r["source"] == "demand + predict")
    orc = next(r for r in rows if r["max_inserts"] == 2 and r["source"] == "oracle: next + demand")
    print(f"\nAt the shipped budget of 2 inserts/layer/step:")
    print(f"  the predictor adds {100 * (pr['full_layer_rate'] - base['full_layer_rate']):+.1f} "
          "points of all-8-resident over demand-driven LRU")
    print(f"  a PERFECT next-token oracle adds {100 * (orc['full_layer_rate'] - base['full_layer_rate']):+.1f}")
    print("\nThe oracle row is the ceiling: no predictor, however good, can beat "
          "knowing\nwhich experts the next token will use. If that ceiling is low, the "
          "limit is the\npolicy, not the prediction.")

    json.dump({"capacity": CAPACITY, "top_p": TOP_P, "model": os.path.basename(MODEL),
               "n_tokens": len(tokens), "rows": rows}, open(OUT, "w"), indent=1, default=float)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
