"""
D3b: what is within-token prefetch actually worth, under the real budget?

D3a settled the policy question and the answer was negative: at PR #27861's
step-boundary admission, a *perfect* next-token oracle is worth +4.2 points of
all-8-resident over plain demand-driven LRU, and a perfect eviction oracle is
worth 0.0. Prediction has almost nothing to win there, because admission happens
after the token has run and the cache already knows the answer it was guessing.

The +20 tok/s in this project's projection came from somewhere else, and it is
worth naming precisely: **issue time**. m3_simulate_speedup.py assumed the copy
for layer L+1 is issued while layer L computes, so it is resident the moment
L+1 runs -- a within-token prefetch. That is a different mechanism from a better
guess, and it is the one the whole 133 figure rests on.

So this prices the mechanism rather than the predictor, under three constraints
D3a and D1 established and m3 did not model:

  bandwidth   D1 measured that more uploads make things WORSE -- inserts 8 and
              16 both fell to 32.6 tok/s, under the 45.6 of no cache at all.
              Host RAM bandwidth is shared with the CPU expert compute this is
              trying to avoid. So a per-layer issue budget is enforced here and
              swept, rather than assumed unlimited.
  deadline    a 2.92 MiB expert takes 54.5 us over PCIe at the 53.66 GB/s AI2
              measured. A layer at the D1 operating point (112.8 tok/s, 48
              layers) is 184 us. So roughly 3 experts per layer can land in
              time, and anything beyond that is LATE and must not be counted as
              a hit -- the mistake m4 caught in the PyTorch engine.
  eviction    real LRU over `capacity` slots, with in-flight slots protected,
              exactly as llama-moecache.cpp does it.

Reported against the SAME simulator as D3a so the rows are comparable, and
against D1's measured 112.8 tok/s rather than against a prediction.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get("DATA", os.path.join(HERE, "m2_hidden.npz"))
OUT = os.path.join(HERE, "d3b_within_token_result.json")
MODEL = os.environ.get("MODEL", os.path.join(ROOT, "models", "predictor-full.bin"))

CAPACITY = int(os.environ.get("CAPACITY", "66"))
TOP_P = int(os.environ.get("TOP_P", "8"))
N_EXPERT, K = 128, 8
SLOPE_MS, FLOOR_MS = 0.3208, 5.876
CPU_EXPERT_MS = SLOPE_MS / K

# the deadline model, all measured in this project or AI2
EXPERT_MIB = 2.92
PCIE_GBPS = 53.66
COPY_US = EXPERT_MIB * 1024 * 1024 / (PCIE_GBPS * 1e9) * 1e6      # 54.5 us
D1_TOK_S = 112.8                                                   # measured
LAYER_US = 1e6 / D1_TOK_S / 48                                     # 184.8 us

sys.path.insert(0, HERE)
from d2_export import read_binary, BLOCK_PCA, BLOCK_PREV, BLOCK_CUR  # noqa: E402
from d3a_policy_sim import Layer  # noqa: E402


def simulate_within(tokens, n_layers, capacity, issue_budget, pred, enforce_deadline=True,
                    oracle=False):
    """Prefetch for layer L+1 is issued while layer L computes, then the layer
    runs and is scored. An expert whose copy could not finish inside the layer's
    compute window is late: resident for later layers, not for this one."""
    layers = [Layer(capacity, N_EXPERT) for _ in range(n_layers)]
    clock = 0
    hits = total = full = seen = uploads = late = 0
    in_flight = []          # (layer, expert, slot, ready_us) issued for the next layer

    for t, tok in enumerate(tokens):
        for L in sorted(tok["layers"]):
            # --- copies issued during layer L-1 land now. One that missed the
            # deadline still lands, but AFTER this layer has been scored -- it is
            # resident for later tokens and not a hit for the layer that wanted
            # it. Publishing it before scoring is the mistake m4 caught in the
            # PyTorch engine, where it silently turned late arrivals into hits.
            deferred = []
            for job in in_flight:
                (jl, je, js, ready) = job
                if enforce_deadline and ready > LAYER_US:
                    late += 1
                    deferred.append(job)
                else:
                    clock += 1
                    layers[jl].publish(je, js, clock)
            in_flight = []

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
            seen += 1
            if miss == 0:
                full += 1

            for (jl, je, js, _) in deferred:          # the late arrivals, now
                clock += 1
                layers[jl].publish(je, js, clock)

            # a miss is computed on the CPU; admit it so LRU behaves as it does today
            for e in ids:
                e = int(e)
                if not lay.resident(e):
                    slot = lay.pick_victim()
                    if slot >= 0:
                        lay.evict_and_reserve(slot)
                        clock += 1
                        lay.publish(e, slot, clock)

            # --- issue the prefetch for L+1 while this layer computes
            nxt = L + 1
            if nxt >= n_layers or nxt not in tok["layers"]:
                continue
            if oracle:
                want = [int(e) for e in tok["layers"][nxt]]
            else:
                want = [int(e) for e in pred[t].get(nxt, ())]
            issued = 0
            for e in want:
                if issued >= issue_budget:
                    break
                nl = layers[nxt]
                if nl.resident(e) or any(j[0] == nxt and j[1] == e for j in in_flight):
                    continue
                slot = nl.pick_victim()
                if slot < 0:
                    break
                nl.evict_and_reserve(slot)
                issued += 1
                uploads += 1
                in_flight.append((nxt, e, slot, issued * COPY_US))

    fl = full / max(seen, 1)
    eh = hits / max(total, 1)
    return {"expert_hit": eh, "full_layer_rate": fl,
            "uploads_per_token": uploads / len(tokens),
            "late_per_token": late / len(tokens),
            "pessimistic_tok_s": 1000 / (FLOOR_MS + (1 - fl) * n_layers * SLOPE_MS),
            "optimistic_tok_s": 1000 / (FLOOR_MS + (1 - eh) * n_layers * K * CPU_EXPERT_MS)}


def main():
    d = np.load(DATA, allow_pickle=True)
    H = d["hidden"].astype(np.float32)
    E = d["experts"].astype(np.int64)
    keys, reg = d["keys"], d["register"].astype(str)
    n_layers = int(keys[:, 2].max()) + 1
    test_r = sorted(set(reg))[6:]

    by_tok = {}
    for i in np.where(np.isin(reg, test_r))[0]:
        p, t, l = keys[i]
        by_tok.setdefault((int(p), int(t)), {"layers": {}, "row": {}})
        by_tok[(int(p), int(t))]["layers"][int(l)] = E[i]
        by_tok[(int(p), int(t))]["row"][int(l)] = i
    tokens = [by_tok[k] for k in sorted(by_tok)]

    m = read_binary(MODEL)
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-6)
    P = ((Hn - m["mean"]) @ m["comp"]) / m["scale"] if BLOCK_PCA in m["blocks"] else None
    pred = [dict() for _ in tokens]
    prev_ids = {}
    for t, tok in enumerate(tokens):
        for L in range(1, n_layers):
            s_L = L - 1
            if s_L not in m["W"] or s_L not in tok["row"]:
                continue
            x = np.zeros(m["feat_dim"], dtype=np.float32)
            for (b, off, size) in m["table"]:
                if b == BLOCK_PCA:
                    x[off:off + size] = P[tok["row"][s_L]]
                elif b == BLOCK_PREV and L in prev_ids:
                    x[off + prev_ids[L]] = 1.0
                elif b == BLOCK_CUR:
                    x[off + E[tok["row"][s_L]]] = 1.0
            sc = x @ m["W"][s_L]
            sc = (sc - sc.mean()) / (sc.std() + 1e-9)       # the exported prior
            if m["prior"][s_L] and L in prev_ids:
                sc[prev_ids[L]] += m["prior"][s_L]
            pred[t][L] = np.argpartition(-sc, TOP_P)[:TOP_P]
        for L in tok["layers"]:
            prev_ids[L] = E[tok["row"][L]]

    print(f"{len(tokens):,} tokens; layer window {LAYER_US:.0f} us, "
          f"one expert {COPY_US:.1f} us -> {LAYER_US / COPY_US:.1f} fit in time",
          file=sys.stderr)

    rows = []
    for budget in (1, 2, 3, 4, 6, 8):
        for name, oracle, dl in (("predictor", False, True),
                                 ("predictor, no deadline", False, False),
                                 ("oracle", True, True)):
            r = simulate_within(tokens, n_layers, CAPACITY, budget, pred,
                                enforce_deadline=dl, oracle=oracle)
            r.update(budget=budget, source=name)
            rows.append(r)
            print(f"  budget {budget} {name:<24} all-8 {100 * r['full_layer_rate']:5.1f}%  "
                  f"{r['pessimistic_tok_s']:5.1f}-{r['optimistic_tok_s']:5.1f} tok/s  "
                  f"{r['uploads_per_token']:5.1f} up/tok", file=sys.stderr, flush=True)

    print(f"\n=== D3b: WITHIN-TOKEN PREFETCH, PRICED (capacity {CAPACITY}, "
          f"{len(tokens):,} held-out tokens) ===\n")
    print("%7s %-24s %11s %15s %12s %11s %17s"
          % ("budget", "source", "expert hit", "all-8 resident", "uploads/tok",
             "late/tok", "tok/s (pess-opt)"))
    for r in rows:
        print("%7d %-24s %10.1f%% %14.1f%% %11.1f %10.1f %8.1f - %-7.1f"
              % (r["budget"], r["source"], 100 * r["expert_hit"], 100 * r["full_layer_rate"],
                 r["uploads_per_token"], r["late_per_token"],
                 r["pessimistic_tok_s"], r["optimistic_tok_s"]))

    # D1 anchors the scale: 112.8 measured where the step-boundary sim says 56.9%
    # all-8 -> 79.9-125.3. Locate 112.8 in that band and read the others at the
    # same point, instead of picking whichever bound flatters the result.
    lo, hi = 79.9, 125.3
    w = (D1_TOK_S - lo) / (hi - lo)
    print(f"\nD1 measured {D1_TOK_S} tok/s where the step-boundary simulation gives "
          f"{lo}-{hi};\nthat is {100 * w:.0f}% of the way up the band. Reading every row at "
          "the same point:\n")
    print("%7s %-24s %12s %10s" % ("budget", "source", "tok/s", "vs D1"))
    best = None
    for r in rows:
        est = r["pessimistic_tok_s"] + w * (r["optimistic_tok_s"] - r["pessimistic_tok_s"])
        r["anchored_tok_s"] = est
        print("%7d %-24s %11.1f %+9.1f" % (r["budget"], r["source"], est, est - D1_TOK_S))
        if r["source"] == "predictor" and (best is None or est > best["anchored_tok_s"]):
            best = r
    print(f"\nBest achievable with this predictor: {best['anchored_tok_s']:.1f} tok/s at "
          f"budget {best['budget']} ({best['anchored_tok_s'] - D1_TOK_S:+.1f} over D1's measured "
          f"{D1_TOK_S}).")

    json.dump({"capacity": CAPACITY, "top_p": TOP_P, "layer_us": LAYER_US,
               "copy_us": COPY_US, "d1_tok_s": D1_TOK_S, "anchor_w": w,
               "rows": rows}, open(OUT, "w"), indent=1, default=float)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
