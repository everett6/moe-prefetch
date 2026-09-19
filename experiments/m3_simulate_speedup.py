"""
Milestone 3 gate: what is 59.7% recall actually worth, in tokens per second?

This is the honest question before asking anyone to install a CUDA toolchain and
patch llama.cpp's MoE graph. It is simulated on the real traces from m2_capture.py
rather than argued from arithmetic.

The modelling point that decides everything comes from AI2's expert_cache_sim.py,
and it is easy to miss: **if even one of a layer's 8 experts is absent from VRAM,
the hidden state has to hop to the CPU and back, and the layer pays close to the
full CPU-layer cost.** Per-expert hit rate is therefore the wrong headline
number. What matters is the fraction of layers where ALL EIGHT are resident.

That distinction is brutal for a predictor. 59.7% per-expert recall, if the
errors were independent, would leave 0.597^8 = 1.6% of layers fully covered. So a
predictor alone cannot work; it has to ride on top of a cache that already holds
most of what is needed, and earn its keep on the few experts the cache misses.

So four policies are simulated, all on held-out registers:

  none          today: whole layers, no cache
  lru           a cache of CAPACITY experts per layer, evicting least-recently-used.
                This is roughly what llama.cpp PR #27861 already does.
  lru+predict   the same cache, plus the ridge probe's top-P predictions for layer
                L+1 inserted while layer L computes
  oracle        the same cache, but prefetching the experts layer L+1 will
                actually use -- the unreachable bound on what prediction is worth

Timing constants are AI2's, measured on this machine (experiments/
expert_cache_sim_result.json): SLOPE 0.321 ms per CPU-side expert layer, FLOOR
5.88 ms/token with every expert on the GPU, 0.057 ms to fetch one expert over
PCIe, 0.040 ms to compute one on the CPU. Fetching costs more than computing, so
a miss is computed on the CPU rather than fetched on demand.

Two bounds are reported because the truth is between them and the gap is wide:
  pessimistic  any layer with >=1 miss pays the full CPU-layer cost
  optimistic   a layer pays only SLOPE/8 per missing expert
"""
import json
import os
import sys
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA", os.path.join(HERE, "m2_hidden.npz"))
OUT = os.path.join(HERE, "m3_simulate_speedup_result.json")

SLOPE_MS = 0.3208          # ms/token per expert layer left on the CPU
FLOOR_MS = 5.876           # ms/token with every expert GPU-resident
FETCH_MS = 0.05706         # one 2.92 MiB expert over PCIe at 53.66 GB/s
CPU_EXPERT_MS = SLOPE_MS / 8
TODAY_MS = float(os.environ.get("TODAY_MS", "12.84"))   # measured Q4_K_M, split 22
CAPACITIES = [int(c) for c in os.environ.get("CAPACITIES", "32,66,96").split(",")]
TOP_P = int(os.environ.get("TOP_P", "8"))
N_COMP, N_EXPERT, K = 256, 128, 8


def fit_probes(H, E, keys, idx, reg, train_r, n_layers):
    """Per-layer ridge probe h_L -> E_{L+1}, fitted on the training registers."""
    tr = np.isin(reg, train_r)
    mu = H[tr].mean(0)
    C = ((H[tr] - mu).T @ (H[tr] - mu)) / max(tr.sum() - 1, 1)
    w, V = np.linalg.eigh(C.astype(np.float64))
    comp = V[:, ::-1][:, :N_COMP].astype(np.float32)
    Hp = (H - mu) @ comp
    Hp /= (Hp[tr].std(0) + 1e-6)

    probes = {}
    for L in range(n_layers - 1):
        src, dst = [], []
        for (p, t, l), i in idx.items():
            if l == L and (p, t, L + 1) in idx:
                src.append(i)
                dst.append(idx[(p, t, L + 1)])
        if len(src) < 100:
            continue
        src, dst = np.array(src), np.array(dst)
        m = np.isin(reg[src], train_r)
        if m.sum() < 50:
            continue
        X = Hp[src][m]
        Y = np.zeros((m.sum(), N_EXPERT), dtype=np.float32)
        Y[np.arange(m.sum())[:, None], E[dst][m]] = 1.0
        G = X @ X.T
        A = np.linalg.solve(G + 100.0 * np.eye(len(G), dtype=np.float32), Y)
        probes[L] = X.T @ A
    return Hp, probes


def simulate(seq, policy, capacity, probes, Hp, n_layers):
    """seq: list of (key_index, layer -> expert ids) per token, in order.

    Returns (per-expert hit rate, fraction of layers fully resident, fetches).
    """
    cache = {L: OrderedDict() for L in range(n_layers)}
    hits = total = full_layers = layer_count = fetches = 0

    for step in seq:
        for L in sorted(step["layers"]):
            ids = step["layers"][L]
            # --- prefetch for this layer, decided while layer L-1 was computing
            if policy in ("lru+predict", "oracle") and L - 1 in probes:
                if policy == "oracle":
                    pred = ids
                else:
                    h = Hp[step["row"][L - 1]] if (L - 1) in step["row"] else None
                    if h is None:
                        pred = []
                    else:
                        sc = h @ probes[L - 1]
                        pred = np.argpartition(-sc, TOP_P)[:TOP_P]
                for e in pred:
                    e = int(e)
                    if e not in cache[L]:
                        fetches += 1
                        cache[L][e] = None
                        if len(cache[L]) > capacity:
                            cache[L].popitem(last=False)
                    else:
                        cache[L].move_to_end(e)

            miss = 0
            for e in ids:
                e = int(e)
                total += 1
                if e in cache[L]:
                    hits += 1
                    cache[L].move_to_end(e)
                else:
                    miss += 1
                    if policy != "none":      # a miss is computed on CPU, then cached
                        cache[L][e] = None
                        if len(cache[L]) > capacity:
                            cache[L].popitem(last=False)
            layer_count += 1
            if miss == 0:
                full_layers += 1

    return (hits / max(total, 1), full_layers / max(layer_count, 1), fetches, layer_count)


def to_tok_s(expert_hit, full_layer_rate, n_layers, host_layers=22):
    """Two bounds on ms/token, per AI2's model."""
    # pessimistic: a layer with any miss costs the full CPU-layer slope
    pess_ms = FLOOR_MS + (1 - full_layer_rate) * n_layers * SLOPE_MS
    # optimistic: only the missing experts cost, at SLOPE/8 each
    opt_ms = FLOOR_MS + (1 - expert_hit) * n_layers * K * CPU_EXPERT_MS
    return 1000 / pess_ms, 1000 / opt_ms


def main():
    if not os.path.exists(DATA):
        sys.exit(f"missing {DATA} -- run m2_capture.py first")
    d = np.load(DATA, allow_pickle=True)
    H, E, keys, reg = (d["hidden"].astype(np.float32), d["experts"].astype(np.int64),
                       d["keys"], d["register"].astype(str))
    pid, tok, lay = keys[:, 0], keys[:, 1], keys[:, 2]
    idx = {(p, t, l): i for i, (p, t, l) in enumerate(zip(pid, tok, lay))}
    n_layers = int(lay.max()) + 1
    regs = sorted(set(reg))
    train_r, test_r = regs[:5], regs[6:]
    print(f"{len(H):,} rows; train registers {train_r}; simulating on {test_r}",
          file=sys.stderr)

    Hp, probes = fit_probes(H, E, keys, idx, reg, train_r, n_layers)
    print(f"fitted {len(probes)} per-layer probes", file=sys.stderr)

    # Build per-token steps for the test registers, in token order within a prompt.
    steps = []
    test_mask = np.isin(reg, test_r)
    by_tok = {}
    for i in np.where(test_mask)[0]:
        p, t, l = keys[i]
        by_tok.setdefault((p, t), {"layers": {}, "row": {}})
        by_tok[(p, t)]["layers"][int(l)] = E[i]
        by_tok[(p, t)]["row"][int(l)] = i
    for key in sorted(by_tok):
        steps.append(by_tok[key])
    print(f"{len(steps):,} test tokens x {n_layers} layers", file=sys.stderr)

    rows = []
    for cap in CAPACITIES:
        for policy in ("none", "lru", "lru+predict", "oracle"):
            eh, fl, fet, lc = simulate(steps, policy, cap, probes, Hp, n_layers)
            pess, opt = to_tok_s(eh, fl, n_layers)
            rows.append({"capacity": cap, "policy": policy, "expert_hit": eh,
                         "full_layer_rate": fl, "fetches_per_token": fet / max(len(steps), 1),
                         "pessimistic_tok_s": pess, "optimistic_tok_s": opt})
            print(f"  cap {cap:3d} {policy:<12} expert-hit {100 * eh:5.1f}%  "
                  f"all-8-resident {100 * fl:5.1f}%  -> {pess:5.1f}-{opt:5.1f} tok/s",
                  file=sys.stderr, flush=True)

    print(f"\n=== MILESTONE 3 GATE: SIMULATED SPEEDUP (Q4_K_M, today {TODAY_MS:.2f} ms/token "
          f"= {1000 / TODAY_MS:.1f} tok/s) ===\n")
    print("%4s %-13s %11s %15s %18s" % ("cap", "policy", "expert hit",
                                        "all-8 resident", "tok/s (pess-opt)"))
    for r in rows:
        print("%4d %-13s %10.1f%% %14.1f%% %8.1f - %-7.1f"
              % (r["capacity"], r["policy"], 100 * r["expert_hit"],
                 100 * r["full_layer_rate"], r["pessimistic_tok_s"], r["optimistic_tok_s"]))

    print("\nThe bar is 110 tok/s (ud-q3_k_xl already does that at Q4_K_M-equal accuracy).")
    for cap in CAPACITIES:
        lru = next(r for r in rows if r["capacity"] == cap and r["policy"] == "lru")
        prd = next(r for r in rows if r["capacity"] == cap and r["policy"] == "lru+predict")
        orc = next(r for r in rows if r["capacity"] == cap and r["policy"] == "oracle")
        print(f"  cap {cap:3d}: prediction adds {100 * (prd['full_layer_rate'] - lru['full_layer_rate']):+.1f} "
              f"points of all-8-resident over LRU alone; a perfect oracle would add "
              f"{100 * (orc['full_layer_rate'] - lru['full_layer_rate']):+.1f}")

    json.dump({"today_ms": TODAY_MS, "constants": {"slope_ms": SLOPE_MS, "floor_ms": FLOOR_MS,
                                                   "fetch_ms": FETCH_MS},
               "test_registers": test_r, "rows": rows}, open(OUT, "w"), indent=1, default=float)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
