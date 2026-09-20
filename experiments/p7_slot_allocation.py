"""
P7 / Phase 2 (N2.1, N2.2): spend the same VRAM better.

PLAN-NEXT.md's Phase 2: every layer gets the same 72 slots, but per-layer
decode miss rates vary 5.2x (L0 misses 1.354 experts of 8, L17 misses 0.261),
so uniform allocation is spending identical VRAM on layers with very
different needs.

Caveat this script does not hide: Phase 0 (PLAN-NEXT.md) -- an engine trace
replayed against this same environment, to prove the simulator matches
measured throughput within +/-1 tok/s -- was never done. The environment is
already known to run ~10% optimistic end to end (docs in this repo). What
*is* trustworthy here is the RELATIVE comparison across layers: layers never
share state in PrefetchEnv (independent resident sets, independent free
pools), so whatever bias the simulator carries applies the same way to every
layer, and the miss-rate ranking between layers should survive it even if
the absolute tok/s number should not be believed. N2.4 is what actually
decides this: the allocation proposed here gets measured end to end in the
real engine (r3_real_bench.py), and that paired result is the one that
counts, not this one.

Method for N2.1: layers are independent, so one full corpus pass at a given
uniform capacity c gives every layer's hit rate at c in a single run --
not one pass per layer. Sweep c over a small grid, record each layer's
miss(c), and take a discrete derivative for the marginal value of the next
slot at the shipped operating point.

N2.2: greedy allocation. Total slot budget held at the current VRAM
footprint (n_layers * 72, uniform baseline); repeatedly hand the next slot
to whichever layer's marginal-miss-reduction-per-slot is currently largest,
using the fitted per-layer curves. Curves are expected to be concave (plan's
own falsification condition), so greedy on a concave objective is optimal
for a fixed sum, not just "almost certainly enough".
"""
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
# PLAN-NEXT.md measured the uniform baseline at 72, but this machine's GPU is
# currently sharing VRAM with a live desktop session (compositor, browser,
# file manager -- ~540 MiB and rising) and a manual check found uniform 72
# leaves only ~512 MiB before cuBLAS's own handle allocation, which is not
# enough: cublasCreate_v2 fails with "the resource allocation failed" on the
# very first decode step, 100% reproducible. Uniform 56 measures at 9040 MiB
# and has run stably all session (Phase 1's paired benchmark, the online-
# learning verification run). N2.4 needs the real engine to actually come up,
# so the budget this script optimises is 56, not 72 -- same method, smaller
# total, chosen for what this GPU can currently hold rather than what an
# earlier session's freer VRAM allowed.
BASELINE_CAP = 56
GRID = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88]


def sweep(ix, rows, decode_ok, cost, grid=GRID, max_inserts=2):
    """One full corpus pass per grid point, uniform capacity. Returns
    per_layer_miss[c][layer] = decode misses / lookups at that layer, and
    the aggregate tok/s at each point as a sanity check that this reproduces
    the known uniform-capacity behaviour."""
    d = ix.d
    out = {}
    for c in grid:
        env = PrefetchEnv(capacity=c, max_inserts=max_inserts, cost=cost,
                          n_layers=ix.n_layers)
        dstats = [{"hit": 0, "lookup": 0} for _ in range(ix.n_layers)]
        agg = {"hit": 0, "lookup": 0, "uploads": 0}
        dtok = 0
        prev_layer = [dict(env.layer_stats[L]) for L in range(ix.n_layers)]
        prev_agg = dict(env.stats)
        last = None
        for r in rows:
            L = int(d["layer"][r])
            key = (int(d["prompt"][r]), int(d["pos"][r]))
            if key != last:
                if last is not None:
                    env.step_boundary()
                    if decode_ok is None or decode_ok.get(last, True):
                        for Ln in range(ix.n_layers):
                            for k in dstats[Ln]:
                                dstats[Ln][k] += (env.layer_stats[Ln][k]
                                                  - prev_layer[Ln][k])
                        for k in agg:
                            agg[k] += env.stats[k] - prev_agg[k]
                        dtok += 1
                    prev_layer = [dict(env.layer_stats[Ln]) for Ln in range(ix.n_layers)]
                    prev_agg = dict(env.stats)
                last = key
            cur = [int(e) for e in d["cur"][r] if e >= 0]
            env.observe(L, cur)
            env.admit_demand(L, cur)
        env.step_boundary()
        if decode_ok is None or decode_ok.get(last, True):
            for Ln in range(ix.n_layers):
                for k in dstats[Ln]:
                    dstats[Ln][k] += env.layer_stats[Ln][k] - prev_layer[Ln][k]
            for k in agg:
                agg[k] += env.stats[k] - prev_agg[k]
            dtok += 1

        hit = agg["hit"] / max(agg["lookup"], 1)
        missed = ix.n_layers * 8 * (1 - hit)
        up = agg["uploads"] / max(dtok, 1)
        ms = cost["A_ms"] + cost["B_ms_per_miss"] * missed + cost["C_ms_per_upload"] * up
        per_layer_miss = [
            (dstats[Ln]["lookup"] - dstats[Ln]["hit"]) / max(dstats[Ln]["lookup"], 1)
            * 8.0  # average misses per visit, on the same "of 8" scale as PLAN-NEXT.md
            for Ln in range(ix.n_layers)
        ]
        out[c] = {"tok_s": 1000.0 / max(ms, 1e-6), "hit": hit,
                  "uploads_per_token": up, "per_layer_miss8": per_layer_miss,
                  "tokens": dtok}
        print(f"  c={c:3d}  tok/s {out[c]['tok_s']:6.1f}  hit {100*hit:5.1f}%  "
              f"L0 miss/8 {per_layer_miss[0]:.3f}  worst layer "
              f"{max(range(ix.n_layers), key=lambda L: per_layer_miss[L])}"
              f"={max(per_layer_miss):.3f}", flush=True)
    return out


def greedy_allocate(per_layer_miss8, grid, n_layers, total_budget, frozen_layers,
                     frozen_cap):
    """Marginal value of the slot that would move layer L from its current
    grid point to the next one, in misses-of-8 avoided. Concave curves ->
    greedy on the steepest available step is optimal for a fixed total.

    `frozen_layers` never move: absence of a marginal-value estimate is not
    evidence the layer needs nothing, it means this corpus has no rows for
    it (see the layer-47 note in main()) and reallocating its budget away
    would be starving a real layer on missing data, not on a measurement."""
    idx = [0] * n_layers        # position in `grid` layer L currently sits at
    cap = [grid[0]] * n_layers
    for L in frozen_layers:
        cap[L] = frozen_cap
    remaining = total_budget - sum(cap)
    if remaining < 0:
        raise ValueError("initial allocation already exceeds the budget")

    def step_value(L):
        if L in frozen_layers:
            return -1.0, 0
        i = idx[L]
        if i + 1 >= len(grid):
            return -1.0, 0  # no more room on the measured grid
        d_slots = grid[i + 1] - grid[i]
        d_miss = per_layer_miss8[grid[i]][L] - per_layer_miss8[grid[i + 1]][L]
        return d_miss / d_slots, d_slots  # miss-of-8 avoided per slot spent

    import heapq
    heap = []
    for L in range(n_layers):
        v, cost_slots = step_value(L)
        if v > 0 and cost_slots <= remaining:
            heapq.heappush(heap, (-v, L))
    while heap and remaining > 0:
        negv, L = heapq.heappop(heap)
        i = idx[L]
        d_slots = grid[i + 1] - grid[i]
        if d_slots > remaining:
            continue
        idx[L] += 1
        cap[L] = grid[idx[L]]
        remaining -= d_slots
        v, cost_slots = step_value(L)
        if v > 0 and cost_slots <= remaining:
            heapq.heappush(heap, (-v, L))
    return cap, remaining


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

    print("=== N2.1: per-layer miss(c) sweep, decode-only rows ===")
    print(f"n_layers={ix.n_layers}  decode rows={sum(1 for r in rows if decode_ok.get((int(d['prompt'][r]), int(d['pos'][r])), True))}/{len(rows)}")
    swept = sweep(ix, rows, decode_ok, cost)

    # The corpus records a row for layer L only when L predicts a "below"
    # target at L+1, so the physically-last engine layer never appears as a
    # source row -- it is not that it never misses, it is that this npz has
    # no data for it at all. Confirmed against the model: qwen3moe.block_count
    # = 48 in the GGUF, i.e. ix.n_layers real layers all exist in the engine,
    # but np.unique(d["layer"]) is missing whichever index sits at the top.
    # Reallocating that layer's budget away on the strength of zero measured
    # misses would be starving a real layer on absence of evidence, not on
    # evidence of no need -- so it is frozen at the uniform baseline instead.
    present = set(int(x) for x in np.unique(d["layer"]))
    frozen_layers = [L for L in range(ix.n_layers) if L not in present]
    if frozen_layers:
        print(f"\nWARNING: corpus has zero rows for layer(s) {frozen_layers} "
              f"-- frozen at uniform {BASELINE_CAP}, excluded from reallocation")

    per_layer_miss8 = {c: swept[c]["per_layer_miss8"] for c in swept}
    baseline = swept[BASELINE_CAP]
    print(f"\nbaseline (uniform {BASELINE_CAP}): tok/s {baseline['tok_s']:.1f} "
          f"hit {100*baseline['hit']:.1f}%")
    scored = [baseline["per_layer_miss8"][L] for L in range(ix.n_layers) if L not in frozen_layers]
    spread = max(scored) / max(min(scored), 1e-6)
    print(f"per-layer miss/8 spread at baseline (measured layers only): {spread:.1f}x "
          f"(worst {max(scored):.3f}, best {min(scored):.3f})")

    total_budget = ix.n_layers * BASELINE_CAP
    print(f"\n=== N2.2: greedy re-allocation, fixed total = {total_budget} slots ===")
    new_cap, leftover = greedy_allocate(per_layer_miss8, GRID, ix.n_layers, total_budget,
                                        frozen_layers, BASELINE_CAP)
    print(f"leftover budget unspent (ran off the measured grid): {leftover} slots")
    changed = [(L, BASELINE_CAP, new_cap[L]) for L in range(ix.n_layers) if new_cap[L] != BASELINE_CAP]
    changed.sort(key=lambda x: x[2] - x[1])
    print(f"{len(changed)} of {ix.n_layers} layers moved off uniform")
    for L, old, new in changed:
        print(f"  layer {L:2d}: {old} -> {new}  ({new - old:+d})")

    # N2.3: measure the allocation's effect in THIS environment (still under
    # the Phase-0 caveat above), then emit the LLAMA_MOE_SLOT_PROFILE string
    # for the real end-to-end run.
    print("\n=== predicted effect of the new allocation, same environment ===")
    env = PrefetchEnv(capacity=new_cap, cost=cost, n_layers=ix.n_layers)
    dstats = {"hit": 0, "lookup": 0, "uploads": 0}
    dtok = 0
    prev = dict(env.stats)
    last = None
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
            last = key
        cur = [int(e) for e in d["cur"][r] if e >= 0]
        env.observe(L, cur)
        env.admit_demand(L, cur)
    env.step_boundary()
    if decode_ok.get(last, True):
        for k in dstats:
            dstats[k] += env.stats[k] - prev[k]
        dtok += 1
    hit = dstats["hit"] / max(dstats["lookup"], 1)
    missed = ix.n_layers * 8 * (1 - hit)
    up = dstats["uploads"] / max(dtok, 1)
    ms = cost["A_ms"] + cost["B_ms_per_miss"] * missed + cost["C_ms_per_upload"] * up
    new_tok_s = 1000.0 / max(ms, 1e-6)
    print(f"reallocated: tok/s {new_tok_s:.1f} (vs uniform {baseline['tok_s']:.1f}, "
          f"{new_tok_s - baseline['tok_s']:+.1f})  hit {100*hit:.1f}%")
    print("This number carries the same Phase-0 optimism bias as every other "
          "simulated figure in this repo -- it says direction, not magnitude. "
          "N2.4 (r3_real_bench.py, paired, in the real engine) is the actual test.")

    profile = ",".join(f"{L}:{new_cap[L]}" for L, old, new in
                       [(i, BASELINE_CAP, new_cap[i]) for i in range(ix.n_layers)]
                       if new != old)
    print(f"\nLLAMA_MOE_SLOT_PROFILE=\"{profile}\"")

    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "p7_slot_allocation.json"), "w") as f:
        json.dump({
            "grid": GRID, "baseline_cap": BASELINE_CAP, "total_budget": total_budget,
            "per_layer_miss8_by_cap": per_layer_miss8,
            "baseline": baseline, "new_cap": new_cap, "leftover": leftover,
            "changed_layers": changed,
            "predicted": {"tok_s": new_tok_s, "hit": hit, "uploads_per_token": up,
                         "gain_vs_baseline": new_tok_s - baseline["tok_s"]},
            "slot_profile_env": profile,
            "caveat": "Phase 0 (trace-validated environment) was never completed; "
                      "this is a relative, not absolute, comparison. N2.4 end-to-end "
                      "engine measurement is authoritative.",
        }, f, indent=1)
    print(f"\nwrote artifacts/p7_slot_allocation.json")


if __name__ == "__main__":
    main()
