"""
P9: if misses were served while the GPU keeps working, how much would still stall?

Today a token pays for every miss in full: the GPU stops, the CPU computes the
missed experts, the GPU resumes. The design in
docs/superpowers/specs/2026-10-01-graph-resident-decode-design.md lets the CPU
serve a layer's misses WHILE the GPU runs that layer's cached experts, so the
token only waits for whatever CPU time is left over once the GPU work is done:

    stall(layer) = max(0, x + m * tau - K)        when the layer has m >= 1 misses

    m    misses in this layer for this token
    tau  CPU time for one missed expert (x2_cpu_expert_cost: memory-bound)
    K    GPU time of the cached-expert chain, the window the CPU can hide in
    x    latency of the exchange itself (x1_mapped_coprocessor)

The total number of misses stops being the quantity that matters. What matters
is how they are DISTRIBUTED over layers: one miss in each of 30 layers hides
completely, thirty misses in one layer do not. This replays the real decode
traces through the same cache (LRU eviction, capacity and admission as given)
and records the per-(token, layer) miss count, which is the one thing the cost
model never looked at.

It also asks what a cheap "warm the CPU cache one layer ahead" prefetch could
rely on: how many misses are of an expert that missed in the same layer within
the last few tokens (a ghost list), and how precise that list is.

Same caveat as every replay here: PrefetchEnv is not trace-validated, so read
the SHAPE of the distribution and the ratios. On 2026-09-30 it predicted the
upload and hit rates of first-miss and 2-in-8 admission within a few percent of
the engine, and got the heat gate wrong.

    nice -n 19 python3 experiments/p9_stall_model.py
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
from p8_admission import FirstMiss, KInWindow  # noqa: E402

ART = os.path.join(ROOT, "artifacts")
GHOST_WINDOWS = (1, 2, 4, 8)


def replay(ix, rows, decode_ok, cost, policy, capacity, max_inserts=2):
    d = ix.d
    env = PrefetchEnv(capacity=capacity, max_inserts=max_inserts, cost=cost, n_layers=ix.n_layers)
    policy.reset(ix.n_layers)
    hist = np.zeros((ix.n_layers, 9), dtype=np.int64)        # [layer][m] over decode tokens
    per_token_m = []                                         # list of per-layer miss vectors
    cur_vec = np.zeros(ix.n_layers, dtype=np.int16)
    seen_layer = np.zeros(ix.n_layers, dtype=bool)
    # ghost: expert -> last token it MISSED in this layer
    ghost = [dict() for _ in range(ix.n_layers)]
    g_hit = {w: 0 for w in GHOST_WINDOWS}                    # misses that were on the ghost list
    g_size = {w: 0 for w in GHOST_WINDOWS}                   # ghost-list entries offered (for precision)
    n_miss = 0
    last, t = None, 0

    def close(key):
        if decode_ok.get(key, True):
            for L in range(ix.n_layers):
                if seen_layer[L]:
                    hist[L, min(int(cur_vec[L]), 8)] += 1
            per_token_m.append(cur_vec.copy())

    for r in rows:
        L = int(d["layer"][r])
        key = (int(d["prompt"][r]), int(d["pos"][r]))
        if key != last:
            if last is not None:
                env.step_boundary()
                close(last)
                t += 1
            cur_vec[:] = 0
            seen_layer[:] = False
            last = key
        cur = [int(e) for e in d["cur"][r] if e >= 0]
        misses = [e for e in cur if e not in env.resident[L]]
        seen_layer[L] = True
        cur_vec[L] = len(misses)
        if decode_ok.get(key, True):
            g = ghost[L]
            for w in GHOST_WINDOWS:
                # entries a host would have warmed: missed here within w tokens, still not resident
                g_size[w] += sum(1 for e, tt in g.items() if t - tt <= w and e not in env.resident[L])
            for e in misses:
                n_miss += 1
                tt = g.get(e)
                for w in GHOST_WINDOWS:
                    if tt is not None and t - tt <= w:
                        g_hit[w] += 1
        for e in misses:
            ghost[L][e] = t
        if len(ghost[L]) > 512:
            for e in [e for e, tt in ghost[L].items() if t - tt > 16]:
                del ghost[L][e]
        env.observe(L, cur)
        policy.use(L, cur, t)
        env.admit_demand(L, policy.admit(env, L, misses, t))
    env.step_boundary()
    close(last)
    return hist, np.array(per_token_m), {
        "misses": n_miss,
        "ghost_recall": {w: g_hit[w] / max(n_miss, 1) for w in GHOST_WINDOWS},
        "ghost_precision": {w: g_hit[w] / max(g_size[w], 1) for w in GHOST_WINDOWS},
        "ghost_offered_per_token": {w: g_size[w] / max(len(per_token_m), 1) for w in GHOST_WINDOWS},
    }


def stall_ms(per_token_m, tau, K, x):
    m = per_token_m.astype(np.float64)
    st = np.where(m > 0, np.maximum(0.0, x + m * tau - K), 0.0)
    return float(st.sum(axis=1).mean()) / 1000.0


def main():
    ix = Index(os.path.join(ROOT, "data", "index-v4-replay.npz"))
    d = ix.d
    rows = ix.rows(1)
    rows = rows[np.lexsort((d["layer"][rows], d["pos"][rows], d["prompt"][rows]))]
    dec = phase_mask(ix, os.path.join(ROOT, "data", "corpus-v4-replay"))
    decode_ok = {(int(d["prompt"][r]), int(d["pos"][r])): bool(dec[r]) for r in rows}
    cost = fit_cost_model()

    out = {}
    for label, policy in (("first-miss", FirstMiss()), ("2-in-8", KInWindow(2, 8))):
        hist, ptm, gh = replay(ix, rows, decode_ok, cost, policy, 56)
        tot = hist.sum(axis=1, keepdims=True).clip(min=1)
        pooled = hist.sum(axis=0) / hist.sum()
        m_tok = float(ptm.sum(axis=1).mean())
        layers_missing = float((ptm > 0).sum(axis=1).mean())
        excess = float(np.maximum(ptm - 1, 0).sum(axis=1).mean())
        print(f"\n=== 56 slots, admission {label}: {len(ptm)} decode tokens ===")
        print(f"misses per token {m_tok:.1f}; layers with at least one miss {layers_missing:.1f} of {ix.n_layers}; "
              f"misses beyond the first in their layer {excess:.1f}")
        print("share of (token, layer) pairs by miss count:  " +
              "  ".join(f"m={k}: {100 * pooled[k]:.1f}%" for k in range(6)) + f"  m>=6: {100 * pooled[6:].sum():.2f}%")
        worst = np.argsort(-(hist[:, 2:].sum(axis=1) / tot[:, 0]))[:6]
        print("layers that most often have 2+ misses: " +
              ", ".join(f"L{int(L)} ({100 * hist[L, 2:].sum() / tot[L, 0]:.0f}%)" for L in worst))
        print(f"\n  stall left per token (ms) if the CPU serves misses while the GPU runs the cached chain")
        print(f"  {'tau us':>7s} {'K us':>6s} {'x us':>5s} {'today: m*tau':>13s} {'overlapped':>11s} {'hidden':>7s}")
        grid = []
        for tau in (30.0, 46.0, 58.0):
            for K in (25.0, 40.0, 55.0):
                for x in (3.0, 10.0):
                    serial = m_tok * tau / 1000.0
                    st = stall_ms(ptm, tau, K, x)
                    grid.append({"tau": tau, "K": K, "x": x, "serial_ms": serial, "stall_ms": st})
                    print(f"  {tau:7.0f} {K:6.0f} {x:5.0f} {serial:13.2f} {st:11.2f} {100 * (1 - st / serial):6.0f}%")
        print("\n  ghost list (experts that missed in this layer within the last w tokens and are still not resident):")
        for w in GHOST_WINDOWS:
            print(f"    w={w}: covers {100 * gh['ghost_recall'][w]:.1f}% of misses, "
                  f"precision {100 * gh['ghost_precision'][w]:.1f}%, {gh['ghost_offered_per_token'][w]:.1f} entries per token")
        out[label] = {"tokens": int(len(ptm)), "misses_per_token": m_tok, "layers_with_miss": layers_missing,
                      "excess_misses_per_token": excess, "pooled_hist": pooled.tolist(),
                      "per_layer_hist": hist.tolist(), "stall_grid": grid, "ghost": gh}
    json.dump(out, open(os.path.join(ART, "p9_stall_model.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
