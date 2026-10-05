"""P10: how predictable are the experts a 56-slot cache will MISS, and how far ahead?

Reads the dumps written by cpp/moe-route-dump.cpp (one per prompt) and the
model's router weights, replays every layer through an LRU cache with the
measured slot profile (admit on first miss), and for each decode step and
layer T scores predictors made d layers earlier:

  gate-ahead     W_T . ffn_norm_{T-d}          (FATE: the router of T applied to an earlier router input)
  renorm-ahead   W_T . rmsnorm(ffn_inp_{T-d}) * g_T   (same, normalised with T's own norm weight)
  embedding      W_T . attn_norm_0             (layers 0..3 only: available as soon as the token is known)
  temporal       the experts T used last token (set, no ranking)

A prefetcher may send B experts per layer. Only non-resident experts are worth
sending, so a predictor's candidates are its top-B NON-RESIDENT experts.
Reported per horizon and budget: recall of the misses that actually happen,
precision of what is sent, and misses per token for scale.

Usage: python3 experiments/p10_lookahead_probe.py <dump dir> [warmup steps, default 64]
Writes artifacts/p10_lookahead_probe.json.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine" / "gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

MODEL = os.environ.get("MODEL", str(Path.home() / ".lmstudio/models/lmstudio-community/"
                       "Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"))
PROFILE = ("0:88,1:88,3:72,5:64,6:48,7:40,10:72,11:64,12:64,13:64,14:48,15:48,18:40,19:40,20:40,"
           "23:64,24:64,25:64,27:48,30:40,31:40,32:40,34:64,35:64,36:64,37:64,40:48,43:40,44:48,45:48,46:48")
DEFAULT_SLOTS = 56
HORIZONS = (1, 2, 3, 4)
BUDGETS = (1, 2, 3, 4, 6)
RANKS = (4, 6, 8, 10, 12)            # send non-resident experts the lookahead router ranks in its top K
THRESHOLDS = (0.02, 0.05, 0.10, 0.20)  # send non-resident experts with lookahead probability >= p
EPS = 1e-6

# Timing model for the schedule simulation (measured, see the 2026-10-01 spec):
LAYER_US = 103.0      # GPU work per layer: attention + router 58, cached experts 45
ROUTER_US = 58.0      # from layer start until its routing (and table lookup) happens
COPY_US = 60.0        # one 2.7 MiB expert over the copy engine (45.7 GB/s, probe x1)
MISS_US = 47.0        # one missed expert on 8 CPU threads (probe x2)
MISS_FIXED_US = float(os.environ.get("MISS_FIXED_US", "40"))
# per layer that missed, on top of 47 us an expert: the sidecar's serve log
# measured 120 us of CPU per missing layer at about 1.7 misses each (S3, 2026-10-04)
WINDOW_US = 45.0      # cached-expert work the CPU's answer hides behind
EXCHANGE_US = 5.0
SCHED_K = (6, 8)
SCHED_HORIZONS = ((1,), (2,), (2, 1), (3, 2, 1))


def stall_us(m):
    return max(0.0, MISS_US * m + ((EXCHANGE_US + MISS_FIXED_US) if m else 0.0) - WINDOW_US)


def schedule_sim(cands, miss, n_layer):
    """cands[(T, d)] -> bool [n_expert] candidates predicted d layers ahead; miss[T] -> bool [n_expert].
    One token. EDF over a single copy engine; a copy that cannot land before its layer's lookup is dropped.
    Returns (stall without prefetch, stall with, copies landed, useful landed, misses converted)."""
    jobs = []  # (issue_time, deadline, T, expert)
    for (T, d), c in cands.items():
        issue = (T - d) * LAYER_US + ROUTER_US
        deadline = T * LAYER_US + ROUTER_US
        for e in np.flatnonzero(c):
            jobs.append((issue, deadline, T, int(e)))
    jobs.sort()
    landed = {T: set() for T in range(n_layer)}
    queued = set()
    t_engine, i, pending = 0.0, 0, []
    import heapq
    while i < len(jobs) or pending:
        if not pending and i < len(jobs) and jobs[i][0] > t_engine:
            t_engine = jobs[i][0]
        while i < len(jobs) and jobs[i][0] <= t_engine:
            issue, dl, T, e = jobs[i]; i += 1
            if (T, e) not in queued:
                queued.add((T, e)); heapq.heappush(pending, (dl, T, e))
        if not pending:
            continue
        dl, T, e = heapq.heappop(pending)
        if t_engine + COPY_US > dl:
            continue                  # would land too late: drop it
        t_engine += COPY_US
        landed[T].add(e)
    base = sum(stall_us(int(miss[T].sum())) for T in range(n_layer))
    conv = useful = n_landed = 0
    with_pf = 0.0
    for T in range(n_layer):
        got = np.zeros_like(miss[T]); got[list(landed[T])] = True
        conv += int((got & miss[T]).sum()); n_landed += int(got.sum())
        with_pf += stall_us(int((miss[T] & ~got).sum()))
    return base, with_pf, n_landed, conv, int(sum(m.sum() for m in miss.values()))


def load_dump(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"MRD1"
        n_layer, n_embd, n_used = np.frombuffer(f.read(12), dtype=np.int32)
        rec = np.dtype([("attn0", np.float16, n_embd),
                        ("layers", [("inp", np.float16, n_embd), ("norm", np.float16, n_embd),
                                    ("ids", np.int32, n_used)], n_layer)])
        data = np.frombuffer(f.read(), dtype=rec)
    return int(n_layer), int(n_embd), int(n_used), data


def load_weights(n_layer):
    r = GGUFReader(MODEL)
    t = {x.name: x for x in r.tensors}

    def get(name):
        x = t[name]
        a = dequantize(x.data, x.tensor_type)
        return np.asarray(a, dtype=np.float32).reshape([int(v) for v in reversed(x.shape)])

    gate = [get(f"blk.{l}.ffn_gate_inp.weight") for l in range(n_layer)]   # [n_expert, n_embd]
    norm = [get(f"blk.{l}.ffn_norm.weight") for l in range(n_layer)]      # [n_embd]
    eps = float(r.fields["qwen3moe.attention.layer_norm_rms_epsilon"].parts[-1][0]) \
        if "qwen3moe.attention.layer_norm_rms_epsilon" in r.fields else EPS
    return gate, norm, eps


def lru_misses(ids, n_slots):
    """ids: [steps, n_used] for one layer, one prompt. Returns (resident-before mask [steps, n_expert], misses list)."""
    n_expert = 128
    last = np.full(n_expert, -1, dtype=np.int64)    # step of last use, -1 = not resident
    resident = np.zeros((len(ids), n_expert), dtype=bool)
    for s, row in enumerate(ids):
        resident[s] = last >= 0
        for e in row:
            if last[e] < 0 and (last >= 0).sum() >= n_slots:
                victim = np.where(last >= 0, last, np.iinfo(np.int64).max).argmin()
                last[victim] = -1
            last[e] = s
    return resident


def main():
    ddir = Path(sys.argv[1])
    warm = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    dumps = sorted(ddir.glob("*.mrd"))
    n_layer, n_embd, n_used, _ = load_dump(dumps[0])
    gate, normw, eps = load_weights(n_layer)
    slots = {l: DEFAULT_SLOTS for l in range(n_layer)}
    for kv in PROFILE.split(","):
        l, s = kv.split(":")
        slots[int(l)] = int(s)

    # accumulators: key -> [hits, sent, misses]
    acc = {}
    sanity = [0, 0]
    total_miss, total_steps = 0, 0
    sim = {}  # (K, horizons) -> [base_us, with_us, landed, converted, misses, tokens]
    per_layer_miss = np.zeros(n_layer)

    def add_mask(key, send, actual_miss_mask):
        hit = send & actual_miss_mask
        a = acc.setdefault(key, [0, 0, 0])
        a[0] += int(hit.sum()); a[1] += int(send.sum()); a[2] += int(actual_miss_mask.sum())

    def add_ranked(name, dlt, logits, resident, miss):
        order = np.argsort(-logits, axis=1)
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(logits.shape[1])[None, :].repeat(len(logits), 0), axis=1)
        for K in RANKS:
            add_mask((name + f" top{K}", dlt, 0), (rank < K) & ~resident, miss)
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        for t in THRESHOLDS:
            add_mask((name + f" p>={t}", dlt, 0), (p >= t) & ~resident, miss)

    def add(key, scores, resident, actual_miss_mask, budget):
        # top-`budget` non-resident experts by score, per step
        s = np.where(resident, -np.inf, scores)
        top = np.argsort(-s, axis=1)[:, :budget]
        sent = np.take_along_axis(~resident, top, axis=1)          # only count real candidates
        hit = np.take_along_axis(actual_miss_mask, top, axis=1) & sent
        a = acc.setdefault(key, [0, 0, 0])
        a[0] += int(hit.sum()); a[1] += int(sent.sum()); a[2] += int(actual_miss_mask.sum())

    for path in dumps:
        _, _, _, d = load_dump(path)
        L = d["layers"]
        steps = len(d)
        if steps <= warm + 8:
            continue
        ev = slice(warm, steps)
        total_steps += steps - warm
        attn0 = d["attn0"].astype(np.float32)[ev]
        cand_store = {}   # (K, T, d) -> bool [eval steps, 128]
        miss_store = {}
        for T in range(n_layer):
            ids_T = L["ids"][:, T]
            resident = lru_misses(ids_T, slots[T])[ev]
            actual = np.zeros((steps, 128), dtype=bool)
            np.put_along_axis(actual, ids_T, True, axis=1)
            actual = actual[ev]
            miss = actual & ~resident
            total_miss += int(miss.sum())
            miss_store[T] = miss
            per_layer_miss[T] += miss.sum()
            W = gate[T]
            # sanity: the true router input reproduces the routed set
            exact = L["norm"][:, T].astype(np.float32)[ev] @ W.T
            top8 = np.argsort(-exact, axis=1)[:, :n_used]
            sanity[0] += int(np.take_along_axis(actual, top8, axis=1).sum()); sanity[1] += actual.sum()
            for dlt in HORIZONS:
                src = T - dlt
                if src < 0:
                    continue
                xn = L["norm"][:, src].astype(np.float32)[ev]
                xi = L["inp"][:, src].astype(np.float32)[ev]
                xi = xi / np.sqrt((xi * xi).mean(axis=1, keepdims=True) + eps) * normw[T]
                lg, li = xn @ W.T, xi @ W.T
                for B in BUDGETS:
                    add(("gate-ahead", dlt, B), lg, resident, miss, B)
                    add(("renorm-ahead", dlt, B), li, resident, miss, B)
                add_ranked("renorm-ahead", dlt, li, resident, miss)
                order = np.argsort(-li, axis=1)
                for K in SCHED_K:
                    m = np.zeros_like(miss); np.put_along_axis(m, order[:, :K], True, axis=1)
                    cand_store[(K, T, dlt)] = m & ~resident
            if T <= 3:
                for B in BUDGETS:
                    add(("embedding", T + 1, B), attn0 @ W.T, resident, miss, B)
            # temporal: last token's set at T (rank by nothing: all candidates tie)
            prev_sets = np.zeros((steps, 128), dtype=np.float32)
            np.put_along_axis(prev_sets[1:], ids_T[:-1], 1.0, axis=1)
            for B in BUDGETS:
                add(("temporal", 1, B), prev_sets[ev], resident, miss, B)
        n_ev = steps - warm
        for K in SCHED_K:
            for hs in SCHED_HORIZONS:
                a = sim.setdefault((K, hs), [0.0, 0.0, 0, 0, 0, 0])
                for s_i in range(n_ev):
                    cands = {(T, h): cand_store[(K, T, h)][s_i] for T in range(n_layer) for h in hs
                             if (K, T, h) in cand_store}
                    b, w, nl, cv, nm = schedule_sim(cands, {T: miss_store[T][s_i] for T in range(n_layer)}, n_layer)
                    a[0] += b; a[1] += w; a[2] += nl; a[3] += cv; a[4] += nm; a[5] += 1

    rows = []
    for (name, dlt, B), (hit, sent, m) in sorted(acc.items()):
        rows.append({"predictor": name, "horizon_layers": dlt, "budget": B,
                     "miss_recall": hit / max(m, 1), "precision": hit / max(sent, 1),
                     "sent_per_layer_step": sent / max(total_steps * n_layer, 1)})
    out = {"steps_evaluated": total_steps, "prompts": len(dumps), "warmup": warm,
           "misses_per_token": total_miss / max(total_steps, 1),
           "misses_per_token_by_layer": (per_layer_miss / max(total_steps, 1)).round(3).tolist(),
           "router_sanity_recall": sanity[0] / max(sanity[1], 1), "rows": rows}
    out["schedule_sim"] = [{"top_k": K, "horizons": list(hs), "stall_ms_without": a[0] / a[5] / 1000,
                            "stall_ms_with": a[1] / a[5] / 1000, "copies_per_token": a[2] / a[5],
                            "misses_converted_pct": 100 * a[3] / max(a[4], 1),
                            "useful_copy_pct": 100 * a[3] / max(a[2], 1)} for (K, hs), a in sorted(sim.items())]
    (ROOT / "artifacts" / "p10_lookahead_probe.json").write_text(json.dumps(out, indent=1))
    print("schedule simulation (one copy engine, EDF, late copies dropped):")
    for r in out["schedule_sim"]:
        print(f"  top{r['top_k']:<2d} horizons {str(r['horizons']):10s} stall {r['stall_ms_without']:.2f} -> "
              f"{r['stall_ms_with']:.2f} ms/token  copies {r['copies_per_token']:5.1f}/token  "
              f"misses converted {r['misses_converted_pct']:5.1f}%  useful copies {r['useful_copy_pct']:5.1f}%")
    print(f"{total_steps} decode steps from {len(dumps)} prompts; misses/token {out['misses_per_token']:.1f}; "
          f"router sanity {out['router_sanity_recall']:.4f}")
    print("selection rules (recall of misses / precision of what is sent / experts sent per token):")
    for r in rows:
        if r["budget"] == 0:
            print(f"  {r['predictor']:24s} d={r['horizon_layers']}  {r['miss_recall']*100:5.1f} / {r['precision']*100:5.1f} / "
                  f"{r['sent_per_layer_step']*n_layer:5.1f}")
    rows_b = [r for r in rows if r["budget"] > 0]
    print(f"{'predictor':14s} {'d':>2s} " + " ".join(f"  B={b}: rec/prec" for b in BUDGETS))
    keys = sorted({(r["predictor"], r["horizon_layers"]) for r in rows_b})
    for name, dlt in keys:
        cells = []
        for b in BUDGETS:
            r = next(x for x in rows_b if x["predictor"] == name and x["horizon_layers"] == dlt and x["budget"] == b)
            cells.append(f"  {r['miss_recall']*100:5.1f}/{r['precision']*100:5.1f}")
        print(f"{name:14s} {dlt:2d} " + " ".join(cells))


if __name__ == "__main__":
    main()
