"""
P5: approximating "how soon" instead of "within H".

The horizon sweep says longer horizons pick better victims -- 46% of the Belady
bound at H=2, 50% at H=4, 52% at H=8 -- but every one of them answers a yes/no
question, and Belady's advantage is ordinal: it knows WHICH of two experts
returns sooner. A single binary horizon cannot express that.

Summing P(used within H) across several horizons does, cheaply. An expert
predicted used within 2 tokens scores on all three heads; one returning in 30
scores on none. The sum is a crude expected-time-to-next-use, and it costs three
small matmuls rather than a new architecture.

If the ensemble beats the best single horizon, the direction to push is ordinal
targets. If it does not, the ceiling is in the features -- the four expert-set
blocks -- rather than in the target, and that is a different piece of work.

Scored on LOCKED TEST groups, against Belady on the same rows.
"""
import argparse
import bisect
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import models_deep as M  # noqa: E402
from prefetch_env import PrefetchEnv, fit_cost_model  # noqa: E402
from r5b_phase import phase_mask  # noqa: E402
from p1_eviction_headroom import build_future  # noqa: E402
from p4_train_evict import EvictIndex, belady_throughput  # noqa: E402

ART = os.path.join(ROOT, "artifacts")


def load(ckpt, n_layers, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    m = M.build(cfg["arch"], n_layers, r=cfg.get("r", 64),
                hidden=cfg.get("hidden", 128),
                context=cfg.get("context", True)).to(device).eval()
    m.load_state_dict(ck["state"])
    return m, cfg


def run(scores, ix, rows, decode_ok, cost, capacity=66, max_inserts=2,
        use_model=True):
    d = ix.d
    row_at = {(int(d["prompt"][r]), int(d["pos"][r]), int(d["layer"][r])): r
              for r in rows}
    cur_key = {"k": None}

    def victim_fn(layer, resident, recency):
        r = row_at.get((cur_key["k"][0], cur_key["k"][1], layer))
        if r is None:
            return min(recency, key=recency.get)
        p = scores[r]
        return min(resident, key=lambda e: (p[e], recency.get(e, 0)))

    env = PrefetchEnv(capacity=capacity, max_inserts=max_inserts, cost=cost,
                      victim_fn=victim_fn if use_model else None,
                      n_layers=ix.n_layers)
    dstats = {"hit": 0, "lookup": 0, "uploads": 0}
    dtok, prev, last = 0, dict(env.stats), None
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
        cur_key["k"] = key
        cur = [int(e) for e in d["cur"][r] if e >= 0]
        env.observe(L, cur)
        env.admit_demand(L, cur)
    hit = dstats["hit"] / max(dstats["lookup"], 1)
    missed = ix.n_layers * 8 * (1 - hit)
    up = dstats["uploads"] / max(dtok, 1)
    ms = (cost["A_ms"] + cost["B_ms_per_miss"] * missed
          + cost["C_ms_per_upload"] * up)
    return {"tok_s": 1000.0 / max(ms, 1e-6), "hit": hit, "uploads_per_token": up}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-index",
                    default=os.path.join(ROOT, "data", "index-evict-replay-h8.npz"))
    ap.add_argument("--replay-corpus",
                    default=os.path.join(ROOT, "data", "corpus-v4-replay"))
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--split", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ix = EvictIndex(args.replay_index)
    d = ix.d
    rows = np.where(ix.split == args.split)[0]
    rows = rows[np.lexsort((d["layer"][rows], d["pos"][rows], d["prompt"][rows]))]
    dec = phase_mask(ix, args.replay_corpus)
    decode_ok = {(int(d["prompt"][r]), int(d["pos"][r])): bool(dec[r]) for r in rows}
    cost = fit_cost_model()

    per = []
    total = np.zeros((ix.n, M.N_EXPERT), dtype=np.float32)
    macs = 0
    for c in args.ckpts:
        m, cfg = load(c, ix.n_layers, device)
        macs += m.macs()
        sc = np.zeros((ix.n, M.N_EXPERT), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(rows), 16384):
                idx = rows[s:s + 16384]
                b, _ = ix.batch(idx, device)
                sc[idx] = torch.sigmoid(m(b)[0]).cpu().numpy()
        per.append((os.path.basename(c), sc))
        total += sc

    lru = run(None, ix, rows, decode_ok, cost, use_model=False)
    bel = belady_throughput(args.replay_index, args.replay_corpus, cost,
                            split=args.split)
    print("\n=== P5: ORDINAL SCORE FROM SEVERAL HORIZONS ===\n")
    print(f"{'policy':28}{'decode tok/s':>14}{'vs LRU':>9}{'uploads/tok':>13}"
          f"{'of Belady':>11}")
    print("-" * 75)
    span = bel["tok_s"] - lru["tok_s"]
    print(f"{'LRU (ships now)':28}{lru['tok_s']:>14.1f}{0.0:>+9.1f}"
          f"{lru['uploads_per_token']:>13.1f}{0:>10.0f}%")
    out = {"lru": lru, "belady": bel, "arms": {}}
    for name, sc in per:
        r = run(sc, ix, rows, decode_ok, cost)
        r["gain"] = r["tok_s"] - lru["tok_s"]
        out["arms"][name] = r
        print(f"{name[:28]:28}{r['tok_s']:>14.1f}{r['gain']:>+9.1f}"
              f"{r['uploads_per_token']:>13.1f}{100*r['gain']/span:>10.0f}%")
    ens = run(total, ix, rows, decode_ok, cost)
    ens["gain"] = ens["tok_s"] - lru["tok_s"]
    ens["macs"] = macs
    out["ensemble"] = ens
    print(f"{'ENSEMBLE (sum)':28}{ens['tok_s']:>14.1f}{ens['gain']:>+9.1f}"
          f"{ens['uploads_per_token']:>13.1f}{100*ens['gain']/span:>10.0f}%")
    print(f"{'Belady (bound)':28}{bel['tok_s']:>14.1f}{span:>+9.1f}"
          f"{bel['uploads_per_token']:>13.1f}{100:>10.0f}%")

    best_single = max(out["arms"].values(), key=lambda r: r["tok_s"])
    print(f"\nensemble vs best single horizon: "
          f"{ens['tok_s']-best_single['tok_s']:+.1f} tok/s, at {macs:,} MAC total")
    if ens["tok_s"] > best_single["tok_s"] + 0.5:
        print("Ordinal information helps: push on richer time-to-next-use targets.")
    else:
        print("Ordinal information does not help here; the limit is in the")
        print("features, not the target.")
    json.dump(out, open(os.path.join(ART, "p5_ensemble.json"), "w"), indent=1,
              default=float)
    print("wrote artifacts/p5_ensemble.json")


if __name__ == "__main__":
    main()
