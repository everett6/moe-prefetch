"""
P6: does recency still carry information the model does not have?

P5 concluded the limit is in the features. There is an obvious candidate: the
model's four input blocks say WHICH experts were used at four specific places in
the trace, and nothing about how long ago any expert was last touched. That is
precisely the signal LRU runs on, and the model has never seen it.

Rather than retrain with new features, this asks the cheap question first: blend
the model's score with the cache's own recency, sweep the weight, and see
whether anything beats either alone.

    victim = argmin over resident of   model_score(e) - lam * normalised_recency(e)

  lam = 0      the model alone, P4's policy
  lam -> inf   LRU, since recency then dominates
  in between   whether the two disagree usefully

If the best blend is at lam = 0, the model has already subsumed recency and new
features are the way forward. If an interior lam wins, recency is orthogonal
information and belongs in the model's inputs -- which costs 2 MAC per expert as
a per-layer scalar weight, not a new architecture.

Scored on LOCKED TEST groups against Belady on the same rows.
"""
import argparse
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
from p4_train_evict import EvictIndex, belady_throughput  # noqa: E402
from p5_ensemble import load  # noqa: E402

ART = os.path.join(ROOT, "artifacts")


def run(scores, ix, rows, decode_ok, cost, lam, capacity=66, max_inserts=2):
    d = ix.d
    row_at = {(int(d["prompt"][r]), int(d["pos"][r]), int(d["layer"][r])): r
              for r in rows}
    cur_key = {"k": None}
    tick = {"t": 0}

    def victim_fn(layer, resident, recency):
        r = row_at.get((cur_key["k"][0], cur_key["k"][1], layer))
        if r is None or scores is None:
            return min(recency, key=recency.get)
        p = scores[r]
        # Rank-normalised recency. A first version divided the cache's absolute
        # tick counter by the token count; that counter advances once per
        # OBSERVE, not once per token, so the ratio was far above 1 and every
        # non-zero lambda was testing near-pure LRU rather than a blend. Ranking
        # within the resident set puts both terms on [0,1] whatever the scale.
        if lam == 0.0:
            return min(resident, key=lambda e: p[e])
        order = sorted(resident, key=lambda e: recency.get(e, 0))
        n = max(len(order) - 1, 1)
        rank = {e: i / n for i, e in enumerate(order)}
        return min(resident, key=lambda e: p[e] - lam * rank[e])

    env = PrefetchEnv(capacity=capacity, max_inserts=max_inserts, cost=cost,
                      victim_fn=victim_fn, n_layers=ix.n_layers)
    dstats = {"hit": 0, "lookup": 0, "uploads": 0}
    dtok, prev, last = 0, dict(env.stats), None
    for r in rows:
        L = int(d["layer"][r])
        key = (int(d["prompt"][r]), int(d["pos"][r]))
        if key != last:
            if last is not None:
                env.step_boundary()
                tick["t"] += 1
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
    ap.add_argument("--ckpt",
                    default=os.path.join(ROOT, "artifacts", "ckpt",
                                         "EVICT_R3_h8-lowrank-lr0.01.pt"))
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

    m, _ = load(args.ckpt, ix.n_layers, device)
    sc = np.zeros((ix.n, M.N_EXPERT), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(rows), 16384):
            idx = rows[s:s + 16384]
            b, _ = ix.batch(idx, device)
            sc[idx] = torch.sigmoid(m(b)[0]).cpu().numpy()

    lru = run(None, ix, rows, decode_ok, cost, 0.0)
    bel = belady_throughput(args.replay_index, args.replay_corpus, cost,
                            split=args.split)
    span = bel["tok_s"] - lru["tok_s"]

    print("\n=== P6: MODEL SCORE BLENDED WITH RECENCY ===\n")
    print(f"{'lambda':>9}{'decode tok/s':>14}{'vs LRU':>9}{'uploads/tok':>13}"
          f"{'of Belady':>11}")
    print("-" * 56)
    print(f"{'LRU':>9}{lru['tok_s']:>14.1f}{0.0:>+9.1f}"
          f"{lru['uploads_per_token']:>13.1f}{0:>10.0f}%")
    out = {"lru": lru, "belady": bel, "points": []}
    for lam in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        r = run(sc, ix, rows, decode_ok, cost, lam)
        r["lam"] = lam
        r["gain"] = r["tok_s"] - lru["tok_s"]
        out["points"].append(r)
        print(f"{lam:>9.2f}{r['tok_s']:>14.1f}{r['gain']:>+9.1f}"
              f"{r['uploads_per_token']:>13.1f}{100*r['gain']/span:>10.0f}%")
    print(f"{'Belady':>9}{bel['tok_s']:>14.1f}{span:>+9.1f}"
          f"{bel['uploads_per_token']:>13.1f}{100:>10.0f}%")

    best = max(out["points"], key=lambda r: r["tok_s"])
    at0 = out["points"][0]
    print(f"\nbest lambda {best['lam']:.2f}: {best['gain']:+.1f} tok/s "
          f"({100*best['gain']/span:.0f}% of Belady), "
          f"{best['tok_s']-at0['tok_s']:+.1f} against the model alone")
    if best["lam"] > 0 and best["tok_s"] > at0["tok_s"] + 0.3:
        print("Recency is orthogonal information. It belongs in the model's")
        print("inputs, at 2 MAC per expert as a per-layer scalar weight.")
    else:
        print("The model has already subsumed recency; blending adds nothing.")
    json.dump(out, open(os.path.join(ART, "p6_blend.json"), "w"), indent=1,
              default=float)
    print("wrote artifacts/p6_blend.json")


if __name__ == "__main__":
    main()
