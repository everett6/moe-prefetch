"""
R7b: the deciding measurement -- a confidence-gated predictor in the real engine.

R7 found that thresholding on confidence reaches 92% precision where
unconditional prefetching manages 50%, and put an upper bound of +1.9 tok/s on
it. That bound ignores the thing that makes prefetching dangerous: a prefetch
installs into a full cache and EVICTS something, and the evicted expert may be
needed a token later. Scoring issued prefetches cannot see that cost.

This runs the gated policy through the same PrefetchEnv that E5's oracle uses,
which models installation, eviction, the slot pool and the per-step insert
throttle. So the number it produces is comparable to E5's +11.0 ceiling and to
the depth-0 baseline, and it is the number that decides whether any of this
ships.

Three arms, costed over the same decode tokens:

  base     no prefetching at all
  gated    prefetch the top-k non-resident experts whose probability clears a
           threshold, and nothing else
  oracle   prefetch exactly what the next token will use -- E5's ceiling

Sweeping the threshold from 0 (prefetch everything the model ranks) upward
traces the whole policy family, so the result is not one lucky operating point.
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
from train_deep import Index  # noqa: E402
from prefetch_env import PrefetchEnv, fit_cost_model  # noqa: E402
from r5b_phase import phase_mask, Subset  # noqa: E402

ART = os.path.join(ROOT, "artifacts")


def replay(ix, rows, cost, scores=None, thresh=None, depth=2, oracle=None,
           capacity=66, max_inserts=2, pool_extra=3, decode_ok=None):
    """One pass. `scores` gives the model's probabilities per row when the arm
    is the gated policy; `oracle` gives the truth when the arm is the ceiling;
    neither, and this is the do-nothing baseline."""
    d = ix.d
    env = PrefetchEnv(capacity=capacity, max_inserts=max_inserts,
                      pool_extra=pool_extra, cost=cost, n_layers=ix.n_layers)
    dstats = {"hit": 0, "lookup": 0, "uploads": 0}
    dtok, issued, correct = 0, 0, 0
    prev_stats = dict(env.stats)
    last = None
    for n, r in enumerate(rows):
        L = int(d["layer"][r]); nxt = L + 1
        key = (int(d["prompt"][r]), int(d["pos"][r]))
        if key != last:
            if last is not None:
                env.step_boundary()
                if decode_ok is None or decode_ok.get(last, True):
                    for k in dstats:
                        dstats[k] += env.stats[k] - prev_stats[k]
                    dtok += 1
                prev_stats = dict(env.stats)
            last = key
        may = decode_ok is None or decode_ok.get(key, True)
        if may:
            want = None
            if oracle is not None:
                w = oracle.get((key[0], key[1] + 1, nxt))
                if w is not None:
                    want = [int(e) for e in w if e >= 0]
            elif scores is not None:
                p = scores[n]
                idx = np.argpartition(-p, depth)[:depth]
                idx = idx[np.argsort(-p[idx])]
                want = [int(e) for e in idx
                        if p[e] >= thresh and e not in env.resident[nxt]]
                if want:
                    truth = set(int(e) for e in d["target"][r] if e >= 0)
                    issued += len(want)
                    correct += sum(1 for e in want if e in truth)
            if want:
                env.prefetch(nxt, want, len(want), min_keep=max_inserts)
        cur = [int(e) for e in d["cur"][r] if e >= 0]
        env.observe(L, cur)
        env.admit_demand(L, cur)
    dhit = dstats["hit"] / max(dstats["lookup"], 1)
    dmissed = ix.n_layers * 8 * (1 - dhit)
    dup = dstats["uploads"] / max(dtok, 1)
    dms = (cost["A_ms"] + cost["B_ms_per_miss"] * dmissed
           + cost["C_ms_per_upload"] * dup)
    return {"decode_tok_s": 1000.0 / max(dms, 1e-6), "decode_hit": dhit,
            "decode_tokens": dtok, "uploads_per_decode_token": dup,
            "issued": issued, "precision": correct / max(issued, 1)}


@torch.no_grad()
def model_scores(model, ix, rows, device, batch=16384):
    out = np.zeros((len(rows), M.N_EXPERT), dtype=np.float32)
    for s in range(0, len(rows), batch):
        idx = rows[s:s + batch]
        b, _, _ = ix.batch(idx, device)
        out[s:s + len(idx)] = torch.sigmoid(model(b)[0]).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(ROOT, "data", "index-v4-replay.npz"))
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus-v4-replay"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ix = Index(args.index)
    d = ix.d
    rows = ix.rows({"train": 0, "val": 1, "test": 2}[args.split])
    rows = rows[np.lexsort((d["layer"][rows], d["pos"][rows], d["prompt"][rows]))]

    dec = phase_mask(ix, args.corpus)
    decode_ok = {}
    for r in rows:
        decode_ok[(int(d["prompt"][r]), int(d["pos"][r]))] = bool(dec[r])
    print(f"{len(rows):,} rows; prefetch allowed at "
          f"{sum(decode_ok.values()):,} of {len(decode_ok):,} token positions")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = M.build(ck["cfg"]["arch"], ix.n_layers, r=ck["cfg"].get("r", 64),
                    hidden=ck["cfg"].get("hidden", 128),
                    context=ck["cfg"].get("context", True)).to(device).eval()
    model.load_state_dict(ck["state"])
    sc = model_scores(model, ix, rows, device)

    cost = fit_cost_model()
    oracle = {(int(d["prompt"][r]), int(d["pos"][r]), int(d["layer"][r])): d["cur"][r]
              for r in rows}

    base = replay(ix, rows, cost, decode_ok=decode_ok)
    orc = replay(ix, rows, cost, oracle=oracle, decode_ok=decode_ok)
    print(f"\nbaseline (no prefetch): {base['decode_tok_s']:.1f} tok/s decode, "
          f"hit {100*base['decode_hit']:.1f}%, "
          f"{base['uploads_per_decode_token']:.1f} uploads/token")
    print(f"oracle  (perfect)     : {orc['decode_tok_s']:.1f} tok/s "
          f"({orc['decode_tok_s']-base['decode_tok_s']:+.1f})\n")

    print(f"{'threshold':>10}{'issued':>10}{'precision':>11}{'decode tok/s':>14}"
          f"{'vs base':>10}{'uploads/tok':>13}")
    print("-" * 68)
    out = {"base": base, "oracle": orc, "depth": args.depth, "points": []}
    for t in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        g = replay(ix, rows, cost, scores=sc, thresh=t, depth=args.depth,
                   decode_ok=decode_ok)
        g["threshold"] = t
        g["gain"] = g["decode_tok_s"] - base["decode_tok_s"]
        out["points"].append(g)
        print(f"{t:>10.2f}{g['issued']:>10,}{100*g['precision']:>10.1f}%"
              f"{g['decode_tok_s']:>14.1f}{g['gain']:>+10.1f}"
              f"{g['uploads_per_decode_token']:>13.1f}")

    B, C = cost["B_ms_per_miss"], cost["C_ms_per_upload"]
    print(f"\n{'thresh':>7}{'iss/tok':>9}{'prec':>7}{'break-even says':>17}"
          f"{'measured':>11}{'extra uploads':>15}")
    print("-" * 66)
    for p in out["points"]:
        iss = p["issued"] / max(p["decode_tokens"], 1)
        ev = 1000 * iss * (p["precision"] * B - (1 - p["precision"]) * C)
        meas = 1000 * (1000 / base["decode_tok_s"] - 1000 / p["decode_tok_s"])
        dup = p["uploads_per_decode_token"] - base["uploads_per_decode_token"]
        p["formula_us"], p["measured_us"], p["extra_uploads"] = ev, meas, dup
        print(f"{p['threshold']:>7.2f}{iss:>9.2f}{100*p['precision']:>6.1f}%"
              f"{ev:>+14.0f} us{meas:>+8.0f} us{dup:>+15.2f}")
    print("\nThe break-even formula C/(B+C)=75% prices only the WRONG prefetches.")
    print("A right one still evicts a resident expert, and in a cache already")
    print("93% effective almost everything resident is about to be used again.")
    print("The extra uploads above are mostly those re-fetches, not mistakes.")

    best = max(out["points"], key=lambda p: p["decode_tok_s"])
    print(f"\nbest gated policy: {best['gain']:+.1f} tok/s at threshold "
          f"{best['threshold']:.2f} ({100*best['precision']:.1f}% precision)")
    print(f"the ceiling is {orc['decode_tok_s']-base['decode_tok_s']:+.1f}, so this "
          f"captures "
          f"{100*best['gain']/max(orc['decode_tok_s']-base['decode_tok_s'],1e-9):.0f}% "
          f"of what a perfect predictor would get.")
    if best["gain"] <= 0:
        print("\nNo threshold is profitable once eviction is modelled.")
    json.dump(out, open(os.path.join(ART, "r7b_gated_replay.json"), "w"),
              indent=1, default=float)
    print("wrote artifacts/r7b_gated_replay.json")


if __name__ == "__main__":
    main()
