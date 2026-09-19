"""
E2: the only predictor number that decides anything -- precision against 75%.

Recall@8 has been this project's headline since milestone 1, and it is the wrong
metric for the decision the engine actually makes. The engine does not ask "of
the 8 experts layer L+1 will use, how many did you name". It asks, for each
expert you want me to fetch, "will this one be used, and is it not already here".

That is precision, restricted to NON-RESIDENT experts, and it has a hard bar:

    break-even precision = C / (B + C) = 75%

from the cost model fitted to measured throughput. Above it, prefetching pays.
Below it, the right number of experts to prefetch is zero, whatever the recall.

Reported at each depth, because precision falls as depth grows and the optimal
depth is wherever it crosses the bar -- which may well be zero.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from prefetch_env import PrefetchEnv, fit_cost_model  # noqa: E402
from train_deep import Index  # noqa: E402
import models_deep as M  # noqa: E402


def evaluate_precision(model, ix, split_id, device, depths=(1, 2, 3, 4, 6, 8),
                       capacity=66, max_tokens=4000):
    """Replay tokens through a real LRU cache and score only the decisions the
    engine would actually have faced."""
    d = ix.d
    rows = ix.rows(split_id)
    order = np.lexsort((d["layer"][rows], d["pos"][rows], d["prompt"][rows]))
    rows = rows[order]
    env = PrefetchEnv(capacity=capacity, n_layers=ix.n_layers)
    # Split by cache warmth. Precision that rises with depth is a warning sign:
    # it means the deeper candidates are more often right, which happens when
    # rows with MANY non-resident candidates (a cold cache, where all 8 of the
    # model's picks survive the filter and ~74% of them are correct) dominate the
    # pooled average. Those rows are not the regime the engine runs in -- in
    # steady state only one or two experts per layer are non-resident, and those
    # are precisely the novel ones that are hardest to predict.
    stats = {p: {"issued": 0, "correct": 0} for p in depths}
    warm = {p: {"issued": 0, "correct": 0} for p in depths}
    tok_seen, last = 0, None
    with torch.no_grad():
        for s in range(0, len(rows), 16384):
            idx = rows[s:s + 16384]
            b, _, _ = ix.batch(idx, device)
            logits, _ = model(b)
            top = logits.topk(max(depths), dim=1).indices.cpu().numpy()
            for n, r in enumerate(idx):
                L = int(d["layer"][r]); nxt = L + 1
                key = (int(d["prompt"][r]), int(d["pos"][r]))
                if key != last:
                    if last is not None:
                        env.step_boundary()
                        tok_seen += 1
                        if tok_seen >= max_tokens:
                            return _finish(stats, warm)
                    last = key
                truth = set(int(e) for e in d["target"][r] if e >= 0)
                cand = [int(e) for e in top[n] if e not in env.resident[nxt]]
                is_warm = len(env.resident[nxt]) >= 0.9 * capacity
                for p in depths:
                    take = cand[:p]
                    n_ok = sum(1 for e in take if e in truth)
                    stats[p]["issued"] += len(take)
                    stats[p]["correct"] += n_ok
                    if is_warm:
                        warm[p]["issued"] += len(take)
                        warm[p]["correct"] += n_ok
                cur = [int(e) for e in d["cur"][r] if e >= 0]
                env.observe(L, cur)
                env.admit_demand(L, cur)
    return _finish(stats, warm)


def _finish(stats, warm):
    return {p: {"issued": v["issued"], "correct": v["correct"],
                "precision": v["correct"] / max(v["issued"], 1),
                "warm_issued": warm[p]["issued"], "warm_correct": warm[p]["correct"],
                "warm_precision": warm[p]["correct"] / max(warm[p]["issued"], 1)}
            for p, v in stats.items()}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(ROOT, "data", "index-v3.npz"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ix = Index(args.index)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = M.build(ck["cfg"]["arch"], ix.n_layers, r=ck["cfg"].get("r", 64),
                    hidden=ck["cfg"].get("hidden", 128),
                    context=ck["cfg"].get("context", True)).to(device).eval()
    model.load_state_dict(ck["state"])
    cm = fit_cost_model()
    bar = cm["C_ms_per_upload"] / (cm["B_ms_per_miss"] + cm["C_ms_per_upload"])
    sid = {"train": 0, "val": 1, "test": 2}[args.split]
    res = evaluate_precision(model, ix, sid, device)

    print(f"\n=== E2: PRECISION ON NON-RESIDENT EXPERTS ({args.split}) ===\n")
    print(f"break-even precision = C/(B+C) = {100 * bar:.0f}%\n")
    print("%7s %12s %11s %14s %11s %9s" % ("depth", "issued", "precision",
                                            "warm issued", "warm prec", "verdict"))
    for p in sorted(res):
        v = res[p]
        print("%7d %12s %10.1f%% %14s %10.1f%% %9s" % (
            p, f"{v['issued']:,}", 100 * v["precision"], f"{v['warm_issued']:,}",
            100 * v["warm_precision"],
            "PAYS" if v["warm_precision"] >= bar else "loses"))
    best = max(res, key=lambda p: res[p]["precision"])
    bw = max(res, key=lambda p: res[p]["warm_precision"])
    print(f"\nPooled best {100 * res[best]['precision']:.1f}% at depth {best}; "
          f"warm-cache best {100 * res[bw]['warm_precision']:.1f}% at depth {bw}, "
          f"against a {100 * bar:.0f}% bar.")
    print("The warm column is the one that matters: it is the steady state the "
          "engine\nspends almost all of its time in.")
    json.dump({"break_even": bar, "by_depth": {str(k): v for k, v in res.items()},
               "ckpt": os.path.basename(args.ckpt), "split": args.split},
              open(os.path.join(ROOT, "artifacts", f"precision_{args.split}.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
