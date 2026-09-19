"""
R7: can the model be profitable on a SUBSET of its predictions?

E5 says a perfect decode-time predictor is worth +11.0 tok/s on real traffic.
E2 says this model's warm precision is 50.1% against a 75% break-even, so
prefetching everything it suggests loses 47 us per issue and would make
throughput worse. Those two facts together do not yet answer the question that
decides whether anything ships:

    is there a CONFIDENCE THRESHOLD above which precision clears 75%,
    on enough volume to be worth the wiring?

A predictor that is wrong half the time overall can still be right 90% of the
time on the predictions it is most sure about. If such a region exists and
carries meaningful volume, the profitable policy is "prefetch when confident,
otherwise let the demand path handle it" -- which is strictly better than both
"always prefetch" and "never prefetch".

The economics are per issued prefetch, from the fitted cost model:

    correct    saves B, the miss it avoids          (~47 us)
    incorrect  costs C, an upload nothing needed    (~142 us)
    expected   p*B - (1-p)*C, zero at p = C/(B+C) = 75%

**What this does not model.** A prefetch installs into a full cache and evicts
something, and the eviction may itself cause a later miss. This scores the
issued prefetches only, so it is an UPPER bound: real gains are smaller by
whatever the evictions cost. E5's replay does model installation, and is the
number to trust for magnitude.

**Why this differs from E2's 50.1%.** E2 takes the model's top-8, filters out
whatever is resident, and prefetches the first p of what remains -- so at depth 1
it always issues something, even on rows where the model's best non-resident
guess ranks eighth. Here the candidates are the top-`depth` by probability that
are ALSO non-resident, so rows where the model is not confident about anything
absent contribute nothing. The two are measuring different policies, and the gap
between them is precisely the value of being selective.

Scored on warm rows of the LOCKED TEST split only -- cold-start rows have
nothing resident and inflate precision, which has produced a wrong number in
this project before.
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


@torch.no_grad()
def collect(model, ix, rows, device, depth, capacity=66, max_tokens=4000):
    """Replay tokens through a real 66-slot LRU and record (score, correct) for
    every candidate the engine would actually have had to decide about.

    The first version of this filtered candidates against the PREVIOUS TOKEN's
    experts -- about eight of them -- instead of against the cache's real
    residency, and so counted as wins a mass of prefetches for experts a 66-slot
    cache already held. Those save nothing: the demand path would have found
    them resident. It reported 100% precision on the top 1% and +28.6 s of
    savings, against E2's 50.1% on the same checkpoint, which is what made it
    obvious something was wrong. This is the same mistake -- crediting a
    prefetch for work that was never going to happen -- that produced a wrong
    break-even number earlier in this project.
    """
    d = ix.d
    order = np.lexsort((d["layer"][rows], d["pos"][rows], d["prompt"][rows]))
    rows = rows[order]
    env = PrefetchEnv(capacity=capacity, n_layers=ix.n_layers)
    scores, correct = [], []
    tok_seen, last = 0, None
    for s0 in range(0, len(rows), 16384):
        idx = rows[s0:s0 + 16384]
        b, _, _ = ix.batch(idx, device)
        logits, _ = model(b)
        prob = torch.sigmoid(logits)
        top = prob.topk(depth, dim=1)
        tv = top.values.cpu().numpy()
        ti = top.indices.cpu().numpy()
        for n, r in enumerate(idx):
            L = int(d["layer"][r]); nxt = L + 1
            key = (int(d["prompt"][r]), int(d["pos"][r]))
            if key != last:
                if last is not None:
                    env.step_boundary()
                    tok_seen += 1
                    if tok_seen >= max_tokens:
                        return np.asarray(scores), np.asarray(correct)
                last = key
            res = env.resident[nxt]
            # warm only: a cache that is not yet full makes every pick look good
            if len(res) >= 0.9 * capacity:
                truth = set(int(e) for e in d["target"][r] if e >= 0)
                for e, v in zip(ti[n], tv[n]):
                    e = int(e)
                    if e in res:
                        continue
                    scores.append(float(v))
                    correct.append(1 if e in truth else 0)
            cur = [int(e) for e in d["cur"][r] if e >= 0]
            env.observe(L, cur)
            env.admit_demand(L, cur)
    return np.asarray(scores), np.asarray(correct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(ROOT, "data", "index-v4.npz"))
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus-v4"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--depth", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = Index(args.index)
    dec = phase_mask(full, args.corpus)
    ix = Subset(full, dec)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = M.build(ck["cfg"]["arch"], ix.n_layers, r=ck["cfg"].get("r", 64),
                    hidden=ck["cfg"].get("hidden", 128),
                    context=ck["cfg"].get("context", True)).to(device).eval()
    model.load_state_dict(ck["state"])

    sid = {"train": 0, "val": 1, "test": 2}[args.split]
    rows = ix.rows(sid)
    # warm only: drop the cold start, where nothing is resident
    print(f"{args.split}: {len(rows):,} decode rows "
          f"(warmth decided by the replayed cache, not by position)",
          file=sys.stderr)

    cost = fit_cost_model()
    B, C = cost["B_ms_per_miss"], cost["C_ms_per_upload"]
    bar = C / (B + C)
    print(f"break-even precision {100*bar:.0f}%  (B={1000*B:.0f} us saved, "
          f"C={1000*C:.0f} us spent)\n")

    sc, ok = collect(model, ix, rows, device, args.depth)
    order = np.argsort(-sc)
    sc, ok = sc[order], ok[order]
    cum_correct = np.cumsum(ok)
    n = np.arange(1, len(sc) + 1)
    prec = cum_correct / n
    # ms saved if we prefetch exactly the top-n most confident predictions
    saved = cum_correct * B - (n - cum_correct) * C

    out = {"break_even": bar, "B_ms": B, "C_ms": C, "depth": args.depth,
           "n_candidates": int(len(sc)), "split": args.split, "points": []}
    print(f"{'keep top':>12}{'threshold':>12}{'precision':>12}"
          f"{'ms saved':>12}{'verdict':>10}")
    print("-" * 58)
    for frac in (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0):
        k = max(int(frac * len(sc)), 1)
        p = float(prec[k - 1])
        row = {"fraction": frac, "n": k, "threshold": float(sc[k - 1]),
               "precision": p, "ms_saved": float(saved[k - 1])}
        out["points"].append(row)
        print(f"{frac*100:>11.1f}%{sc[k-1]:>12.3f}{100*p:>11.1f}%"
              f"{saved[k-1]:>12.1f}{'WINS' if p > bar else 'loses':>10}")

    best_i = int(np.argmax(saved))
    out["best"] = {"n": best_i + 1, "threshold": float(sc[best_i]),
                   "precision": float(prec[best_i]),
                   "ms_saved": float(saved[best_i]),
                   "fraction": (best_i + 1) / len(sc)}
    peak_p = float(prec[:max(len(sc)//1000, 1)].min())
    print(f"\nbest total saving: {saved[best_i]:+.1f} ms from the top "
          f"{best_i+1:,} predictions ({100*(best_i+1)/len(sc):.2f}% of candidates) "
          f"at {100*prec[best_i]:.1f}% precision")
    print(f"precision on the most confident 0.1%: {100*peak_p:.1f}%")
    tot_tokens = 4000
    print(f"\nUPPER BOUND: {saved[best_i]:.0f} ms saved over ~{tot_tokens:,} "
          f"replayed tokens. Eviction cost is not modelled here, so the real "
          f"figure is smaller; E5 is the number to trust for magnitude.")
    if saved[best_i] <= 0:
        print("\nNo confidence threshold is profitable: the model is below "
              "break-even even where it is most certain.")
    json.dump(out, open(os.path.join(ART, "r7_confidence.json"), "w"), indent=1)
    print(f"wrote artifacts/r7_confidence.json")


if __name__ == "__main__":
    main()
