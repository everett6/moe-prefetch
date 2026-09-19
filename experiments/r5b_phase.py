"""
Does training on prefill rows hurt the model where it is actually used?

The v4 index is 48% prefill and 52% decode, and the two are different problems:

  prefill   94% of expert lookups miss a 66-slot LRU, and the previous token's
            experts overlap the current one's by 0.014 of 8 -- essentially no
            temporal signal at all
  decode    5.9% miss, overlap 3.56 of 8 -- the repeat structure the predictor
            has always relied on

The prefetcher runs at decode. A model fitted to both is being asked to serve
two regimes with one set of weights, and the prefill half is both larger in
difficulty and useless in deployment. That could be diluting the weights, or it
could be harmless extra data -- an argument can be made either way, which is why
this measures rather than assumes.

Three arms, all evaluated on the SAME decode validation rows:

  mixed    the sweep's selected model, trained on everything
  decode   the same architecture trained on decode rows only
  prefill  the same architecture trained on prefill rows only, as a control --
           if this scores well on decode rows then the two regimes are not as
           separate as the statistics suggest

The phase of a row is decided by comparing its position against the prompt's
tokenised length, from the corpus manifest -- the same rule r4_coverage uses.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import models_deep as M  # noqa: E402
from train_deep import Index, evaluate, train_one, measure_latency, log_experiment, git_commit  # noqa: E402

ART = os.path.join(ROOT, "artifacts")


def phase_mask(ix, corpus):
    """True where the row is a DECODE position."""
    m = json.load(open(os.path.join(corpus, "manifest.json")))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "/home/everett/AI2/models/qwen3-30b-a3b-2507-tokenizer")
    n_tok = {p["id"]: len(tok(p["text"], add_special_tokens=True).input_ids)
             for p in m["prompts"]}
    bound = np.zeros(max(n_tok) + 2, dtype=np.int64)
    for pid, n in n_tok.items():
        bound[pid] = n
    return ix.d["pos"].astype(np.int64) >= bound[ix.d["prompt"].astype(np.int64)]


class Subset:
    """An Index restricted to a row mask, so train_one can be reused unchanged."""

    def __init__(self, ix, keep):
        self.ix = ix
        self.d = ix.d
        self.registers = ix.registers
        self.n_layers = ix.n_layers
        self.n = int(keep.sum())
        self._keep = keep
        self.split = ix.split

    def rows(self, split_id):
        r = np.where((self.split == split_id) & self._keep)[0]
        return r

    def batch(self, idx, device):
        return self.ix.batch(idx, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(ROOT, "data", "index-v4.npz"))
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus-v4"))
    ap.add_argument("--arch", default="linearctx")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ix = Index(args.index)
    dec = phase_mask(ix, args.corpus)
    print(f"index {ix.n:,} pairs: {dec.sum():,} decode, {(~dec).sum():,} prefill",
          file=sys.stderr)

    val_decode = np.where((ix.split == 1) & dec)[0]
    val_prefill = np.where((ix.split == 1) & ~dec)[0]
    print(f"validation: {len(val_decode):,} decode rows, {len(val_prefill):,} prefill",
          file=sys.stderr)

    arms = {
        "mixed": np.ones(ix.n, dtype=bool),
        "decode": dec,
        "prefill": ~dec,
    }
    results = {}
    for name, keep in arms.items():
        sub = Subset(ix, keep)
        cfg = {"name": f"phase-{name}", "arch": args.arch, "lr": args.lr,
               "epochs": args.epochs, "patience": args.patience, "batch": 16384}
        print(f"\n[{name}] training on {int(keep.sum()):,} rows "
              f"({len(sub.rows(0)):,} of them train)", file=sys.stderr)
        model, info = train_one(cfg, sub, device)
        vd = evaluate(model, ix, val_decode, device, ks=(8,))
        vp = evaluate(model, ix, val_prefill, device, ks=(8,))
        lat = measure_latency(model, ix, device)
        results[name] = {
            "train_rows": int(keep.sum()),
            "val_decode_recall8": vd["recall@8"],
            "val_decode_recall8_hard": vd["recall@8_hard"],
            "val_prefill_recall8": vp["recall@8"],
            "latency_us": lat, "params": info["params"], "macs": info["macs"],
        }
        log_experiment({
            "experiment_id": f"FRESH_REAL-phase-{name}",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git_commit(), "label": "FRESH_REAL",
            "dataset_version": "v4-20260919-real", "dataset_pairs": int(keep.sum()),
            "arch": args.arch,
            "hyperparams": json.dumps({"phase": name, "lr": args.lr}),
            "params": info["params"], "macs": info["macs"],
            "val_recall8": f"{vd['recall@8']:.4f}",
            "val_recall8_hard": f"{vd['recall@8_hard']:.4f}",
            "latency_us": f"{lat:.1f}", "status": "measured"})
        torch.save({"cfg": cfg, "state": model.state_dict(), "info": info,
                    "dataset_version": "v4-20260919-real",
                    "tag": f"FRESH_REAL-phase-{name}"},
                   os.path.join(ART, "ckpt", f"FRESH_REAL-phase-{name}.pt"))
        print(f"  -> decode val {100*vd['recall@8']:.2f}%  "
              f"(hard {100*vd['recall@8_hard']:.2f}%)  "
              f"prefill val {100*vp['recall@8']:.2f}%", file=sys.stderr)

    print("\n=== R5b: WHICH ROWS SHOULD THE MODEL BE FITTED TO? ===\n")
    print(f"{'trained on':10}{'rows':>14}{'decode val@8':>15}{'decode hard':>13}"
          f"{'prefill val@8':>15}")
    print("-" * 67)
    for name, r in results.items():
        print(f"{name:10}{r['train_rows']:>14,}{100*r['val_decode_recall8']:>14.2f}%"
              f"{100*r['val_decode_recall8_hard']:>12.2f}%"
              f"{100*r['val_prefill_recall8']:>14.2f}%")
    best = max(results, key=lambda k: results[k]["val_decode_recall8"])
    d = 100 * (results[best]["val_decode_recall8"] - results["mixed"]["val_decode_recall8"])
    print(f"\nbest on decode rows: {best} "
          f"({d:+.2f} points against training on everything)")
    json.dump(results, open(os.path.join(ART, "r5b_phase.json"), "w"), indent=1)
    print(f"wrote artifacts/r5b_phase.json")


if __name__ == "__main__":
    main()
