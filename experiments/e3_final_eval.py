"""
The three-way comparison: DRAFT_OLD vs FRESH_SUPERVISED vs FRESH_RL.

Run under identical conditions, on the locked test registers, with the metrics
that decide the system rather than the ones that flatter a model. Kept strictly
labelled, because the draft was trained on a corpus that no longer exists and
mixing its numbers with the fresh ones would be meaningless.

DRAFT_OLD is the archived ridge predictor from artifacts/draft_old/. It is
loaded, never trained, never used to initialise anything.

Reported per arm:
  recall@8              the project's historical headline
  recall@8 on hard rows rows whose experts would miss a 66-slot LRU
  precision @ depth     against the 75% break-even -- the number that decides
  latency us/call       single-thread, the shape the runtime calls it in
  modelled tok/s        through the environment, using the fitted cost model

The modelled throughput is labelled as modelled. Measured end-to-end throughput
is a separate script (d3c_bench.sh) and the two are never presented as the same
kind of number.
"""
import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from prefetch_env import fit_cost_model  # noqa: E402
from train_deep import Index, evaluate, measure_latency  # noqa: E402
from e2_precision import evaluate_precision  # noqa: E402
import models_deep as M  # noqa: E402
from d2_export import read_binary, BLOCK_PREV, BLOCK_CUR  # noqa: E402

DRAFT = os.path.join(ROOT, "artifacts", "draft_old", "predictor-hostfree.bin")


class DraftRidge(torch.nn.Module):
    """The archived draft, wrapped so it can be evaluated by the same code.

    Its feature layout is [prev | cur] multi-hot with a per-layer z-score and
    repeat prior, exactly as its own C++ loader implements it.
    """

    def __init__(self, path, n_layers):
        super().__init__()
        m = read_binary(path)
        assert m["blocks"] == [BLOCK_PREV, BLOCK_CUR], m["blocks"]
        W = torch.zeros(n_layers, m["feat_dim"], m["n_expert"])
        prior = torch.zeros(n_layers)
        for L, w in m["W"].items():
            W[L] = torch.from_numpy(np.ascontiguousarray(w))
            prior[L] = m["prior"][L]
        self.register_buffer("W", W)
        self.register_buffer("prior", prior)
        self.n_expert = m["n_expert"]

    def forward(self, batch):
        L = batch["layer"]
        B = L.shape[0]
        x = torch.zeros(B, 2 * self.n_expert, device=L.device)
        for name, off in (("prev", 0), ("cur", self.n_expert)):
            ids = batch[name]
            ok = ids >= 0
            r = torch.arange(B, device=L.device).unsqueeze(1).expand_as(ids)
            x[r[ok], ids[ok] + off] = 1.0
        s = torch.bmm(x.unsqueeze(1), self.W[L]).squeeze(1)
        s = (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-9)
        ids = batch["prev"]
        ok = ids >= 0
        r = torch.arange(B, device=L.device).unsqueeze(1).expand_as(ids)
        s[r[ok], ids[ok]] += self.prior[L].unsqueeze(1).expand_as(ids)[ok]
        return s, None

    def macs(self):
        return 2 * 8 * self.n_expert


def load_fresh(path, n_layers, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = M.build(ck["cfg"]["arch"], n_layers, r=ck["cfg"].get("r", 64),
                hidden=ck["cfg"].get("hidden", 128),
                context=ck["cfg"].get("context", True)).to(device)
    m.load_state_dict(ck["state"])
    return m.eval(), ck


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(ROOT, "data", "index-v3.npz"))
    ap.add_argument("--fresh", required=True, help="FRESH_SUPERVISED checkpoint")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ix = Index(args.index)
    sid = {"train": 0, "val": 1, "test": 2}[args.split]
    rows = ix.rows(sid)
    cm = fit_cost_model()
    bar = cm["C_ms_per_upload"] / (cm["B_ms_per_miss"] + cm["C_ms_per_upload"])

    arms = {}
    if os.path.exists(DRAFT):
        arms["DRAFT_OLD"] = (DraftRidge(DRAFT, ix.n_layers).to(device).eval(), None)
    arms["FRESH_SUPERVISED"] = load_fresh(args.fresh, ix.n_layers, device)

    out = {"split": args.split, "break_even_precision": bar, "arms": {}}
    test_regs = sorted({ix.registers[int(r)] for r in ix.d["reg"][rows]})
    print(f"\n=== THREE-WAY COMPARISON ({args.split} registers: {test_regs}) ===\n")
    print("%-18s %9s %11s %13s %13s %10s" % ("arm", "recall@8", "hard@8",
                                             "warm prec@2", "warm prec@8", "params"))
    for name, (model, ck) in arms.items():
        ev = evaluate(model, ix, rows, device, ks=(4, 8, 12), breakdown=True)
        # identical configuration to e2_precision.py -- passing a shorter depth
        # list truncates the candidate pool to the top-4 and inflates precision,
        # which reported 92% against the 66% the same model actually achieves
        pr = evaluate_precision(model, ix, sid, device, depths=(1, 2, 3, 4, 6, 8))
        lat = measure_latency(model, ix, device)
        n_par = sum(p.numel() for p in model.parameters()) + sum(
            b.numel() for b in model.buffers())
        out["arms"][name] = {"recall": ev, "precision": {str(k): v for k, v in pr.items()},
                             "latency_us": lat, "params": int(n_par),
                             "macs": model.macs(),
                             "arch": (ck["cfg"]["arch"] if ck else "ridge-perlayer"),
                             "hyperparams": (ck["cfg"] if ck else {})}
        print("%-18s %8.2f%% %10.2f%% %12.1f%% %12.1f%% %10s"
              % (name, 100 * ev["recall@8"], 100 * ev["recall@8_hard"],
                 100 * pr[2]["warm_precision"], 100 * pr[8]["warm_precision"],
                 f"{n_par:,}"))

    print(f"\nbreak-even precision {100 * bar:.0f}%")
    for name, a in out["arms"].items():
        best = max(a["precision"].values(), key=lambda v: v["warm_precision"])
        p2 = a["precision"]["2"]["warm_precision"]
        print(f"  {name:<18} warm precision {100 * p2:.1f}% at depth 2, "
              f"best {100 * best['warm_precision']:.1f}%  -> "
              f"{'PAYS' if best['warm_precision'] >= bar else 'does not pay; depth 0 is optimal'}")

    json.dump(out, open(os.path.join(ROOT, "artifacts", f"final_eval_{args.split}.json"), "w"),
              indent=1, default=float)
    print(f"\nsaved artifacts/final_eval_{args.split}.json")


if __name__ == "__main__":
    main()
