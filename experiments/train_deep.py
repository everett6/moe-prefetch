"""
Stage A: supervised training of a fresh expert predictor on the v3 corpus.

Nothing here is initialised from the draft. Fresh corpus, fresh weights, fresh
optimizer state; the draft survives only as an archived file used later as one
arm of the comparison.

Three things this reports that Recall@8 alone does not, because Recall@8 alone is
a poor guide to whether the system gets faster:

  recall on HARD rows    rows whose experts would miss a 66-slot LRU cache. On an
                         easy row the cache already has everything and a perfect
                         prediction is worth nothing, so a model can lift overall
                         recall while changing no prefetch decision that mattered
  per-register recall    on the locked test registers, so a single workload
                         cannot carry the average
  measured latency       microseconds per call, single-threaded, on the CPU that
                         has to run it between two expert FFNs

Selection -- architecture, loss, width, learning rate, early stopping -- uses the
VALIDATION registers only. The test registers are read when `--test` is passed,
which is meant to happen once.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import models_deep as M  # noqa: E402

INDEX = os.environ.get("INDEX", os.path.join(ROOT, "data", "index-v3.npz"))
ARTIFACTS = os.path.join(ROOT, "artifacts")
CKPT = os.path.join(ARTIFACTS, "ckpt")
K = 8
FIELDS = ("prev", "cur", "below", "self_prev")


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


class Index:
    def __init__(self, path):
        z = np.load(path)
        self.d = {k: z[k] for k in z.files}
        self.registers = [str(r) for r in self.d["registers"]]
        self.n = len(self.d["layer"])
        self.n_layers = int(self.d["layer"].max()) + 2
        self.split = self.d["split"]

    def rows(self, split_id):
        return np.where(self.split == split_id)[0]

    def batch(self, idx, device):
        d = self.d
        b = {f: torch.from_numpy(d[f][idx].astype(np.int64)).to(device) for f in FIELDS}
        b["layer"] = torch.from_numpy(d["layer"][idx].astype(np.int64)).to(device)
        tgt = torch.from_numpy(d["target"][idx].astype(np.int64)).to(device)
        y = torch.zeros(len(idx), M.N_EXPERT, device=device)
        y.scatter_(1, tgt.clamp(min=0), 1.0)
        return b, y, tgt


@torch.no_grad()
def evaluate(model, ix, rows, device, batch=32768, ks=(8,), breakdown=False):
    model.eval()
    hit = {k: 0.0 for k in ks}
    n = 0
    hard_hit, hard_n = 0.0, 0
    per_reg, per_layer = {}, {}
    miss = ix.d["miss66"]
    regs = ix.d["reg"]
    lays = ix.d["layer"]
    for s in range(0, len(rows), batch):
        idx = rows[s:s + batch]
        b, y, tgt = ix.batch(idx, device)
        logits, _ = model(b)
        for k in ks:
            top = logits.topk(k, dim=1).indices
            m = torch.zeros_like(y).scatter_(1, top, 1.0)
            inter = (m * y).sum(1)
            hit[k] += float(inter.sum())
        top8 = logits.topk(K, dim=1).indices
        m8 = torch.zeros_like(y).scatter_(1, top8, 1.0)
        inter8 = (m8 * y).sum(1).cpu().numpy()
        n += len(idx)
        h = miss[idx] > 0
        hard_hit += float(inter8[h].sum())
        hard_n += int(h.sum())
        if breakdown:
            for r in np.unique(regs[idx]):
                sel = regs[idx] == r
                a, c = per_reg.get(int(r), (0.0, 0))
                per_reg[int(r)] = (a + float(inter8[sel].sum()), c + int(sel.sum()))
            for L in np.unique(lays[idx]):
                sel = lays[idx] == L
                a, c = per_layer.get(int(L), (0.0, 0))
                per_layer[int(L)] = (a + float(inter8[sel].sum()), c + int(sel.sum()))
    out = {f"recall@{k}": hit[k] / (K * max(n, 1)) for k in ks}
    out["recall@8_hard"] = hard_hit / (K * max(hard_n, 1))
    out["hard_fraction"] = hard_n / max(n, 1)
    out["n"] = n
    if breakdown:
        out["per_register"] = {ix.registers[r]: v[0] / (K * max(v[1], 1))
                               for r, v in sorted(per_reg.items())}
        out["per_layer"] = {L: v[0] / (K * max(v[1], 1)) for L, v in sorted(per_layer.items())}
    return out


def measure_latency(model, ix, device="cpu", n=2000):
    """Single-row, single-thread latency: the shape the runtime actually calls it in.

    Batched throughput would flatter it by an order of magnitude and would be the
    wrong number -- the engine predicts one layer at a time, between two FFNs.
    """
    model = model.to("cpu").eval()
    torch.set_num_threads(1)
    rows = ix.rows(0)[:n]
    b, _, _ = ix.batch(rows[:1], "cpu")
    with torch.no_grad():
        for _ in range(50):
            model(b)
        t0 = time.perf_counter()
        for i in range(n):
            bb, _, _ = ix.batch(rows[i:i + 1], "cpu")
            model(bb)
        el = time.perf_counter() - t0
    model.to(device)
    return 1e6 * el / n


def train_one(cfg, ix, device, log=True):
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))
    model = M.build(cfg["arch"], ix.n_layers, r=cfg.get("r", 64),
                    hidden=cfg.get("hidden", 128), context=cfg.get("context", True),
                    dropout=cfg.get("dropout", 0.0)).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg.get("wd", 0.01))
    tr, va = ix.rows(0), ix.rows(1)
    bs = cfg.get("batch", 8192)
    steps_per_epoch = max(len(tr) // bs, 1)
    total = steps_per_epoch * cfg["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=total,
                                                pct_start=0.15)
    best, best_state, bad, hist = -1.0, None, 0, []
    step = 0
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = np.random.permutation(tr)
        for s in range(0, steps_per_epoch * bs, bs):
            idx = perm[s:s + bs]
            if len(idx) == 0:
                continue
            b, y, _ = ix.batch(idx, device)
            logits, conf = model(b)
            ct = None
            if conf is not None:
                with torch.no_grad():
                    top = logits.topk(K, dim=1).indices
                    m = torch.zeros_like(y).scatter_(1, top, 1.0)
                    ct = (m * y).sum(1) / K
            loss = M.loss_fn(logits, y, cfg.get("loss", "bce"), conf, ct)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
        v = evaluate(model, ix, va, device)
        hist.append(v["recall@8"])
        if log:
            print(f"    epoch {ep + 1:2d}/{cfg['epochs']}  loss {float(loss):.4f}  "
                  f"val recall@8 {100 * v['recall@8']:.2f}%  hard {100 * v['recall@8_hard']:.2f}%",
                  file=sys.stderr, flush=True)
        if v["recall@8"] > best + 1e-5:
            best, bad = v["recall@8"], 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.get("patience", 3):
                if log:
                    print(f"    early stop: no val gain for {bad} epochs", file=sys.stderr)
                break
    model.load_state_dict(best_state)
    return model, {"val_recall": best, "params": n_par, "macs": model.macs(),
                   "epochs_run": len(hist), "history": hist,
                   "train_s": time.time() - t0}


def log_experiment(row):
    os.makedirs(ARTIFACTS, exist_ok=True)
    path = os.path.join(ARTIFACTS, "experiments.csv")
    cols = ["experiment_id", "utc", "git_commit", "label", "dataset_version",
            "dataset_pairs", "arch", "hyperparams", "params", "macs",
            "val_recall8", "val_recall8_hard", "test_recall8", "test_recall8_hard",
            "latency_us", "status", "note"]
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--configs", default="sweep")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--tag", default="FRESH_SUPERVISED")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ix = Index(args.index)
    meta = json.load(open(args.index + ".meta.json"))
    print(f"index: {ix.n:,} pairs, {ix.n_layers} layers, registers {ix.registers}",
          file=sys.stderr)
    print(f"  train {len(ix.rows(0)):,}  val {len(ix.rows(1)):,}  test {len(ix.rows(2)):,}",
          file=sys.stderr)

    E = args.epochs
    if args.configs == "sweep":
        configs = [
            {"name": "linear",        "arch": "linear",  "lr": 3e-2, "epochs": E, "loss": "bce"},
            {"name": "linear-mse",    "arch": "linear",  "lr": 3e-2, "epochs": E, "loss": "mse"},
            {"name": "lowrank-r32",   "arch": "lowrank", "lr": 3e-3, "epochs": E, "r": 32},
            {"name": "lowrank-r64",   "arch": "lowrank", "lr": 3e-3, "epochs": E, "r": 64},
            {"name": "lowrank-r96",   "arch": "lowrank", "lr": 3e-3, "epochs": E, "r": 96},
            {"name": "resmlp-h64",    "arch": "resmlp",  "lr": 3e-3, "epochs": E, "r": 64, "hidden": 64},
            {"name": "resmlp-h128",   "arch": "resmlp",  "lr": 3e-3, "epochs": E, "r": 64, "hidden": 128},
            {"name": "resmlp-noctx",  "arch": "resmlp",  "lr": 3e-3, "epochs": E, "r": 64, "hidden": 128,
             "context": False},
            {"name": "resmlp-listnet", "arch": "resmlp", "lr": 3e-3, "epochs": E, "r": 64,
             "hidden": 128, "loss": "listnet"},
        ]
    else:
        configs = json.loads(args.configs)

    os.makedirs(CKPT, exist_ok=True)
    results = {}
    for cfg in configs:
        print(f"\n[{cfg['name']}]  {cfg['arch']}  lr={cfg['lr']}  loss={cfg.get('loss','bce')}",
              file=sys.stderr)
        model, info = train_one(cfg, ix, device)
        lat = measure_latency(model, ix, device)
        v = evaluate(model, ix, ix.rows(1), device, ks=(4, 8, 12, 16))
        info.update(latency_us=lat, val=v)
        torch.save({"cfg": cfg, "state": model.state_dict(), "info": info,
                    "dataset_version": meta.get("dataset_version"), "tag": args.tag},
                   os.path.join(CKPT, f"{args.tag}-{cfg['name']}.pt"))
        row = {"experiment_id": f"{args.tag}-{cfg['name']}",
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "git_commit": git_commit(), "label": args.tag,
               "dataset_version": meta.get("dataset_version"), "dataset_pairs": ix.n,
               "arch": cfg["arch"], "hyperparams": json.dumps(
                   {k: v2 for k, v2 in cfg.items() if k not in ("name", "arch")}),
               "params": info["params"], "macs": info["macs"],
               "val_recall8": f"{v['recall@8']:.4f}",
               "val_recall8_hard": f"{v['recall@8_hard']:.4f}",
               "latency_us": f"{lat:.1f}", "status": "measured"}
        log_experiment(row)
        results[cfg["name"]] = info
        print(f"    -> val {100 * v['recall@8']:.2f}%  hard {100 * v['recall@8_hard']:.2f}%  "
              f"{info['params']:,} params  {info['macs']:,} MAC  {lat:.1f} us/call",
              file=sys.stderr)

    print("\n=== STAGE A: SUPERVISED (validation registers; test untouched) ===\n")
    print("%-16s %9s %9s %11s %9s %9s %9s" % ("config", "val@8", "val@8 hard",
                                              "params", "MAC", "us/call", "budget"))
    for n, i in sorted(results.items(), key=lambda kv: -kv[1]["val"]["recall@8"]):
        print("%-16s %8.2f%% %8.2f%% %11s %9s %8.1f %9s"
              % (n, 100 * i["val"]["recall@8"], 100 * i["val"]["recall@8_hard"],
                 f"{i['params']:,}", f"{i['macs']:,}", i["latency_us"],
                 "ok" if i["latency_us"] <= 10.0 else "OVER"))

    ok = {n: i for n, i in results.items() if i["latency_us"] <= 10.0}
    if not ok:
        print("\nNothing fits the 10 us budget -- the winner must be chosen on speed.")
        ok = results
    best = max(ok, key=lambda n: ok[n]["val"]["recall@8"])
    print(f"\nSelected on validation, within budget: {best}")
    json.dump({n: {"val": i["val"], "params": i["params"], "macs": i["macs"],
                   "latency_us": i["latency_us"], "epochs": i["epochs_run"],
                   "history": i["history"]} for n, i in results.items()} |
              {"_selected": best, "_tag": args.tag},
              open(os.path.join(ARTIFACTS, f"stage_a_{args.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
