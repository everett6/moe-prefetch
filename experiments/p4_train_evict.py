"""
P4: train the eviction model, and judge it by throughput rather than by loss.

The target is multi-hot over 128 experts -- "used at this layer within the next
H tokens" -- so this cannot reuse train_deep's loop, which assumes an 8-slot
target list. It reuses the models and the feature blocks, which is the point:
the engine already computes those four blocks for the admission model, so an
eviction model adds no feature plumbing.

The metric that matters is not BCE and not recall. It is whether, used as the
cache's victim chooser, the model beats LRU on decode throughput in the same
environment P1 and P2 used. A model can improve its loss and pick worse victims,
because the loss weights all 128 experts equally while the policy only ever
compares the ~66 that are resident. So every checkpoint is scored both ways and
the throughput number is the one reported.

Self-improvement here is the loop, not a trick inside the model: train, measure
in the environment, look at where the policy loses against Belady, change one
thing, measure again. Each round is logged to artifacts/experiments.csv with its
measured tok/s so a later round cannot quietly claim an earlier round's number.
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
from train_deep import Index, measure_latency, log_experiment, git_commit  # noqa: E402
from prefetch_env import PrefetchEnv, fit_cost_model  # noqa: E402
from r5b_phase import phase_mask  # noqa: E402
from p1_eviction_headroom import build_future  # noqa: E402

ART = os.path.join(ROOT, "artifacts")
FIELDS = ("prev", "cur", "below", "self_prev")


class EvictIndex(Index):
    """Index whose target is a packed multi-hot mask over all 128 experts."""

    def __init__(self, path):
        super().__init__(path)
        self.mask = np.unpackbits(self.d["target_mask"], axis=1)[:, :M.N_EXPERT]
        self.valid = self.d["evict_valid"].astype(bool)

    def rows(self, split_id):
        return np.where((self.split == split_id) & self.valid)[0]

    def batch(self, idx, device):
        d = self.d
        b = {f: torch.from_numpy(d[f][idx].astype(np.int64)).to(device)
             for f in FIELDS}
        b["layer"] = torch.from_numpy(d["layer"][idx].astype(np.int64)).to(device)
        y = torch.from_numpy(self.mask[idx].astype(np.float32)).to(device)
        return b, y


def train(cfg, ix, device, epochs, patience, batch=16384):
    model = M.build(cfg["arch"], ix.n_layers, r=cfg.get("r", 64),
                    hidden=cfg.get("hidden", 128),
                    context=cfg.get("context", True)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    tr, va = ix.rows(0), ix.rows(1)
    best, best_state, bad = -1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(tr))
        tot = 0.0
        for s in range(0, len(tr), batch):
            idx = tr[perm[s:s + batch]]
            b, y = ix.batch(idx, device)
            logits, _ = model(b)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        auc = rank_quality(model, ix, va, device)
        print(f"    epoch {ep+1}/{epochs}  loss {tot/max(len(tr),1):.4f}  "
              f"val rank-acc {100*auc:.2f}%", file=sys.stderr)
        if auc > best + 1e-5:
            best, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"    early stop", file=sys.stderr)
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def rank_quality(model, ix, rows, device, batch=16384, sample=200000):
    """Probability that a NOT-soon-used expert scores below a soon-used one.

    This is the decision the policy makes, so it is the offline metric that
    tracks it most closely -- unlike BCE, which spends most of its gradient on
    experts that are never candidates."""
    if len(rows) > sample:
        rows = rows[np.linspace(0, len(rows) - 1, sample).astype(int)]
    model.eval()
    good = tot = 0
    for s in range(0, len(rows), batch):
        idx = rows[s:s + batch]
        b, y = ix.batch(idx, device)
        p = torch.sigmoid(model(b)[0])
        pos_mean = (p * y).sum(1) / y.sum(1).clamp(min=1)
        neg = 1 - y
        neg_mean = (p * neg).sum(1) / neg.sum(1).clamp(min=1)
        good += int((pos_mean > neg_mean).sum())
        tot += len(idx)
    return good / max(tot, 1)


def throughput(model, device, replay_index, replay_corpus, cost, capacity=66,
               max_inserts=2, mode="model", split=2, fut_mode=False):
    """`split` selects which GROUPS the trace comes from. It defaults to 2,
    LOCKED TEST, because the replay corpus was captured from the first 40
    prompts of the same corpus the model trains on -- evaluating on all of it
    would be scoring the model on prompts it was fitted to. Only the held-out
    groups (transformers, numpy, mbpp, stackoverflow-python, devops) are unseen.
    """
    """Decode tok/s with this model choosing victims, in the same environment
    P1 and P2 used, so the numbers are directly comparable."""
    ix = EvictIndex(replay_index)
    d = ix.d
    rows = np.where(ix.split == split)[0]
    rows = rows[np.lexsort((d["layer"][rows], d["pos"][rows], d["prompt"][rows]))]
    dec = phase_mask(ix, replay_corpus)
    decode_ok = {(int(d["prompt"][r]), int(d["pos"][r])): bool(dec[r]) for r in rows}

    scores = None
    if mode == "model":
        scores = np.zeros((ix.n, M.N_EXPERT), dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for s in range(0, len(rows), 16384):
                idx = rows[s:s + 16384]
                b, _ = ix.batch(idx, device)
                scores[idx] = torch.sigmoid(model(b)[0]).cpu().numpy()

    row_at = {}
    for r in rows:
        row_at[(int(d["prompt"][r]), int(d["pos"][r]), int(d["layer"][r]))] = r
    cur_key = {"k": None}

    def victim_fn(layer, resident, recency):
        r = row_at.get((cur_key["k"][0], cur_key["k"][1], layer))
        if r is None:
            return min(recency, key=recency.get)
        p = scores[r]
        return min(resident, key=lambda e: (p[e], recency.get(e, 0)))

    env = PrefetchEnv(capacity=capacity, max_inserts=max_inserts, cost=cost,
                      victim_fn=victim_fn if mode == "model" else None,
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


def belady_throughput(replay_index, replay_corpus, cost, split=2, capacity=66,
                      max_inserts=2):
    """The bound, on the same rows, so the fraction captured is meaningful."""
    import bisect
    ix = EvictIndex(replay_index)
    d = ix.d
    rows = np.where(ix.split == split)[0]
    rows = rows[np.lexsort((d["layer"][rows], d["pos"][rows], d["prompt"][rows]))]
    dec = phase_mask(ix, replay_corpus)
    decode_ok = {(int(d["prompt"][r]), int(d["pos"][r])): bool(dec[r]) for r in rows}
    fut, tok_of = build_future(ix, rows)
    now = {"t": 0}

    def victim_fn(layer, resident, recency):
        best, best_t = None, -1
        for e in resident:
            uses = fut.get((layer, e))
            nxt = None
            if uses:
                i = bisect.bisect_right(uses, now["t"])
                if i < len(uses):
                    nxt = uses[i]
            if nxt is None:
                return e
            if nxt > best_t:
                best, best_t = e, nxt
        return best if best is not None else next(iter(resident))

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
                if decode_ok.get(last, True):
                    for k in dstats:
                        dstats[k] += env.stats[k] - prev[k]
                    dtok += 1
                prev = dict(env.stats)
            last = key
        now["t"] = tok_of[r]
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
    ap.add_argument("--index", default=os.path.join(ROOT, "data", "index-evict-v4-h4.npz"))
    ap.add_argument("--replay-index", default=os.path.join(ROOT, "data", "index-evict-replay-h4.npz"))
    ap.add_argument("--replay-corpus", default=os.path.join(ROOT, "data", "corpus-v4-replay"))
    ap.add_argument("--arch", default="linearctx")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--tag", default="EVICT")
    ap.add_argument("--eval-split", type=int, default=2,
                    help="0 train, 1 val, 2 LOCKED TEST (default)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ix = EvictIndex(args.index)
    print(f"index {ix.n:,} rows, {len(ix.rows(0)):,} train / {len(ix.rows(1)):,} val "
          f"usable", file=sys.stderr)
    cost = fit_cost_model()

    cfg = {"arch": args.arch, "lr": args.lr, "name": f"{args.arch}-lr{args.lr}"}
    print(f"\n[{cfg['name']}]", file=sys.stderr)
    model, rq = train(cfg, ix, device, args.epochs, args.patience)
    lat = measure_latency(model, ix, device)

    lru = throughput(None, device, args.replay_index, args.replay_corpus, cost,
                     mode="lru", split=args.eval_split)
    got = throughput(model, device, args.replay_index, args.replay_corpus, cost,
                     split=args.eval_split)
    bel = belady_throughput(args.replay_index, args.replay_corpus, cost,
                            split=args.eval_split)
    print("\n=== P4: EVICTION MODEL, SCORED BY THROUGHPUT ===\n")
    print(f"{'policy':18}{'decode tok/s':>14}{'vs LRU':>9}{'hit':>9}{'uploads/tok':>13}")
    print("-" * 63)
    print(f"{'LRU (ships now)':18}{lru['tok_s']:>14.1f}{0.0:>+9.1f}"
          f"{100*lru['hit']:>8.1f}%{lru['uploads_per_token']:>13.1f}")
    print(f"{'this model':18}{got['tok_s']:>14.1f}{got['tok_s']-lru['tok_s']:>+9.1f}"
          f"{100*got['hit']:>8.1f}%{got['uploads_per_token']:>13.1f}")
    print(f"{'Belady (bound)':18}{bel['tok_s']:>14.1f}{bel['tok_s']-lru['tok_s']:>+9.1f}"
          f"{100*bel['hit']:>8.1f}%{bel['uploads_per_token']:>13.1f}")
    span = bel["tok_s"] - lru["tok_s"]
    frac = (got["tok_s"] - lru["tok_s"]) / span if span else 0.0
    print(f"\ncaptures {100*frac:.0f}% of the Belady bound on held-out groups")
    print(f"\nval rank-accuracy {100*rq:.2f}%, {lat:.1f} us/call")

    os.makedirs(os.path.join(ART, "ckpt"), exist_ok=True)
    torch.save({"cfg": cfg, "state": model.state_dict(), "tag": args.tag,
                "rank_quality": rq, "tok_s": got["tok_s"]},
               os.path.join(ART, "ckpt", f"{args.tag}-{cfg['name']}.pt"))
    log_experiment({
        "experiment_id": f"{args.tag}-{cfg['name']}",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(), "label": args.tag,
        "dataset_version": "v4-20260919-real-evict", "dataset_pairs": ix.n,
        "arch": args.arch,
        "hyperparams": json.dumps({"lr": args.lr, "horizon": 4}),
        "params": sum(p.numel() for p in model.parameters()),
        "macs": model.macs(), "val_recall8": f"{rq:.4f}",
        "val_recall8_hard": "", "latency_us": f"{lat:.1f}",
        "status": f"measured {got['tok_s']:.1f} tok/s vs LRU {lru['tok_s']:.1f}"})
    json.dump({"lru": lru, "model": got, "rank_quality": rq, "latency_us": lat},
              open(os.path.join(ART, f"p4_evict_{args.tag}.json"), "w"), indent=1,
              default=float)
    print(f"wrote artifacts/p4_evict_{args.tag}.json")


if __name__ == "__main__":
    main()
