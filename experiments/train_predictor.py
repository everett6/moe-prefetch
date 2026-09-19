"""
Train the expert predictor on the sharded corpus, without loading it.

Every model here is least-squares, and least-squares only needs the Gram
matrices: `G = X'X` (D x D) and `C = X'Y` (D x 128), per layer. Those are the
same size whether they were accumulated from 170 thousand rows or 4 million, so
one streaming pass fits exactly the model that loading everything would have --
not an approximation of it.

Accumulation is in float64. Over millions of rows a float32 Gram matrix loses
enough precision to change which ridge strength wins, which would be a silent
way to pick the wrong model.

**Leakage discipline.** Ridge strength, the repeat-prior weight and the choice of
feature set are all selected on the VALIDATION registers. The test registers are
read only when `--test` is passed, and that flag is meant to be used once, at the
end. Splits come from the corpus manifest and are never recomputed here -- if
this script could assign splits it could also, one refactor later, assign them
differently for training and evaluation.

Feature sets (blocks are laid out in this order inside x):

  prev      128  the previous token's experts at layer L+1  -- free on the host
  cur       128  this token's experts at layer L            -- free on the host
  pca       256  PCA of the L2-normalised hidden state      -- needs a device readback

Usage:
  python3 experiments/train_predictor.py --sets prev+cur,h_norm+prev+cur
  python3 experiments/train_predictor.py --sets prev+cur --test   # once, at the end
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from dataset import Corpus, build_pairs, prev_experts, multihot, N_EXPERT, K  # noqa: E402
from d2_export import write_binary, BLOCK_PCA, BLOCK_PREV, BLOCK_CUR  # noqa: E402

CORPUS = os.environ.get("CORPUS", os.path.join(ROOT, "data", "corpus"))
ARTIFACTS = os.path.join(ROOT, "artifacts")
LAMBDAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
PRIOR_WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
N_COMP = 256


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def recall_at_k(scores, truth_ids, k=K):
    """Mean fraction of the k true experts that appear in the top k predicted."""
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    hits = 0
    for t, p in zip(truth_ids, top):
        hits += len(set(t.tolist()) & set(p.tolist()))
    return hits / (k * max(len(truth_ids), 1))


def zs(a):
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)


BLOCKS = {"prev": BLOCK_PREV, "cur": BLOCK_CUR, "h_norm": BLOCK_PCA}


def set_blocks(name):
    return [BLOCKS[p] for p in name.split("+")]


def block_dim(b):
    return N_COMP if b == BLOCK_PCA else N_EXPERT


def fit_pca(corpus, verbose=True):
    """Mean and covariance of the L2-normalised hidden state over TRAIN rows only.

    Streamed: the covariance is 2048x2048 regardless of row count."""
    d = corpus.d_model
    n = 0
    s = np.zeros(d, dtype=np.float64)
    S = np.zeros((d, d), dtype=np.float64)
    for H, E, keys in corpus.iter_shards():
        m = corpus.split_mask(keys, "train")
        if not m.any():
            continue
        X = H[m].astype(np.float32)
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
        X = X.astype(np.float64)
        n += len(X)
        s += X.sum(0)
        S += X.T @ X
        if verbose:
            print(f"    pca pass: {n:,} train rows", file=sys.stderr, end="\r", flush=True)
    mean = (s / n).astype(np.float32)
    cov = (S - np.outer(s, s) / n) / max(n - 1, 1)
    w, V = np.linalg.eigh(cov)
    comp = V[:, ::-1][:, :N_COMP].astype(np.float32)
    evr = float(w[::-1][:N_COMP].sum() / w.sum())
    if verbose:
        print(f"\n    PCA on {n:,} train rows, {100 * evr:.1f}% of variance kept",
              file=sys.stderr)
    return mean, comp, evr


def features_for_shard(H, E, keys, blocks, pca=None):
    """Build (X, Y_ids, layer) for every (layer L -> L+1) pair in one shard."""
    src, dst, lay, index = build_pairs(keys)
    if len(src) == 0:
        return None
    cols = []
    for b in blocks:
        if b == BLOCK_PREV:
            cols.append(multihot(prev_experts(keys, E, index, src, lay)))
        elif b == BLOCK_CUR:
            cols.append(multihot(E[src]))
        else:
            mean, comp, scale = pca
            Hn = H[src].astype(np.float32)
            Hn /= (np.linalg.norm(Hn, axis=1, keepdims=True) + 1e-6)
            cols.append(((Hn - mean) @ comp) / scale)
    X = np.concatenate(cols, axis=1) if len(cols) > 1 else cols[0]
    return X, E[dst], lay, src, index, keys


def accumulate_grams(corpus, blocks, pca, split="train", verbose=True):
    """One pass: per-layer X'X and X'Y over the requested split."""
    D = sum(block_dim(b) for b in blocks)
    G, C, N = {}, {}, {}
    seen = 0
    for H, E, keys in corpus.iter_shards(split):
        m = corpus.split_mask(keys, split)
        if not m.any():
            continue
        f = features_for_shard(H, E, keys, blocks, pca)
        if f is None:
            continue
        X, Yid, lay, src, _, _ = f
        keep = m[src]
        X, Yid, lay = X[keep], Yid[keep], lay[keep]
        for L in np.unique(lay):
            sel = lay == L
            Xi = X[sel].astype(np.float64)
            Y = np.zeros((int(sel.sum()), N_EXPERT), dtype=np.float64)
            Y[np.arange(len(Y))[:, None], Yid[sel]] = 1.0
            L = int(L)
            if L not in G:
                G[L] = np.zeros((D, D), dtype=np.float64)
                C[L] = np.zeros((D, N_EXPERT), dtype=np.float64)
                N[L] = 0
            G[L] += Xi.T @ Xi
            C[L] += Xi.T @ Y
            N[L] += len(Xi)
        seen += int(keep.sum())
        if verbose:
            print(f"    gram: {seen:,} {split} pairs", file=sys.stderr, end="\r", flush=True)
    if verbose:
        print(f"\n    {seen:,} {split} pairs over {len(G)} layers, D={D}", file=sys.stderr)
    return G, C, N, D


def score_split(corpus, blocks, pca, W, prior, split, per_register=False):
    """Stream the split and score it. Returns overall recall plus per-layer and,
    optionally, per-register breakdowns."""
    num = den = 0.0
    per_layer, per_reg = {}, {}
    for H, E, keys in corpus.iter_shards(split):
        m = corpus.split_mask(keys, split)
        if not m.any():
            continue
        f = features_for_shard(H, E, keys, blocks, pca)
        if f is None:
            continue
        X, Yid, lay, src, _, _ = f
        keep = m[src]
        X, Yid, lay, src = X[keep], Yid[keep], lay[keep], src[keep]
        prev_ids = prev_experts(keys, E, build_pairs(keys)[3], src, lay) if prior else None
        for L in np.unique(lay):
            L = int(L)
            if L not in W:
                continue
            sel = lay == L
            s = X[sel] @ W[L]
            if prior and prior.get(L):
                s = zs(s)
                pid = prev_ids[sel]
                rows = np.repeat(np.arange(len(pid)), pid.shape[1])
                flat = pid.reshape(-1)
                ok = flat >= 0
                s[rows[ok], flat[ok]] += prior[L]
            r = recall_at_k(s, Yid[sel])
            n = int(sel.sum())
            num += r * n
            den += n
            a, b = per_layer.get(L, (0.0, 0))
            per_layer[L] = (a + r * n, b + n)
            if per_register:
                for reg in {corpus.prompt_register[int(p)] for p in keys[src[sel], 0]}:
                    a2, b2 = per_reg.get(reg, (0.0, 0))
                    per_reg[reg] = (a2 + r * n, b2 + n)
    out = {"recall": num / max(den, 1), "n": int(den),
           "per_layer": {L: v[0] / max(v[1], 1) for L, v in sorted(per_layer.items())}}
    if per_register:
        out["per_register"] = {r: v[0] / max(v[1], 1) for r, v in sorted(per_reg.items())}
    return out


def solve(G, C, D, lam):
    return {L: np.linalg.solve(G[L] + lam * np.eye(D), C[L]).astype(np.float32)
            for L in G}


def log_experiment(row):
    path = os.path.join(ARTIFACTS, "experiments.csv")
    cols = ["experiment_id", "utc", "git_commit", "dataset_rows", "dataset_gb",
            "features", "dims", "model", "hyperparams", "val_recall", "test_recall",
            "predictor_us_per_call", "status", "note"]
    new = not os.path.exists(path)
    os.makedirs(ARTIFACTS, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="prev+cur,cur,prev,h_norm+prev+cur")
    ap.add_argument("--test", action="store_true",
                    help="also evaluate on the locked test registers (use once)")
    ap.add_argument("--export", default="", help="feature set to write to models/")
    ap.add_argument("--corpus", default=CORPUS)
    args = ap.parse_args()

    corpus = Corpus(args.corpus)
    gb = corpus.total_bytes() / 1e9
    print(f"corpus: {len(corpus.shards)} shards, {corpus.total_rows():,} rows, {gb:.2f} GB",
          file=sys.stderr)
    for s in ("train", "val", "test"):
        print(f"  {s:<5} {corpus.registers(s)}", file=sys.stderr)

    sets = args.sets.split(",")
    need_pca = any(BLOCK_PCA in set_blocks(s) for s in sets)
    pca = None
    if need_pca:
        print("  fitting PCA over train rows...", file=sys.stderr)
        mean, comp, evr = fit_pca(corpus)
        # component scaling, also train-only
        n, s1, s2 = 0, np.zeros(N_COMP), np.zeros(N_COMP)
        for H, E, keys in corpus.iter_shards():
            m = corpus.split_mask(keys, "train")
            if not m.any():
                continue
            X = H[m].astype(np.float32)
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
            P = (X - mean) @ comp
            n += len(P); s1 += P.sum(0); s2 += (P.astype(np.float64) ** 2).sum(0)
        scale = (np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0)) + 1e-6).astype(np.float32)
        pca = (mean, comp, scale)

    results = {}
    for name in sets:
        t0 = time.time()
        blocks = set_blocks(name)
        print(f"\n[{name}]", file=sys.stderr)
        G, C, N, D = accumulate_grams(corpus, blocks, pca, "train")
        best = None
        for lam in LAMBDAS:
            W = solve(G, C, D, lam)
            v = score_split(corpus, blocks, pca, W, None, "val")
            print(f"    lam {lam:>8.0f}  val {100 * v['recall']:.2f}%", file=sys.stderr)
            if best is None or v["recall"] > best[0]:
                best = (v["recall"], lam, W)
        _, lam, W = best
        # prior weight, also chosen on validation
        bestp = (best[0], {L: 0.0 for L in W})
        for pw in PRIOR_WEIGHTS[1:]:
            prior = {L: pw for L in W}
            v = score_split(corpus, blocks, pca, W, prior, "val")
            if v["recall"] > bestp[0]:
                bestp = (v["recall"], prior)
        val_recall, prior = bestp
        el = time.time() - t0
        print(f"    chosen: lam={lam:.0f} prior={list(prior.values())[0]:.2f}  "
              f"val {100 * val_recall:.2f}%  ({el:.0f}s)", file=sys.stderr)

        row = {"experiment_id": f"ridge-{name}", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "git_commit": git_commit(), "dataset_rows": corpus.total_rows(),
               "dataset_gb": f"{gb:.2f}", "features": name, "dims": D, "model": "ridge",
               "hyperparams": f"lam={lam};prior={list(prior.values())[0]}",
               "val_recall": f"{val_recall:.4f}", "status": "measured"}
        results[name] = {"W": W, "prior": prior, "lam": lam, "val": val_recall, "D": D,
                         "blocks": blocks}
        if args.test:
            t = score_split(corpus, blocks, pca, W, prior, "test", per_register=True)
            results[name]["test"] = t
            row["test_recall"] = f"{t['recall']:.4f}"
            print(f"    TEST {100 * t['recall']:.2f}%  by register: " +
                  ", ".join(f"{r} {100 * v:.1f}" for r, v in t["per_register"].items()),
                  file=sys.stderr)
        log_experiment(row)

    print("\n=== predictor training ===\n")
    print("%-20s %6s %10s %12s" % ("features", "dims", "val", "test" if args.test else ""))
    for name, r in results.items():
        print("%-20s %6d %9.2f%% %11s" % (
            name, r["D"], 100 * r["val"],
            f"{100 * r['test']['recall']:.2f}%" if "test" in r else "-"))

    if args.export:
        r = results[args.export]
        f = {"W": r["W"], "prior": r["prior"], "blocks": r["blocks"], "dim": r["D"]}
        n_layers = max(r["W"]) + 2
        host_free = BLOCK_PCA not in r["blocks"]
        path = os.path.join(ROOT, "models",
                            "predictor-hostfree.bin" if host_free else "predictor-full.bin")
        mean, comp, scale = pca if not host_free else (None, None, None)
        write_binary(path, f, mean, comp, scale, n_layers, corpus.d_model)
        print(f"\nexported {args.export} -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

    os.makedirs(ARTIFACTS, exist_ok=True)
    json.dump({name: {"val": r["val"], "lam": r["lam"], "D": r["D"],
                      "prior": list(r["prior"].values())[0],
                      "test": r.get("test", {}).get("recall"),
                      "per_register": r.get("test", {}).get("per_register")}
               for name, r in results.items()},
              open(os.path.join(ARTIFACTS, "train_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
