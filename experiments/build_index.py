"""
One expensive pass over the 28 GB corpus, producing a compact training index.

The models in models_deep.py take only expert ids -- `prev`, `cur`, and two
context sets -- because those are what the runtime has on the host when the
prediction has to be made. Expert ids are 16 bytes a row against 4,096 for the
hidden state, so the whole training set fits in ~600 MB and every subsequent
experiment runs from RAM instead of re-reading 28 GB.

That matters for the experiment loop more than for any single run: the brief asks
for many training runs across architectures, losses and hyperparameters, and a
pipeline that re-streams 28 GB per run would make that a day's work instead of an
hour's.

The hidden state is deliberately left behind. It is available in the corpus for
anyone who wants to test a variant that uses it, but it is not reachable at the
integration point -- llama.cpp's MoE observation callback sees routed ids on the
host and nothing else -- so a model that needs it cannot be deployed as-is.

Emitted per (token, layer) pair, all int16:
  layer                       which layer is predicting
  prev[8]     experts layer L+1 used on the PREVIOUS token
  cur[8]      experts layer L   used on THIS token
  below[8]    experts layer L-1 used on this token
  self_prev[8] experts layer L  used on the previous token
  target[8]   experts layer L+1 uses on this token          <- the label
  split, lru_miss_66, repeat_hit, register_id, prompt_id, position
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from dataset import Corpus  # noqa: E402

K = 8
SPLIT_ID = {"train": 0, "val": 1, "test": 2}


def build(corpus_dir, out_path):
    c = Corpus(corpus_dir)
    regs = sorted({p["register"] for p in c.manifest["prompts"]})
    reg_id = {r: i for i, r in enumerate(regs)}
    cols = {k: [] for k in ("layer", "prev", "cur", "below", "self_prev", "target",
                            "split", "miss66", "repeat", "reg", "prompt", "pos")}
    t0 = time.time()
    n_shards = len(c.shards)
    for si, sh in enumerate(c.shards):
        z = np.load(os.path.join(corpus_dir, sh["file"]))
        E = z["experts"].astype(np.int16)
        Kk = z["keys"].astype(np.int64)
        miss = z["lru_miss_66"] if "lru_miss_66" in z else np.zeros(len(Kk), np.uint8)
        rep = z["repeat_hit"] if "repeat_hit" in z else np.zeros(len(Kk), np.uint8)
        index = {(int(p), int(t), int(l)): i for i, (p, t, l) in enumerate(Kk)}
        for (p, t, L), i in index.items():
            j = index.get((p, t, L + 1))
            if j is None:
                continue
            def get(pp, tt, ll):
                q = index.get((pp, tt, ll))
                return E[q] if q is not None else np.full(K, -1, np.int16)
            cols["layer"].append(L)
            cols["prev"].append(get(p, t - 1, L + 1))
            cols["cur"].append(E[i])
            cols["below"].append(get(p, t, L - 1))
            cols["self_prev"].append(get(p, t - 1, L))
            cols["target"].append(E[j])
            cols["split"].append(SPLIT_ID[c.prompt_split[p]])
            cols["miss66"].append(int(miss[j]))     # difficulty of the row PREDICTED
            cols["repeat"].append(int(rep[j]))
            cols["reg"].append(reg_id[c.prompt_register[p]])
            cols["prompt"].append(p)
            cols["pos"].append(t)
        print(f"  shard {si + 1}/{n_shards}  {len(cols['layer']):,} pairs  "
              f"{time.time() - t0:.0f}s", file=sys.stderr, end="\r", flush=True)
    print(file=sys.stderr)

    out = {k: np.asarray(v, dtype=np.int16 if k not in ("prompt", "pos") else np.int32)
           for k, v in cols.items()}
    for k in ("prev", "cur", "below", "self_prev", "target"):
        out[k] = np.stack(cols[k]).astype(np.int16)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fh:
        np.savez(fh, registers=np.asarray(regs), **out)
    meta = {"dataset_version": c.manifest.get("dataset_version"),
            "corpus_rows": c.total_rows(), "corpus_bytes": c.total_bytes(),
            "pairs": int(len(out["layer"])), "registers": regs,
            "splits": c.splits, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "index_bytes": os.path.getsize(out_path)}
    json.dump(meta, open(out_path + ".meta.json", "w"), indent=1)
    print(f"{meta['pairs']:,} pairs -> {out_path} "
          f"({meta['index_bytes'] / 1e6:.0f} MB)", file=sys.stderr)
    for s, i in SPLIT_ID.items():
        n = int((out["split"] == i).sum())
        print(f"  {s:<5} {n:>12,} pairs ({100 * n / len(out['layer']):.1f}%)", file=sys.stderr)
    return meta


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "corpus-v3")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "data", "index-v3.npz")
    build(src, dst)
