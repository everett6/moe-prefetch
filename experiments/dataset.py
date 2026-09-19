"""
Streaming access to the sharded corpus.

At 667 MB the old corpus could be loaded whole and indexed with a dict. At 15 GB
it cannot: the hidden states alone exceed half of this machine's RAM in fp16 and
all of it in fp32. Everything downstream therefore has to work a shard at a time.

That is less restrictive than it sounds, because the models this project fits are
all least-squares: ridge only needs the Gram matrices `X'X` and `X'Y`, which are
D x D and D x 128 regardless of how many rows went into them. So a single pass
over the corpus, accumulating Gram matrices per layer, fits the same model as
loading everything at once -- exactly, not approximately.

The feature construction is the part worth being careful about, because it is
sequential. `prev` is the previous token's experts AT THE LAYER BEING PREDICTED,
so building it needs rows ordered by (prompt, position, layer) and a running
memory of the last token's routing. Shards never split a prompt, which is what
makes that safe to do shard-locally.
"""
import json
import os

import numpy as np

N_EXPERT, K = 128, 8


class Corpus:
    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.splits = self.manifest["splits"]
        self.prompt_split = {p["id"]: p["split"] for p in self.manifest["prompts"]}
        self.prompt_register = {p["id"]: p["register"] for p in self.manifest["prompts"]}
        self.d_model = self.manifest["d_model"]
        self.n_expert = self.manifest["n_expert"]

    @property
    def shards(self):
        return self.manifest["shards"]

    def total_rows(self):
        return sum(s["rows"] for s in self.shards)

    def total_bytes(self):
        return sum(s["bytes"] for s in self.shards)

    def registers(self, split):
        return sorted(r for r, v in self.splits.items() if v == split)

    def iter_shards(self, split=None):
        """Yield (hidden, experts, keys) for shards containing `split` rows.

        A shard can hold prompts from more than one split, so the caller still
        has to mask by prompt id -- `split_mask` below does it.
        """
        for sh in self.shards:
            path = os.path.join(self.root, sh["file"])
            if not os.path.exists(path):
                raise FileNotFoundError(f"{sh['file']} listed in the manifest is missing")
            if split is not None and not any(
                    self.prompt_split.get(p) == split for p in sh["prompt_ids"]):
                continue
            z = np.load(path)
            yield z["hidden"], z["experts"].astype(np.int64), z["keys"].astype(np.int64)

    def split_mask(self, keys, split):
        want = {p for p, v in self.prompt_split.items() if v == split}
        return np.fromiter((int(p) in want for p in keys[:, 0]), bool, len(keys))


def build_pairs(keys):
    """Index rows so row i (layer L) predicts row j (layer L+1) of the same token.

    Returns (src, dst, layer). Built with a dict per shard; a shard is ~120K rows,
    so this stays cheap and never needs the whole corpus.
    """
    index = {(int(p), int(t), int(l)): i for i, (p, t, l) in enumerate(keys)}
    src, dst, lay = [], [], []
    for (p, t, l), i in index.items():
        j = index.get((p, t, l + 1))
        if j is not None:
            src.append(i)
            dst.append(j)
            lay.append(l)
    return (np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64),
            np.asarray(lay, dtype=np.int64), index)


def prev_experts(keys, experts, index, src, lay):
    """For each pair, the previous token's experts at layer L+1 -- multi-hot rows
    are built by the caller; this returns ids, or -1 padding where there is no
    previous token (start of a prompt)."""
    out = np.full((len(src), K), -1, dtype=np.int64)
    for m, (i, L) in enumerate(zip(src, lay)):
        p, t = int(keys[i][0]), int(keys[i][1])
        j = index.get((p, t - 1, int(L) + 1))
        if j is not None:
            out[m] = experts[j]
    return out


def multihot(ids, n=N_EXPERT):
    """ids: (rows, k) with -1 padding -> (rows, n) float32 multi-hot."""
    out = np.zeros((len(ids), n), dtype=np.float32)
    rows = np.repeat(np.arange(len(ids)), ids.shape[1])
    flat = ids.reshape(-1)
    ok = flat >= 0
    out[rows[ok], flat[ok]] = 1.0
    return out
