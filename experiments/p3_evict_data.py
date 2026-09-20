"""
P3: training data for an eviction model.

P1 measured perfect eviction at +53.3 tok/s against LRU, five times the
admission ceiling. P2 measured every counter-based policy I could implement
cheaply at 3% of that, with LFU actively harmful. So the gap is real and it
needs a model.

What Belady knows and a counter does not is TIME TO NEXT USE. That is the thing
to learn. The target here is its decision-relevant form:

    will expert e be used at layer L within the next H tokens?

A victim is whatever scores lowest. Predicting the exact time would be a
regression with most of its loss spent on distinctions the policy never makes --
between an expert returning in 400 tokens and one returning in 900, when both
are equally evictable.

The features are deliberately the SAME four blocks the admission model uses
(prev / cur / below / self_prev experts), for two reasons. The engine already
computes them, so an eviction model costs no new feature plumbing. And it makes
the comparison honest: if eviction turns out to be more learnable than
admission, that is a fact about the targets, not about who got better inputs.

Built from index-v4 rather than from the shards, because the index already
carries (prompt, pos, layer) keys and the per-row expert sets, and the horizon
target is a scan over those.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

N_EXPERT = 128


def build(src, dst, horizon, corpus):
    z = np.load(src)
    d = {k: z[k] for k in z.files}
    prompt, pos, layer = d["prompt"], d["pos"], d["layer"].astype(np.int64)
    cur = d["cur"]
    n = len(layer)

    # Decode rows only. During prefill every layer uses all 128 experts within
    # 32 positions, so a horizon target there is all-ones and teaches nothing --
    # a first version included prefill and measured 95.7 of 128 experts "used",
    # which is not a signal.
    sys.path.insert(0, HERE)
    from train_deep import Index
    from r5b_phase import phase_mask
    ix = Index(src)
    dec = phase_mask(ix, corpus)

    mask = np.zeros((n, N_EXPERT), dtype=np.uint8)
    valid = np.zeros(n, dtype=bool)

    order = np.lexsort((pos, layer, prompt))
    groups, gstart, key_prev = [], 0, None
    for i in range(len(order)):
        k = (int(prompt[order[i]]), int(layer[order[i]]))
        if k != key_prev:
            if key_prev is not None:
                groups.append((gstart, i))
            gstart, key_prev = i, k
    groups.append((gstart, len(order)))

    n_pos, n_val = 0, 0
    for a, b in groups:
        idx = order[a:b]
        p = pos[idx]
        for j in range(len(idx)):
            r = idx[j]
            if not dec[r]:
                continue
            # The corpus stores positions in contiguous blocks of 8, so a
            # horizon that runs past the end of a block would look at positions
            # that were never captured and call the row safe when it is not.
            # Require the whole horizon to be present and adjacent.
            k, ok = j + 1, True
            need = set(range(int(p[j]) + 1, int(p[j]) + horizon + 1))
            seen = set()
            while k < len(idx) and p[k] <= p[j] + horizon:
                seen.add(int(p[k]))
                k += 1
            if need - seen:
                ok = False
            if not ok:
                continue
            acc = set()
            k = j + 1
            while k < len(idx) and p[k] <= p[j] + horizon:
                for e in cur[idx[k]]:
                    e = int(e)
                    if e >= 0:
                        acc.add(e)
                k += 1
            if acc:
                mask[r, sorted(acc)] = 1
                valid[r] = True
                n_pos += len(acc)
                n_val += 1

    print(f"{n:,} rows -> {n_val:,} usable (decode, full horizon present); "
          f"mean {n_pos/max(n_val,1):.1f} of {N_EXPERT} experts used within "
          f"{horizon} tokens", file=sys.stderr)

    out = dict(d)
    out["target_mask"] = np.packbits(mask, axis=1)
    out["evict_valid"] = valid
    with open(dst, "wb") as fh:
        np.savez(fh, **out)
    meta = json.load(open(src + ".meta.json"))
    meta["eviction_horizon"] = horizon
    meta["usable_rows"] = int(n_val)
    meta["target"] = (f"multi-hot: experts used at layer L within the next "
                      f"{horizon} tokens; evict_valid marks rows whose whole "
                      f"horizon was captured")
    json.dump(meta, open(dst + ".meta.json", "w"), indent=1)
    print(f"wrote {dst} ({os.path.getsize(dst)/1e6:.0f} MB)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "index-v4.npz"))
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--dst", default=None)
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus-v4"))
    args = ap.parse_args()
    dst = args.dst or os.path.join(ROOT, "data", f"index-v4-evict-h{args.horizon}.npz")
    build(args.src, dst, args.horizon, args.corpus)


if __name__ == "__main__":
    main()
