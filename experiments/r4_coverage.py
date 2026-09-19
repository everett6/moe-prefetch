"""
What the v4 corpus actually contains, reported without the cold-start bias.

The capture stores the first 16 positions of every prompt in full, because
cold-start rows behave differently and are worth having. That makes them roughly
12% of stored rows against roughly 3% of a real trajectory, so any figure pooled
over stored rows over-weights them. Cold-start rows pooled with warm ones has
already produced one wrong number in this project (a precision that rose with
depth because of it), so the same mistake is not repeated on the corpus summary.

Reported separately, never pooled:

  cold    the first 16 positions of each prompt, where the per-layer LRU has
          barely been populated and almost everything misses by construction
  warm    everything after, which is the regime the engine actually runs in

Labels are read as captured. They were computed on the FULL trajectory before
subsampling, so a miss count here describes the real sequence rather than the
stored sample -- the bias being corrected is which rows are present, not what
they say.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COLD = 16


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "corpus-v4")
    m = json.load(open(os.path.join(corpus, "manifest.json")))
    agg = {"cold": {}, "warm": {}}
    for k in list(agg):
        agg[k] = {"n": 0, "miss66": 0.0, "miss32": 0.0, "repeat": 0.0}
    hist = np.zeros(m["n_expert"], dtype=np.int64)
    by_domain = {}

    # Phase split. The prefetcher runs during DECODE, so decode-phase routing is
    # the only regime in which a predictor could ever be paid. Prefill is the
    # human's text; decode is the model's own continuation, and the two are
    # different kinds of text with different routing. A figure pooled over both
    # describes neither -- and on the real corpus prefill is a large share of
    # rows, where on the synthetic one it was almost none, so pooling also makes
    # the two corpora incomparable.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "/home/everett/AI2/models/qwen3-30b-a3b-2507-tokenizer")
    n_prompt_tok = {}
    for pr in m["prompts"]:
        n = len(tok(pr["text"], add_special_tokens=True).input_ids)
        n_prompt_tok[pr["id"]] = n
    for k in ("prefill", "decode"):
        agg[k] = {"n": 0, "miss66": 0.0, "miss32": 0.0, "repeat": 0.0}

    reg_of = {p["id"]: p["register"] for p in m["prompts"]}
    split_of = {p["id"]: p["split"] for p in m["prompts"]}
    by_split = {}

    for sh in m["shards"]:
        path = os.path.join(corpus, sh["file"])
        if not os.path.exists(path):
            continue
        z = np.load(path)
        K = z["keys"]
        miss66 = z["lru_miss_66"].astype(np.float64)
        miss32 = z["lru_miss_32"].astype(np.float64)
        rep = z["repeat_hit"].astype(np.float64)
        hist += np.bincount(z["experts"].reshape(-1).astype(np.int64),
                            minlength=m["n_expert"])
        # "cold" is by position within the prompt, which is what the LRU sees
        cold = K[:, 1] < COLD
        # boundary per prompt: positions below the prompt's token count are
        # prefill, the rest are the generated continuation
        bound = np.array([n_prompt_tok.get(int(p), 0) for p in K[:, 0]])
        prefill = (K[:, 1] < bound) & ~cold
        decode = (K[:, 1] >= bound) & ~cold
        for name, sel in (("cold", cold), ("warm", ~cold),
                          ("prefill", prefill), ("decode", decode)):
            a = agg[name]
            a["n"] += int(sel.sum())
            a["miss66"] += float(miss66[sel].sum())
            a["miss32"] += float(miss32[sel].sum())
            a["repeat"] += float(rep[sel].sum())
        warm = ~cold
        for pid in np.unique(K[:, 0]):
            sel = (K[:, 0] == pid) & warm
            if not sel.any():
                continue
            dom = reg_of[int(pid)].split("/", 1)[0]
            d = by_domain.setdefault(dom, {"n": 0, "miss66": 0.0, "repeat": 0.0})
            d["n"] += int(sel.sum())
            d["miss66"] += float(miss66[sel].sum())
            d["repeat"] += float(rep[sel].sum())
            s = by_split.setdefault(split_of[int(pid)], {"n": 0, "miss66": 0.0})
            s["n"] += int(sel.sum())
            s["miss66"] += float(miss66[sel].sum())

    out = {"corpus": corpus, "dataset_version": m.get("dataset_version"),
           "cold_positions": COLD}
    print(f"{'':8}{'rows':>12}{'miss66/8':>11}{'miss rate':>11}{'miss32/8':>11}"
          f"{'repeat/8':>11}")
    print("-" * 64)
    for name in ("cold", "warm", "prefill", "decode"):
        a = agg[name]
        n = max(a["n"], 1)
        row = {"rows": a["n"], "miss66": a["miss66"] / n, "miss32": a["miss32"] / n,
               "repeat": a["repeat"] / n}
        row["miss_rate_pct"] = 100 * row["miss66"] / 8
        out[name] = row
        print(f"{name:8}{a['n']:>12,}{row['miss66']:>11.3f}"
              f"{row['miss_rate_pct']:>10.1f}%{row['miss32']:>11.3f}"
              f"{row['repeat']:>11.3f}")

    print(f"\nwarm rows by domain:")
    out["by_domain"] = {}
    for dom, d in sorted(by_domain.items()):
        n = max(d["n"], 1)
        out["by_domain"][dom] = {"rows": d["n"], "miss66": d["miss66"] / n,
                                 "repeat": d["repeat"] / n}
        print(f"  {dom:10}{d['n']:>12,}  miss66 {d['miss66']/n:.3f}  "
              f"repeat {d['repeat']/n:.3f}")

    print(f"\nwarm rows by split:")
    out["by_split"] = {}
    for sp, d in sorted(by_split.items()):
        n = max(d["n"], 1)
        out["by_split"][sp] = {"rows": d["n"], "miss66": d["miss66"] / n}
        print(f"  {sp:10}{d['n']:>12,}  miss66 {d['miss66']/n:.3f}")

    out["expert_hist"] = {"min": int(hist.min()), "max": int(hist.max()),
                          "never_seen": int((hist == 0).sum()),
                          "imbalance": float(hist.max() / max(hist.min(), 1))}
    print(f"\nexperts: rarest seen {hist.min():,}, commonest {hist.max():,} "
          f"({hist.max()/max(hist.min(),1):.1f}x), never seen {(hist==0).sum()}")

    # Named after the corpus: running this on v3 and v4 is the whole point, and
    # a single fixed filename means the second run silently destroys the first.
    tag = os.path.basename(os.path.normpath(corpus))
    dst = os.path.join(ROOT, "artifacts", f"r4_coverage_{tag}.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
