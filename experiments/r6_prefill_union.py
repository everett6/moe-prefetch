"""
Is there anything to predict during prefill?

Half the rows in the real corpus are prefill, and prefill misses 94% of expert
lookups against 6% in decode. That looks like a large untapped opportunity, and
I asserted -- without measuring -- that it is not one, on the grounds that
prefill processes many tokens at once and the experts needed at a layer are the
UNION over the batch. If that union approaches all 128 experts, a prefetcher has
nothing to discriminate between and "prefetch" degenerates into "load
everything".

Plausible is not measured, so this measures it: slide a window of w consecutive
positions over each prompt's prefill region and count the distinct experts used
at each layer within the window.

  small union    prefill prediction is selective, and worth pricing
  union -> 128   there is nothing to predict; the misses are unavoidable

Run on the CONTIGUOUS replay corpus. The training corpus stores blocks of 8
positions, so a 512-wide window over it would span gaps and the union would be
drawn from positions the real batch never contained.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
N_EXPERT = 128
WINDOWS = (1, 8, 32, 128, 512)


def main():
    corpus = (sys.argv[1] if len(sys.argv) > 1
              else os.path.join(ROOT, "data", "corpus-v4-replay"))
    m = json.load(open(os.path.join(corpus, "manifest.json")))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "/home/everett/AI2/models/qwen3-30b-a3b-2507-tokenizer")
    n_prompt_tok = {p["id"]: len(tok(p["text"], add_special_tokens=True).input_ids)
                    for p in m["prompts"]}

    acc = {w: [] for w in WINDOWS}
    decode_acc = {w: [] for w in WINDOWS}
    for sh in m["shards"]:
        path = os.path.join(corpus, sh["file"])
        if not os.path.exists(path):
            continue
        z = np.load(path)
        K, E = z["keys"], z["experts"]
        order = np.lexsort((K[:, 2], K[:, 1], K[:, 0]))
        K, E = K[order], E[order]
        for pid in np.unique(K[:, 0]):
            bound = n_prompt_tok.get(int(pid), 0)
            psel = K[:, 0] == pid
            for L in np.unique(K[psel, 2]):
                sel = psel & (K[:, 2] == L)
                pos, ex = K[sel, 1], E[sel]
                # contiguity is required for a window to mean anything
                pre = pos < bound
                dec = pos >= bound
                for tag, mask, store in (("pre", pre, acc), ("dec", dec, decode_acc)):
                    p, e = pos[mask], ex[mask]
                    if len(p) < 2:
                        continue
                    for w in WINDOWS:
                        if w > len(p):
                            continue
                        # non-overlapping windows, which is how batches tile
                        for s in range(0, len(p) - w + 1, w):
                            chunk = e[s:s + w]
                            if chunk.size == 0:
                                continue
                            u = len(np.unique(chunk[chunk >= 0]))
                            store[w].append(u)

    out = {"corpus": corpus, "n_expert": N_EXPERT, "windows": {}}
    print(f"{'window':>8}{'prefill union':>16}{'% of 128':>11}"
          f"{'decode union':>15}{'% of 128':>11}")
    print("-" * 61)
    for w in WINDOWS:
        a, b = acc[w], decode_acc[w]
        if not a and not b:
            continue
        ra = float(np.mean(a)) if a else float("nan")
        rb = float(np.mean(b)) if b else float("nan")
        out["windows"][w] = {"prefill_union": ra, "decode_union": rb,
                             "prefill_n": len(a), "decode_n": len(b)}
        print(f"{w:>8}{ra:>16.1f}{100*ra/N_EXPERT:>10.1f}%"
              f"{rb:>15.1f}{100*rb/N_EXPERT:>10.1f}%")

    dst = os.path.join(ROOT, "artifacts", "r6_prefill_union.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\nA union near 128 means a prefetcher has nothing to choose between.")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
