"""
R2: how different is real traffic from the corpus I wrote?

Worth measuring rather than asserting. If the two corpora turned out to be
statistically similar, the retraining in R5 would be pointless and the
benchmark finding in R3 would be a non-event -- so this is the step that says
whether the rest of the plan is justified.

Compared on the axes that plausibly reach expert routing:

  length        routing is per position; a 40-token prompt and a 4,000-token
                prompt exercise entirely different amounts of context
  code content  a pasted stack trace routes differently from a sentence about
                stack traces. Measured by fenced blocks and by indented lines,
                not by the word "code" appearing
  vocabulary    token-level overlap between the two corpora. Low overlap means
                the old training set never saw the tokens the deployed model
                will meet
  shape         how much of a prompt is the human's own words versus quoted
                material -- the thing templates cannot produce at all

Tokenised with the model's own tokenizer, because "characters" is not the unit
the router sees and code has a very different characters-per-token ratio from
prose (roughly 3.0 vs 4.3 here), which would flatter the synthetic corpus.
"""
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

TOKENIZER = "/home/everett/AI2/models/qwen3-30b-a3b-2507-tokenizer"
PROMPTDIR = os.path.join(ROOT, "data", "prompts")
ART = os.path.join(ROOT, "artifacts")


def load_real():
    out = []
    for fn in sorted(os.listdir(PROMPTDIR)):
        if fn.endswith(".jsonl"):
            with open(os.path.join(PROMPTDIR, fn)) as f:
                out += [json.loads(l) for l in f]
    return out


def load_synthetic():
    from corpus_prompts import build_corpus
    return [{"source": f"synthetic:{src}", "group": reg, "domain": "synthetic",
             "text": text, "chars": len(text)}
            for reg, text, src in build_corpus(36)]


CODE_FENCE = re.compile(r"```|~~~")
INDENTED = re.compile(r"^(    |\t)\S", re.M)
TRACE = re.compile(r"Traceback \(most recent call last\)|^\s*at [\w.$]+\(|"
                   r"^\s*File \"[^\"]+\", line \d+", re.M)
IDENT = re.compile(r"\b[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*\b|\b\w+\(\)|::")


def shape(recs, tok):
    texts = [r["text"] for r in recs]
    lens = [len(x) for x in tok(texts, add_special_tokens=False).input_ids]
    q = statistics.quantiles(lens, n=10) if len(lens) > 10 else [min(lens)] * 9
    has_fence = sum(bool(CODE_FENCE.search(t)) for t in texts)
    has_indent = sum(bool(INDENTED.search(t)) for t in texts)
    has_trace = sum(bool(TRACE.search(t)) for t in texts)
    has_ident = sum(bool(IDENT.search(t)) for t in texts)
    # "any" rather than a sum: a prompt with a fence usually also has indented
    # lines, and adding the two indicators produced a 200% column.
    has_any = sum(bool(CODE_FENCE.search(t) or INDENTED.search(t)
                       or IDENT.search(t)) for t in texts)
    return {
        "n": len(recs),
        "tokens_p10": round(q[0]), "tokens_p50": round(statistics.median(lens)),
        "tokens_p90": round(q[8]), "tokens_max": max(lens),
        "tokens_mean": round(statistics.fmean(lens), 1),
        "tokens_total": sum(lens),
        "chars_per_token": round(sum(len(t) for t in texts) / sum(lens), 2),
        "pct_code_fence": round(100 * has_fence / len(recs), 1),
        "pct_indented_code": round(100 * has_indent / len(recs), 1),
        "pct_stack_trace": round(100 * has_trace / len(recs), 1),
        "pct_code_identifier": round(100 * has_ident / len(recs), 1),
        "pct_any_code": round(100 * has_any / len(recs), 1),
        "_lens": lens,
    }


def vocab(recs, tok, cap=400):
    """Token types seen, over a capped sample so the two corpora are compared at
    the same number of prompts rather than at the same number of tokens."""
    import random
    rng = random.Random(7)
    sample = rng.sample(recs, min(cap, len(recs)))
    ids = tok([r["text"] for r in sample], add_special_tokens=False).input_ids
    return {t for seq in ids for t in seq}


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    real = load_real()
    synth = load_synthetic()
    print(f"real {len(real)} prompts, synthetic {len(synth)} prompts\n")

    out = {"real": shape(real, tok), "synthetic": shape(synth, tok), "by_source": {}}
    for src in sorted({r["source"] for r in real}):
        sub = [r for r in real if r["source"] == src]
        s = shape(sub, tok)
        s.pop("_lens")
        out["by_source"][src] = s

    vr, vs = vocab(real, tok), vocab(synth, tok)
    out["vocab"] = {
        "real_types": len(vr), "synthetic_types": len(vs),
        "shared": len(vr & vs),
        "pct_of_real_seen_in_synthetic": round(100 * len(vr & vs) / len(vr), 1),
        "pct_of_synthetic_seen_in_real": round(100 * len(vr & vs) / len(vs), 1),
    }

    hdr = f"{'':22}{'real':>12}{'synthetic':>12}"
    print(hdr); print("-" * len(hdr))
    for k in ("n", "tokens_p10", "tokens_p50", "tokens_p90", "tokens_max",
              "tokens_mean", "tokens_total", "chars_per_token", "pct_code_fence",
              "pct_indented_code", "pct_stack_trace", "pct_code_identifier",
              "pct_any_code"):
        print(f"{k:22}{out['real'][k]:>12}{out['synthetic'][k]:>12}")
    print(f"\nvocabulary (400-prompt sample each)")
    for k, v in out["vocab"].items():
        print(f"  {k:32} {v}")

    print("\nby source:")
    print(f"  {'source':16}{'n':>6}{'tok p50':>9}{'tok p90':>9}{'%code':>8}")
    for src, s in out["by_source"].items():
        print(f"  {src:16}{s['n']:>6}{s['tokens_p50']:>9}{s['tokens_p90']:>9}"
              f"{s['pct_any_code']:>8.0f}")

    for d in (out["real"], out["synthetic"]):
        d.pop("_lens", None)
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "r2_characterisation.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote artifacts/r2_characterisation.json")


if __name__ == "__main__":
    main()
