"""
Q1: what does cache-prior routing cost in quality?

LLAMA_MOE_CACHE_PRIOR=delta adds delta to the router's SELECTION probability of
every expert that is already in VRAM, so a resident expert can replace a
non-resident one the router ranked up to delta higher. Unlike everything else
in this repo it CHANGES THE MODEL'S OUTPUT, so throughput alone is not a result
for it: a delta that doubles the hit rate and wrecks the text is worth nothing.

The published numbers this has to be held against:
  arXiv 2412.00099  cache-prior re-ranking: >50% fewer misses for 0.1-3% PPL
  arXiv 2608.18261  tolerance rerouting on Qwen3-30B (this model): ~80% fewer
                    misses for 2-3% PPL

Why not llama-perplexity: it evaluates in batches, and the expert cache -- and
therefore the prior -- only exists on the single-token decode graph. A batched
perplexity run would measure a model the prior never touches and report zero
cost. So this scores human-written text ONE TOKEN AT A TIME through the same
server and the same graph that decode uses: for each position, the log
probability the model gives the token that actually comes next.

Reported per delta: perplexity, its change against delta=0, and the cache
counters from the same run, so quality and miss rate come from one process.

NOT YET RUN (written 2026-09-30 with the GPU busy). The first response is
checked for the probability format and the script stops if it does not find
it, rather than scoring garbage.
"""
import json
import math
import os
import statistics
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import r3_real_bench as rb  # noqa: E402

DELTAS = [float(x) for x in os.environ.get("PRIOR_DELTAS", "0,0.005,0.01,0.02,0.05").split(",")]
N_TEXTS = int(os.environ.get("N_TEXTS", "24"))
CONTEXT = int(os.environ.get("CONTEXT_TOKENS", "64"))     # prefilled, not scored
SCORED = int(os.environ.get("SCORED_TOKENS", "128"))      # scored one at a time
N_PROBS = int(os.environ.get("N_PROBS", "64"))
URL = f"http://127.0.0.1:{rb.PORT}"


def tokenize(text):
    r = requests.post(f"{URL}/tokenize", json={"content": text, "add_special": True},
                      timeout=120)
    return r.json()["tokens"]


def next_token_logprobs(prefix):
    """Top-N log-probabilities for the token after `prefix`. The prefix differs
    from the previous call's by one token and the prompt cache is on, so the
    server decodes exactly one token -- the single-token graph the cache and the
    prior live on."""
    r = requests.post(f"{URL}/completion", json={
        "prompt": prefix, "n_predict": 1, "n_probs": N_PROBS, "temperature": 0,
        "top_k": 0, "seed": 0, "cache_prompt": True, "id_slot": 0}, timeout=600)
    j = r.json()
    cp = j.get("completion_probabilities")
    if not cp:
        raise RuntimeError("server returned no completion_probabilities -- "
                           f"keys were {sorted(j)}")
    first = cp[0]
    if "top_logprobs" in first:
        return {e["id"]: e["logprob"] for e in first["top_logprobs"]}
    if "top_probs" in first:
        return {e["id"]: math.log(max(e["prob"], 1e-30)) for e in first["top_probs"]}
    raise RuntimeError(f"unrecognised probability format: {sorted(first)}")


def score_text(tokens):
    """(sum of negative log-likelihood, tokens scored, tokens found in top-N)."""
    nll, n, found = 0.0, 0, 0
    for i in range(CONTEXT, min(len(tokens), CONTEXT + SCORED)):
        lp = next_token_logprobs(tokens[:i])
        want = tokens[i]
        if want in lp:
            nll -= lp[want]
            found += 1
        else:
            # Censored: the true token is below the N-th. Charge it the N-th
            # log-probability minus one nat. The same rule in every arm, and
            # `found` is reported so a difference in censoring cannot hide.
            nll -= min(lp.values()) - 1.0
        n += 1
    return nll, n, found


# TAG keeps a second model's result from overwriting the first's.
OUT_NAME = ("q1_cache_prior_quality-" + os.environ["TAG"] + ".json") if os.environ.get("TAG") \
    else "q1_cache_prior_quality.json"


def main():
    recs = rb.load_prompts()
    # Long human-written text only: the scored span has to be the author's
    # words, not a template, and long enough to be worth a cache.
    pool = [r for r in rb.select(recs, max(N_TEXTS * 3, 72)) if r["chars"] > 2500]
    if len(pool) < 4:
        sys.exit("not enough long prompts to score")

    out = {"context_tokens": CONTEXT, "scored_tokens": SCORED, "n_probs": N_PROBS,
           "deltas": {}}
    texts = None
    for d in DELTAS:
        label = f"Q1-PRIOR-{d:g}"
        env = {} if d == 0 else {"LLAMA_MOE_CACHE_PRIOR": f"{d:g}"}
        rb.EXPECT[label] = "admission gate off" if d == 0 else "CACHE-PRIOR ROUTING"
        print(f"\n[{label}]", flush=True)
        p, vram = rb.launch(label, rb.BASE56, env)
        try:
            if texts is None:
                texts = []
                for r in pool:
                    t = tokenize(r["text"])
                    if len(t) >= CONTEXT + SCORED:
                        texts.append({"id": r["id"], "source": r["source"], "tokens": t})
                    if len(texts) >= N_TEXTS:
                        break
                print(f"  {len(texts)} texts, {CONTEXT} context + {SCORED} scored tokens each")
            tot_nll, tot_n, tot_found, per_text = 0.0, 0, 0, []
            for t in texts:
                nll, n, found = score_text(t["tokens"])
                tot_nll += nll; tot_n += n; tot_found += found
                per_text.append({"id": t["id"], "nll_per_token": nll / max(n, 1)})
            ppl = math.exp(tot_nll / max(tot_n, 1))
            out["deltas"][f"{d:g}"] = {
                "ppl": ppl, "nll_per_token": tot_nll / max(tot_n, 1), "tokens": tot_n,
                "found_in_top_n": tot_found / max(tot_n, 1), "vram": vram,
                "cache": rb.cache_stats(label), "per_text": per_text}
            print(f"  ppl {ppl:.4f}  ({tot_n} tokens, {100*tot_found/max(tot_n,1):.1f}% "
                  f"in top-{N_PROBS})  cache {rb.cache_stats(label)}", flush=True)
        finally:
            rb.stop(p)

    base = out["deltas"].get("0")
    print("\n=== cache-prior routing: quality against miss rate ===")
    print(f"{'delta':>8}{'ppl':>10}{'vs 0':>9}{'paired t':>10}{'hit':>8}{'missed/tok':>12}"
          f"{'uploads/tok':>13}")
    for k, v in out["deltas"].items():
        dp, t = "", ""
        if base and k != "0":
            dp = f"{100 * (v['ppl'] / base['ppl'] - 1):+.2f}%"
            diffs = [a["nll_per_token"] - b["nll_per_token"]
                     for a, b in zip(v["per_text"], base["per_text"])]
            if len(diffs) > 1 and statistics.stdev(diffs) > 0:
                t = f"{statistics.fmean(diffs) / (statistics.stdev(diffs) / len(diffs) ** 0.5):+.2f}"
            v["ppl_change_pct"] = 100 * (v["ppl"] / base["ppl"] - 1)
        c = v["cache"]
        print(f"{k:>8}{v['ppl']:>10.4f}{dp:>9}{t:>10}{str(c.get('hit_pct', '')):>8}"
              f"{str(c.get('missed_per_token', '')):>12}{str(c.get('uploads_per_token', '')):>13}")
    print("\nA delta is only usable if its perplexity change is one you would accept "
          "for the miss reduction beside it. Nothing here is on by default.")

    os.makedirs(rb.ART, exist_ok=True)
    with open(os.path.join(rb.ART, OUT_NAME), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote artifacts/{OUT_NAME}")


if __name__ == "__main__":
    main()
