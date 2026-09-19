"""
R3b: why are real prompts faster than the control under the cache?

R3 measured real coding and decision prompts running faster than the 150-
character prose control on both cache configurations, and identically to it on
the baseline. That shape suggests a mechanism rather than noise: decode tok/s is
measured over the decode phase only, so a long prefill is free to the metric --
but it is not free to the cache, which it fills. On the baseline there is no
cache to fill, so prompt length should buy nothing.

The prediction is therefore specific and falsifiable:

  baseline      decode tok/s uncorrelated with prompt length
  cache configs decode tok/s rising with prompt length, flattening once the
                prefill has touched more experts than the cache can hold

Tested by regressing decode tok/s on log prompt tokens within each config, on
the per-prompt records R3 already wrote. Spearman as well as Pearson, because
the expected shape is saturating rather than linear.
"""
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "artifacts")


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(xs, ys):
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def spearman(xs, ys):
    return pearson(rank(xs), rank(ys))


def main():
    path = os.path.join(ART, "r3_real_throughput.json")
    d = json.load(open(path))
    out = {}
    print(f"{'config':22}{'pearson(log tok)':>18}{'spearman':>11}"
          f"{'short':>9}{'long':>9}{'delta':>8}")
    print("-" * 77)
    for label, cfg in d["configs"].items():
        rows = cfg["per_prompt"]
        # average the repeats of each prompt first: otherwise n is inflated by
        # rounds and the correlation is tested against its own repetitions.
        by_id = {}
        for r in rows:
            by_id.setdefault(r["id"], []).append(r)
        pts = [(statistics.fmean([x["n_prompt"] for x in v]),
                statistics.fmean([x["decode_tps"] for x in v]))
               for v in by_id.values()]
        pts.sort()
        xs = [math.log(max(p, 1)) for p, _ in pts]
        ys = [t for _, t in pts]
        half = len(pts) // 2
        short = statistics.fmean(ys[:half])
        long_ = statistics.fmean(ys[half:])
        out[label] = {
            "n_prompts": len(pts),
            "pearson_log_tokens": round(pearson(xs, ys), 3),
            "spearman": round(spearman(xs, ys), 3),
            "short_half_tok_s": round(short, 1),
            "long_half_tok_s": round(long_, 1),
            "delta_tok_s": round(long_ - short, 1),
            "median_tokens": round(statistics.median([p for p, _ in pts])),
        }
        o = out[label]
        print(f"{label:22}{o['pearson_log_tokens']:>18}{o['spearman']:>11}"
              f"{o['short_half_tok_s']:>9}{o['long_half_tok_s']:>9}"
              f"{o['delta_tok_s']:>8}")

    # Variance decomposition. The "+/-" on a real-prompt mean is mostly the
    # spread BETWEEN prompts -- real prompts genuinely differ from each other --
    # while the control's "+/-" is repeat-to-repeat measurement noise on one
    # prompt. Reporting them as the same kind of number would overstate the
    # uncertainty of the real-prompt mean by a wide margin.
    print(f"\n{'config':22}{'between-prompt sd':>20}{'within-prompt sd':>19}"
          f"{'sem of mean':>13}")
    print("-" * 74)
    for label, cfg in d["configs"].items():
        by_id = {}
        for r in cfg["per_prompt"]:
            by_id.setdefault(r["id"], []).append(r["decode_tps"])
        means = [statistics.fmean(v) for v in by_id.values()]
        within = [statistics.stdev(v) for v in by_id.values() if len(v) > 1]
        between = statistics.stdev(means) if len(means) > 1 else 0.0
        sem = between / math.sqrt(len(means))
        out[label].update(
            between_prompt_sd=round(between, 2),
            within_prompt_sd=round(statistics.fmean(within), 2) if within else 0.0,
            sem_of_mean=round(sem, 2),
            mean_tok_s=round(statistics.fmean(means), 2))
        print(f"{label:22}{between:>20.2f}{statistics.fmean(within):>19.2f}"
              f"{sem:>13.2f}")

    with open(os.path.join(ART, "r3b_length_effect.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nshort/long halves are split at the median prompt length of each config.")
    print("wrote artifacts/r3b_length_effect.json")


if __name__ == "__main__":
    main()
