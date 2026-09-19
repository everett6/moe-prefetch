"""
M5b: collapse the range and answer the question.

Since milestone 3 the projection has been a range -- 91.1 to 125.7 tok/s against
a 110 bar -- because of one unmeasured constant. M5a measured it:

    FIXED      = 10.5 us   the CPU hop, paid once per layer that misses anything
    PER_EXPERT = 39 us     CPU compute for one expert
    SLOPE      = 321 us    a fully host-resident layer (8 misses), AI2's measurement

FIXED is 3% of SLOPE, so the pessimistic model -- "any miss costs a whole layer"
-- is wrong, and cost scales with the number of misses almost linearly. That
turns the range into a number.

Applied to the rates milestone 4 measured on REAL copies with REAL deadlines,
not to simulated ones:

    token_ms = FLOOR + n_layers * ( P(layer misses anything) * FIXED
                                    + E[misses per layer] * PER_EXPERT )

Late arrivals count as misses throughout, which is the conservative choice: an
expert whose copy has not landed is one the layer cannot use.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M4 = os.path.join(HERE, "m4_prefetch_bench_result.json")
M5A = os.path.join(HERE, "m5a_fixed_hop_cost_result.json")
OUT = os.path.join(HERE, "m5b_verdict_result.json")

FLOOR_MS = 5.876        # AI2: ms/token with every expert GPU-resident
N_LAYERS, K = 48, 8
BAR = 110.0             # ud-q3_k_xl's speed at Q4_K_M-equal accuracy
TODAY = 77.9            # measured Q4_K_M today, --n-cpu-moe 22


def token_ms(expert_hit, full_layer_rate, fixed, per_expert):
    miss_rate = 1.0 - expert_hit                 # late counts as miss
    e_misses = K * miss_rate                     # per layer, on average
    p_layer_misses = 1.0 - full_layer_rate
    return FLOOR_MS + N_LAYERS * (p_layer_misses * fixed + e_misses * per_expert)


def main():
    for p in (M4, M5A):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    m4 = json.load(open(M4))
    m5a = json.load(open(M5A))
    fixed, per_expert = m5a["fixed_ms"], m5a["per_expert_ms"]

    print(f"FIXED {fixed * 1000:.1f} us, PER_EXPERT {per_expert * 1000:.1f} us "
          f"({100 * fixed / (fixed + 8 * per_expert):.0f}% of a full layer is fixed)\n",
          file=sys.stderr)

    rows = []
    for policy, s in m4["results"].items():
        ms = token_ms(s["expert_hit_rate"], s["full_layer_rate"], fixed, per_expert)
        rows.append({"policy": policy, "expert_hit": s["expert_hit_rate"],
                     "full_layer_rate": s["full_layer_rate"], "token_ms": ms,
                     "tok_s": 1000 / ms})

    print(f"=== M5b VERDICT (measured constants, measured rates, capacity "
          f"{m4['capacity']}/layer) ===\n")
    print("%-16s %11s %14s %11s %11s" % ("policy", "expert hit", "all-8 on time",
                                         "ms/token", "tok/s"))
    print("%-16s %11s %14s %11.2f %11.1f" % ("today (layers)", "--", "--",
                                             1000 / TODAY, TODAY))
    for r in rows:
        print("%-16s %10.1f%% %13.1f%% %11.2f %11.1f"
              % (r["policy"], 100 * r["expert_hit"], 100 * r["full_layer_rate"],
                 r["token_ms"], r["tok_s"]))

    best = max(rows, key=lambda r: r["tok_s"])
    lru = next(r for r in rows if r["policy"] == "lru")
    print(f"\nBar: {BAR:.0f} tok/s (ud-q3_k_xl, at accuracy indistinguishable from Q4_K_M)")
    print(f"Best: {best['policy']} at {best['tok_s']:.1f} tok/s "
          f"-- {'CLEARS' if best['tok_s'] > BAR else 'MISSES'} it by "
          f"{abs(best['tok_s'] - BAR):.1f}")
    print(f"Speedup over Q4_K_M today: {best['tok_s'] / TODAY:.2f}x "
          f"({TODAY:.1f} -> {best['tok_s']:.1f} tok/s)")
    print(f"Prediction's share: {best['tok_s'] - lru['tok_s']:+.1f} tok/s over the "
          f"LRU cache alone ({lru['tok_s']:.1f})")

    # How wrong would FIXED have to be to change the answer?
    lo, hi = fixed, 8 * per_expert + fixed
    crit = None
    for i in range(1, 2001):
        f = lo + (hi - lo) * i / 2000
        pe = (0.3208 - f) / 8
        if pe < 0:
            break
        if 1000 / token_ms(best["expert_hit"], best["full_layer_rate"], f, pe) < BAR:
            crit = f
            break
    if crit:
        print(f"\nRobustness: the verdict flips only if FIXED is actually "
              f"{crit * 1000:.0f} us or more -- {crit / fixed:.0f}x the measured "
              f"{fixed * 1000:.1f} us.")
    else:
        print("\nRobustness: the verdict holds for every value of FIXED up to a "
              "whole layer.")

    json.dump({"fixed_ms": fixed, "per_expert_ms": per_expert, "bar": BAR,
               "today_tok_s": TODAY, "rows": rows,
               "critical_fixed_ms": crit}, open(OUT, "w"), indent=1, default=float)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
