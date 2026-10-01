"""
Refit the token cost model from this session's A/B runs.

The model this repo has quoted since 2026-09-19,

    ms/token = 4.18 + 0.047 * missed + 0.142 * uploads

was fitted to five configurations spanning 33 to 106 tok/s, one of which was the
cache in its broken state (3,134 queued uploads, 0.6 resident experts a layer).
Its upload term came almost entirely from that point. The admission-gate A/B on
2026-09-30 is the first measurement that moves uploads and misses in OPPOSITE
directions on a working cache, which is what separates the two terms.

Every (config, round) of every r3 artifact given is one observation: the mean
ms/token over that round's real prompts, against the engine's own missed and
uploaded counts for that server instance. Fits:

    free   ms = A + B*missed + C*uploads
    tied   ms = A + K*(missed + uploads)      one price per expert touched on the
                                              host, i.e. the memory-bandwidth model

    python3 experiments/f1_refit_cost_model.py [artifact.json ...]

Reads artifacts only; touches neither GPU nor engine.
"""
import glob
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(os.path.dirname(HERE), "artifacts")
DEFAULT = ["tables", "admit", "slots", "prior", "stack"]


def observations(paths):
    obs = []
    for p in paths:
        d = json.load(open(p))
        rounds = d["rounds"]
        for name, c in d["configs"].items():
            pp = c["per_prompt"]
            per_round = len(pp) // rounds
            stats = c.get("cache_per_round") or []
            if len(stats) != rounds or per_round * rounds != len(pp):
                continue
            for r in range(rounds):
                chunk = pp[r * per_round:(r + 1) * per_round]
                s = stats[r]
                if "missed_per_token" not in s:
                    continue
                ms = float(np.mean([1000.0 / x["decode_tps"] for x in chunk]))
                obs.append({"src": os.path.basename(p), "config": name, "round": r, "ms": ms,
                            "missed": s["missed_per_token"], "uploads": s["uploads_per_token"],
                            "hit": s.get("hit_pct")})
    return obs


def fit(obs, tied):
    y = np.array([o["ms"] for o in obs])
    if tied:
        X = np.array([[1.0, o["missed"] + o["uploads"]] for o in obs])
    else:
        X = np.array([[1.0, o["missed"], o["uploads"]] for o in obs])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return beta, float(np.sqrt(np.mean(res ** 2))), float(np.max(np.abs(res)))


def boot(obs, tied, n=4000, seed=7):
    rng = random.Random(seed)
    # resample whole configs, not rounds: rounds of one config are not independent
    by = {}
    for o in obs:
        by.setdefault((o["src"], o["config"]), []).append(o)
    keys = list(by)
    out = []
    for _ in range(n):
        sample = []
        for _k in keys:
            sample += by[rng.choice(keys)]
        try:
            b, _, _ = fit(sample, tied)
        except np.linalg.LinAlgError:
            continue
        out.append(b)
    a = np.array(out)
    return np.percentile(a, [2.5, 97.5], axis=0)


def main():
    paths = sys.argv[1:] or [p for m in DEFAULT
                             for p in glob.glob(os.path.join(ART, f"r3_real_throughput_{m}.json"))]
    obs = observations(paths)
    if len(obs) < 6:
        sys.exit("not enough observations with cache counters")
    print(f"{len(obs)} observations from {len(paths)} artifacts\n")
    print(f"{'config':18s} {'round':>5s} {'ms/token':>9s} {'tok/s':>7s} {'missed':>7s} {'uploads':>8s}")
    for o in obs:
        print(f"{o['config']:18s} {o['round']:5d} {o['ms']:9.3f} {1000 / o['ms']:7.1f} "
              f"{o['missed']:7.2f} {o['uploads']:8.2f}")

    out = {"n": len(obs), "artifacts": [os.path.basename(p) for p in paths]}
    b, rms, mx = fit(obs, tied=False)
    lo, hi = boot(obs, tied=False)
    print(f"\nfree  ms = {b[0]:.3f} + {b[1] * 1000:.1f} us * missed + {b[2] * 1000:.1f} us * uploads"
          f"     rms {rms * 1000:.0f} us, worst {mx * 1000:.0f} us")
    print(f"      95% (configs resampled): A {lo[0]:.2f}..{hi[0]:.2f} ms   "
          f"per miss {lo[1] * 1000:.0f}..{hi[1] * 1000:.0f} us   per upload {lo[2] * 1000:.0f}..{hi[2] * 1000:.0f} us")
    out["free"] = {"A_ms": float(b[0]), "B_ms_per_miss": float(b[1]), "C_ms_per_upload": float(b[2]),
                   "rms_ms": rms, "worst_ms": mx, "ci95_lo": lo.tolist(), "ci95_hi": hi.tolist()}

    k, rms2, mx2 = fit(obs, tied=True)
    lo2, hi2 = boot(obs, tied=True)
    print(f"tied  ms = {k[0]:.3f} + {k[1] * 1000:.1f} us * (missed + uploads)"
          f"                 rms {rms2 * 1000:.0f} us, worst {mx2 * 1000:.0f} us")
    print(f"      95%: A {lo2[0]:.2f}..{hi2[0]:.2f} ms   per expert touched on the host "
          f"{lo2[1] * 1000:.0f}..{hi2[1] * 1000:.0f} us")
    out["tied"] = {"A_ms": float(k[0]), "K_ms_per_host_expert": float(k[1]), "rms_ms": rms2,
                   "worst_ms": mx2, "ci95_lo": lo2.tolist(), "ci95_hi": hi2.tolist()}

    old = np.array([4.184264935349625, 0.046897897188132705, 0.1415659208910706])
    y = np.array([o["ms"] for o in obs])
    X = np.array([[1.0, o["missed"], o["uploads"]] for o in obs])
    r_old = y - X @ old
    print(f"\nthe 2026-09-19 model on the same observations: rms {np.sqrt(np.mean(r_old ** 2)) * 1000:.0f} us, "
          f"mean error {np.mean(r_old) * 1000:+.0f} us ({'it predicts slower than measured' if np.mean(r_old) < 0 else 'it predicts faster than measured'})")
    out["old_model_rms_ms"] = float(np.sqrt(np.mean(r_old ** 2)))

    m_now = float(np.mean([o["missed"] for o in obs if o["config"] in ("UNIFORM-56", "ADMIT-ALL", "TABLES-PER-ENTRY", "PLAIN-56", "PRIOR-0")]) or 0)
    u_now = float(np.mean([o["uploads"] for o in obs if o["config"] in ("UNIFORM-56", "ADMIT-ALL", "TABLES-PER-ENTRY", "PLAIN-56", "PRIOR-0")]) or 0)
    if m_now:
        tot = b[0] + b[1] * m_now + b[2] * u_now
        print(f"\nat the plain 56-slot cache ({m_now:.1f} missed, {u_now:.1f} uploaded per token), free fit:")
        print(f"  fixed {b[0]:.2f} ms ({100 * b[0] / tot:.0f}%)   misses {b[1] * m_now:.2f} ms ({100 * b[1] * m_now / tot:.0f}%)"
              f"   uploads {b[2] * u_now:.2f} ms ({100 * b[2] * u_now / tot:.0f}%)   -> {1000 / tot:.1f} tok/s")
        print(f"  with no misses and no uploads: {1000 / b[0]:.0f} tok/s")
        out["operating_point"] = {"missed": m_now, "uploads": u_now, "ms": float(tot)}
    json.dump(out, open(os.path.join(ART, "cost_model_v2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
