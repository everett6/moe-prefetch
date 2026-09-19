"""
Stage C/D: the prefetch decision as a sequential decision problem, and a cost
model fitted to measured throughput rather than assumed.

## Why this exists

D1's correction removed the ground under the old cost model: the 113 tok/s it
"validated" was never measured. So the reward here is not derived from first
principles and then trusted -- it is FITTED to end-to-end throughput actually
measured on this machine, at configurations that span the whole range from a
collapsed cache (33 tok/s) to a healthy one (106).

The fitted form is deliberately the simplest one that can express the two effects
the measurements show:

    ms_per_token = A + B * missed_experts_per_token + C * uploads_per_token

  A  floor: everything resident, no traffic
  B  what a cache miss costs -- the CPU has to compute that expert
  C  what an upload costs -- PCIe and host RAM bandwidth, shared with the very
     CPU expert compute the cache is trying to avoid

C is the term the draft's cost model never had, and it is why the draft predictor
made the real system slower while improving Recall@8: prefetching more experts
raises the hit rate and pays for it in bandwidth. Any reward that omits C will
rediscover exactly that mistake.

## The decision problem

STATE, at layer L of token t -- everything the host genuinely has at that moment:
  - the experts layer L just routed to (`cur`), known because the callback fires
    inside layer L's own mul_mat_id
  - the experts layer L+1 used on the previous token (`prev`)
  - context: layer L-1 this token, layer L on the previous token
  - the cache occupancy for layer L+1: which of the 128 experts are resident
  - how many pre-freed slots layer L+1 has available right now
  - the layer index, and a running hit rate

ACTION:
  - how many experts to prefetch for layer L+1, from 0 to the free-slot budget
  - which ones (ranked by the supervised model's scores)

The "how many" half is the part supervised learning cannot do, and the part that
matters. A predictor trained on Recall@8 always wants to fetch 8. The measured
system says the right answer is usually 0 -- the cache already has them -- and
occasionally 2 or 3. That is a policy decision about cost, not a prediction.

REWARD, per layer-step, in milliseconds of token time saved:
    +B for each expert that would have missed and is now resident in time
    -C for each upload issued
    -B for each expert whose copy arrives late (paid for, no benefit)
so the agent is optimising modelled ms/token directly, and the units are the
units of the thing the project is actually judged on.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARTIFACTS = os.path.join(ROOT, "artifacts")
N_EXPERT, K, N_LAYERS = 128, 8, 48

# Measured on this machine with the validated harness (see docs/D1-RESULT.md).
# uploads/token and hit rate come from the in-engine counters; tok/s from the
# server's own decode timings.
MEASURED = [
    # label,             uploads/token, expert hit rate, tok/s
    ("no-cache",                   0.0,           0.000,  45.0),
    ("shipped-inserts2",          86.4,           0.234,  33.1),
    ("shipped-inserts1",          21.1,           0.849, 105.7),
    ("early",                     23.6,           0.910, 104.0),
    ("early+pred3",               30.2,           0.894,  97.1),
]


def fit_cost_model(rows=MEASURED):
    """Least squares for ms/token = A + B*missed + C*uploads.

    Five configurations spanning 33 to 106 tok/s. Three parameters, so this is
    over-determined -- the residuals are reported because a model that cannot fit
    its own calibration points has no business scoring a policy.
    """
    X, y, labels = [], [], []
    for label, up, hit, tok in rows:
        missed = N_LAYERS * K * (1.0 - hit)
        X.append([1.0, missed, up])
        y.append(1000.0 / tok)
        labels.append(label)
    X, y = np.asarray(X), np.asarray(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    return {"A_ms": float(coef[0]), "B_ms_per_miss": float(coef[1]),
            "C_ms_per_upload": float(coef[2]),
            "fit": [{"label": l, "measured_tok_s": 1000.0 / yy,
                     "modelled_tok_s": 1000.0 / pp, "residual_pct": 100 * (pp - yy) / yy}
                    for l, yy, pp in zip(labels, y, pred)],
            "max_abs_residual_pct": float(np.max(np.abs(100 * (pred - y) / y)))}


class PrefetchEnv:
    """Replays a real trace through the engine's actual mechanics.

    Faithful to llama-moecache.cpp as this project patched it: a pool of slots
    freed at the previous step boundary, uploads issued from the observe callback
    during the graph, publication at the step boundary, LRU refill of the pool.
    Nothing here can prefetch an expert the real engine could not.
    """

    def __init__(self, capacity=66, max_inserts=2, pool_extra=3, cost=None,
                 n_layers=N_LAYERS):
        self.capacity, self.max_inserts, self.pool_extra = capacity, max_inserts, pool_extra
        self.n_layers = n_layers
        self.cost = cost or fit_cost_model()
        self.reset()

    def reset(self):
        self.resident = [set() for _ in range(self.n_layers)]
        self.recency = [dict() for _ in range(self.n_layers)]
        self.free = [self.capacity for _ in range(self.n_layers)]
        self.inflight = [dict() for _ in range(self.n_layers)]   # expert -> ready flag
        self.clock = 0
        self.stats = {"hit": 0, "lookup": 0, "uploads": 0, "late": 0, "wasted": 0}
        return self

    def observe(self, layer, ids):
        """Layer `layer` routes to `ids`. Returns how many missed."""
        miss = 0
        for e in ids:
            self.stats["lookup"] += 1
            if e in self.resident[layer]:
                self.stats["hit"] += 1
                self.clock += 1
                self.recency[layer][e] = self.clock
            else:
                miss += 1
        return miss

    def admit_demand(self, layer, ids):
        """A miss is computed on the CPU and then becomes resident, exactly as the
        engine's LRU does. Costs no prefetch budget."""
        for e in ids:
            if e not in self.resident[layer]:
                self._install(layer, e)

    def _install(self, layer, e):
        if len(self.resident[layer]) >= self.capacity:
            victim = min(self.recency[layer], key=self.recency[layer].get)
            self.resident[layer].discard(victim)
            self.recency[layer].pop(victim, None)
        self.resident[layer].add(e)
        self.clock += 1
        self.recency[layer][e] = self.clock

    def prefetch(self, layer, ids, budget):
        """Issue up to `budget` uploads for `layer`. Returns those actually issued."""
        issued = []
        for e in ids:
            if len(issued) >= budget or self.free[layer] <= 0:
                break
            if e in self.resident[layer] or e in self.inflight[layer]:
                continue
            self.inflight[layer][e] = True
            self.free[layer] -= 1
            issued.append(e)
            self.stats["uploads"] += 1
        return issued

    def step_boundary(self):
        """Publish in-flight uploads and refill each layer's free-slot pool."""
        for L in range(self.n_layers):
            for e in list(self.inflight[L]):
                self._install(L, e)
            self.inflight[L].clear()
            self.free[L] = self.max_inserts + self.pool_extra

    def tok_s(self, n_tokens):
        s = self.stats
        hit = s["hit"] / max(s["lookup"], 1)
        missed = self.n_layers * K * (1 - hit)
        up = s["uploads"] / max(n_tokens, 1)
        ms = (self.cost["A_ms"] + self.cost["B_ms_per_miss"] * missed
              + self.cost["C_ms_per_upload"] * up)
        return 1000.0 / max(ms, 1e-6), hit, up

    def reward(self, n_newly_covered, n_issued, n_late):
        """Milliseconds of token time saved by this decision."""
        B, C = self.cost["B_ms_per_miss"], self.cost["C_ms_per_upload"]
        return B * n_newly_covered - C * n_issued - B * n_late


if __name__ == "__main__":
    cm = fit_cost_model()
    print("=== cost model fitted to measured throughput ===\n")
    print(f"  ms/token = {cm['A_ms']:.3f} + {cm['B_ms_per_miss']:.4f} * missed_experts "
          f"+ {cm['C_ms_per_upload']:.4f} * uploads\n")
    print("%-20s %14s %14s %10s" % ("config", "measured tok/s", "modelled", "residual"))
    for f in cm["fit"]:
        print("%-20s %13.1f %14.1f %9.1f%%"
              % (f["label"], f["measured_tok_s"], f["modelled_tok_s"], f["residual_pct"]))
    print(f"\nworst residual {cm['max_abs_residual_pct']:.1f}%")
    print(f"\nOne cache miss costs {cm['B_ms_per_miss'] * 1000:.1f} us of token time;")
    print(f"one upload costs {cm['C_ms_per_upload'] * 1000:.1f} us.")
    print(f"So a prefetch only pays if its chance of converting a miss exceeds "
          f"{100 * cm['C_ms_per_upload'] / max(cm['B_ms_per_miss'], 1e-9):.0f}%.")
    os.makedirs(ARTIFACTS, exist_ok=True)
    json.dump(cm, open(os.path.join(ARTIFACTS, "cost_model.json"), "w"), indent=1)
