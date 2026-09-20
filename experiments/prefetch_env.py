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
                 victim_fn=None,
                 n_layers=N_LAYERS):
        # capacity: an int (every layer gets the same budget, the shipped
        # behaviour) or a per-layer sequence (N2.1/N2.2 slot-allocation sweep).
        # Layers never share state -- each keeps its own resident set, free
        # pool and recency clock -- so a non-uniform capacity list is exactly
        # as faithful as the uniform case; it just lets one pass answer the
        # question for every layer at once instead of one engine run per layer.
        self.cap = ([capacity] * n_layers if isinstance(capacity, (int, float))
                    else list(capacity))
        self.max_inserts, self.pool_extra = max_inserts, pool_extra
        self.n_layers = n_layers
        self.cost = cost or fit_cost_model()
        self.victim_fn = victim_fn
        self.reset()

    def reset(self):
        self.resident = [set() for _ in range(self.n_layers)]
        self.recency = [dict() for _ in range(self.n_layers)]
        self.free = [self.cap[L] for L in range(self.n_layers)]
        self.inflight = [dict() for _ in range(self.n_layers)]   # expert -> ready flag
        self.clock = 0
        self.stats = {"hit": 0, "lookup": 0, "uploads": 0, "late": 0, "wasted": 0}
        self.layer_stats = [{"hit": 0, "lookup": 0, "uploads": 0}
                            for _ in range(self.n_layers)]
        return self

    def observe(self, layer, ids):
        """Layer `layer` routes to `ids`. Returns how many missed."""
        miss = 0
        ls = self.layer_stats[layer]
        for e in ids:
            self.stats["lookup"] += 1
            ls["lookup"] += 1
            if e in self.resident[layer]:
                self.stats["hit"] += 1
                ls["hit"] += 1
                self.clock += 1
                self.recency[layer][e] = self.clock
            else:
                miss += 1
        return miss

    def admit_demand(self, layer, ids, budget=None):
        """A miss is computed on the CPU. It becomes resident only if an upload is
        issued for it AND published, and the engine issues at most `max_inserts`
        per layer per step out of the same pre-freed pool a prefetch draws from.

        An earlier version installed every miss immediately. That made the model
        admit 54.5 experts a token where the engine measures 23.6, and put depth-0
        throughput at 69 tok/s against a system that runs at 104 -- the simulator
        was paying for uploads the engine never performs.
        """
        if budget is None:
            budget = self.max_inserts
        return self.prefetch(layer, [e for e in ids if e not in self.resident[layer]],
                             budget)

    def _install(self, layer, e):
        # Every install is a real upload -- a demand admission copies the expert
        # to VRAM exactly as a prefetch does. Counting only speculative ones made
        # depth 0 look like 147 tok/s against a system that measures 104, because
        # the ~24 demand uploads a token actually performs were free in the model
        # and charged in reality.
        self.stats["uploads"] += 1
        self.layer_stats[layer]["uploads"] += 1
        if len(self.resident[layer]) >= self.cap[layer]:
            # Eviction policy is pluggable because it turned out to be the
            # binding constraint, not admission: at 93% residency a correct
            # prefetch still displaces something that is about to be used, and
            # those re-fetches cost about three times what wrong predictions do
            # (artifacts/r7b_gated_replay.json). LRU stays the default so every
            # earlier number remains reproducible.
            if self.victim_fn is not None:
                victim = self.victim_fn(layer, self.resident[layer],
                                        self.recency[layer])
            else:
                victim = min(self.recency[layer], key=self.recency[layer].get)
            self.resident[layer].discard(victim)
            self.recency[layer].pop(victim, None)
        self.resident[layer].add(e)
        self.clock += 1
        self.recency[layer][e] = self.clock

    def prefetch(self, layer, ids, budget, min_keep=0):
        """Issue up to `budget` uploads for `layer`. Returns those actually issued.

        `min_keep` reserves pool slots for the demand path, which has ground truth
        where a prefetch has a guess. Without it the predictor drains the pool one
        layer ahead and the demand path finds it empty."""
        issued = []
        for e in ids:
            if len(issued) >= budget or self.free[layer] <= min_keep:
                break
            if e in self.resident[layer] or e in self.inflight[layer]:
                continue
            self.inflight[layer][e] = True
            self.free[layer] -= 1
            issued.append(e)          # counted when it installs, not when issued
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

    def reward(self, n_correct, n_incorrect, n_late=0):
        """Milliseconds of token time saved. See the module docstring: a correct
        prefetch costs no extra upload, an incorrect one costs a whole upload."""
        B, C = self.cost["B_ms_per_miss"], self.cost["C_ms_per_upload"]
        return B * n_correct - C * n_incorrect - B * n_late

    def break_even_precision(self):
        B, C = self.cost["B_ms_per_miss"], self.cost["C_ms_per_upload"]
        return C / (B + C)


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
    be = cm["C_ms_per_upload"] / (cm["B_ms_per_miss"] + cm["C_ms_per_upload"])
    print(f"\nA correct prefetch costs no extra upload -- the demand path would have")
    print(f"fetched that expert anyway -- so it is worth +{cm['B_ms_per_miss'] * 1000:.0f} us.")
    print(f"An incorrect one is a whole wasted upload: -{cm['C_ms_per_upload'] * 1000:.0f} us.")
    print(f"BREAK-EVEN PRECISION = C/(B+C) = {100 * be:.0f}%.")
    cm["break_even_precision"] = be
    os.makedirs(ARTIFACTS, exist_ok=True)
    json.dump(cm, open(os.path.join(ARTIFACTS, "cost_model.json"), "w"), indent=1)
