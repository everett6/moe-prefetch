# Scope: how large a model can this actually target?

"Q4_K_M on a larger model" runs into a wall that is worth stating in numbers
before any code is written, because it decides what gets built.

## The memory budget

| | |
|---|---|
| VRAM | 12 GB (11.94 usable, ~0.2 held by the desktop) |
| System RAM | **29 GB total, ~26 GB available** |
| Practical ceiling for weights | **~36 GB** (12 VRAM + 26 RAM, leaving the OS room) |
| NVMe free | 499 GB |

Note the RAM number is 29 GB, not 32 — that is what `free` reports on this box.

## What fits

| model | Q4_K_M size | fits in 36 GB? |
|---|---|---|
| Qwen3-30B-A3B (AI2's model) | 17.3 GiB | **yes**, comfortably |
| Qwen3-Next-80B-A3B | ~45 GiB | no |
| GLM-4.5-Air 106B-A12B | ~60 GiB | no |
| Qwen3-235B-A22B | ~130 GiB | not remotely |

So **"Q4_K_M" and "a larger model" cannot both be satisfied from RAM.** One of
three things has to give.

## Option A — 30B-A3B at Q4_K_M (recommended)

Keep the model, take the quality. Q4_K_M is the reference quantization this
machine has never been able to run fast: 77.9 tok/s, because 22 of 48 expert
layers sit in host RAM. That is precisely the condition prefetching is for — the
more host-resident layers, the more there is to hide.

- **Everything fits in RAM.** No third tier, no NVMe path, no new failure modes.
- **The model is already downloaded** and already has 414 graded problems of
  accuracy baseline in AI2 (HumanEval 150/164, GSM8K 239/250).
- **The win condition is crisp:** beat 110 tok/s and this project has produced
  something that does not otherwise exist. Miss it and `ud-q3_k_xl` was the
  better answer all along — which is a real, publishable result either way.
- Prefetch slack is genuinely there at this speed: 0.255 ms/layer of transfer
  against a 12.8 ms/token (77.9 tok/s) budget — **~0.27 ms/layer available**.
  Roughly break-even, and the whole point is that overlapping makes it free.

## Option B — a genuinely larger model, streaming from NVMe

Add a third tier: NVMe → RAM → VRAM, with the RAM tier as an LRU cache over a
model that does not fit in it.

- This is a much more ambitious and much more interesting project. It is also
  where predictive prefetching stops being an optimization and becomes *the
  thing that makes it work at all* — at NVMe latency, a miss is catastrophic,
  so prediction quality directly sets the token rate.
- **The honest risk:** a 30B-A3B token needs ~656 MB of experts. A consumer NVMe
  does 3–7 GB/s, so a cold token costs ~100–200 ms — **5–10 tok/s**. It only
  works if the RAM cache hit rate is very high, and the measured predictability
  (44–48%) is nowhere near high enough on its own. It would live or die on the
  token predictor being much better than the naive baseline.
- Requires downloading 45–60 GB and the disk space is there.

## Option C — larger model, lower quantization

Drops the Q4_K_M requirement, which was the point of the exercise. An 80B at Q2
would be ~25 GiB and fit. But AI2 already showed that quantizing down is the easy
lever, and this project exists to stop doing that.

## C1 result (2026-09-18): priced, and it is a separate project

NVMe was measured directly rather than assumed —
`experiments/c1_nvme_tier.py`, 300 random 2.92 MiB reads (one expert's worth)
against a probe file larger than free RAM so the page cache cannot serve them:

```
per-expert read from NVMe:  median 1,164 us,  p90 1,291 us  (2.63 GB/s)
  vs computing it on the CPU:   39 us   (30x)
  vs fetching it from RAM:      57 us   (20x)
```

The RAM tier's hit rate then decides everything, and it is grounded in this
project's own measurement rather than a guess: an LRU cache holding 52% of each
layer's experts achieves **85.4%** on real traces (`m3_simulate_speedup.py`), and
~26 GB of RAM over a 45–60 GB model is 43–58% resident — the same shape.

| model | RAM hit | disk reads/token | added ms | tok/s |
|---|---|---|---|---|
| 80-layer | **85.4%** (measured shape) | 8.60 | 12.88 | **44.1** |
| 80-layer | 90% | 5.89 | 9.88 | 50.8 |
| 80-layer | 95% | 2.94 | 6.62 | 60.9 |
| 62-layer | **85.4%** | 6.66 | 9.98 | **56.9** |

**Verdict: feasible, roughly 44 tok/s for an 80-layer model — and a separate
project rather than a continuation.** It is not the catastrophe the plan
predicted (that guess was 5–10 tok/s, and it was wrong because it costed a whole
token's experts against disk instead of only the ~9% that fall through two cache
tiers). But it is a third of what the 30B now does, for a 45–60 GB download, a
fresh trace capture, a re-fit of every per-layer probe, and an architecture whose
routing behaviour nothing here has measured.

Nothing transfers automatically except the code. That is a reasonable thing to
start deliberately; it is not a reasonable thing to drift into.

## Recommendation

**Start with Option A**, and treat Option B as the follow-on it naturally is.

The reason is not caution — it is that A and B share their entire foundation.
The expert tracer, the token predictor, the router-derived expert oracle, the
async prefetch engine and the hit-rate harness are identical in both. A proves
the mechanism works, in a regime where everything fits in RAM and a miss costs
0.4 ms instead of 150 ms. If the predictor clears the 47.8% bar in A, B becomes
worth its download. If it does not, B was never going to work, and finding that
out on a model that is already on disk costs nothing.
