# Training on real prompts: what changed

Every number in this repo used to come from 616 prompts I wrote myself. This
records what happened when they were replaced with 6,720 real ones — real GitHub
issues, real Stack Exchange questions, real crowd-sourced programming tasks —
targeting coding and decision-making.

## 1. The old corpus was not a smaller version of the real thing

| | real | synthetic |
|---|---|---|
| prompts | 6,720 | 616 |
| tokens p50 / p90 | 123 / 1,049 | 14 / 17 |
| contains any code | 52.5% | **0.0%** |
| share of ROWS that are human text | 48% | **0.15%** |

The last row is the one that matters most. The synthetic corpus was 616 short
prompts and 6.4 M rows of the model continuing them — 99.8% of it was the model
talking to itself. Only 3.7% of the token types in real traffic appear anywhere
in it.

## 2. The headline throughput survives, and improves

Measured end-to-end, three rounds interleaved, n=69 per config
(`artifacts/r3_real_throughput.json`):

| config | real prompts | my single control prompt |
|---|---|---|
| AI2 shipped, no cache | 78.7 | 79.1 |
| **this work + `--no-mmap`** | **126.2** | 118.2 |

**1.60× on real coding and decision traffic**, against 1.49× on my prompt.

The more useful correction is what the error bar means. Repeating one prompt
measures within-prompt noise of 2.70. Real workloads vary *between* prompts by
7.27 — three times as much, and invisible to any single-prompt benchmark. The
old `± 1.66` was answering a narrower question than it appeared to.

## 3. Real routing is much harder to predict

Stage A, same sweep, same budget (`artifacts/experiments.csv`):

| | synthetic (v3) | real (v4) |
|---|---|---|
| best val recall@8 | 67.8% | 49.6% pooled |
| winning architecture | resmlp-h128 | linearctx |

The architecture ranking **inverted**. The residual MLPs that won on prose lose
on real data and now exceed the MAC budget for nothing.

## 4. Fitting the model to the phase it runs in is worth +2.95 points

The real index is 48% prefill and 52% decode, and they are different problems.
The prefetcher only runs at decode. Three arms, all scored on the same decode
validation rows (`artifacts/r5b_phase.json`):

| trained on | decode val@8 | decode hard |
|---|---|---|
| everything | 71.34% | 67.76% |
| **decode only** | **74.29%** | **70.82%** |
| prefill only (control) | 35.26% | 34.01% |

The prefill-only control collapsing on decode rows is the evidence that these
are two problems, not one with extra data. The pooled 49.6% above decomposes
into 71.3% decode and 39.0% prefill — dragged down by rows that never occur in
deployment.

## 5. Prefill has nothing to predict, measured

Distinct experts used at one layer within a window of w consecutive positions
(`artifacts/r6_prefill_union.json`):

| window | prefill | decode |
|---|---|---|
| 1 | 8.0 | 8.0 |
| 32 | **128.0 (all of them)** | 52.8 |
| 512 | **128.0 (all of them)** | — |

At any prefill batch of 32 or more, every layer uses every expert. Prefill's 94%
miss rate is unavoidable demand traffic, not mispredictions.

Incidentally: over 128 consecutive decode tokens a layer draws on ~75 distinct
experts against 66 cache slots. The cache is slightly smaller than the working
set it chases.

## 6. The stopping criterion reverses — and the model still cannot reach it

E5 asked what a *perfect* predictor would be worth. On the synthetic corpus:
**−0.4 tok/s**, which is why this project stopped. On real traces, with
prefetching gated to decode and both arms costed over the same decode tokens:
**+11.0 tok/s (~9%)** (`artifacts/oracle_ceiling_index-v4-replay-decode.json`).

The simulated decode baseline is 120.7 tok/s against 126.2 measured in §2, which
is independent evidence the cost model is calibrated rather than fitted to
itself.

So there is headroom. The model cannot capture it:

| policy | precision | break-even | verdict |
|---|---|---|---|
| prefetch depth-1 always (E2) | 50.1% warm | 75% | loses |
| prefetch only when confident (R7) | up to 92% | 75% | wins, but small |

Confidence gating *is* profitable where unconditional prefetching is not — and
it is worth **at most +1.9 tok/s**, about 17% of an already-modest ceiling,
before eviction costs which that simulation does not model.

**Conclusion: the predictor is still not worth shipping.** But the reason has
changed from "even perfection is worthless" to "perfection is worth 9% and this
model reaches a sixth of it", which is a model-quality target rather than a dead
end.

## 7. The model, exported and verified anyway

`models/predictor-real.bin`, 12.3 MB, 47 layers, trained from scratch on real
decode rows. C++ loader agreement on 2,000 cases from LOCKED TEST groups:
**3.16e-06 max difference, 100% top-8 agreement**.

## What I got wrong on the way

Recorded because the corrections are most of the value here.

- **"Real prompts miss the cache 13× more."** Wrong: it compared a prefill-heavy
  corpus against a decode-only one. In the phase that matters it is 1.7×.
- **E5 "+11.9 tok/s".** My decode-only mode removed prefill *rows*, which also
  removed the cache warming they do. Decode then started cold.
- **E5 "+92.7 tok/s".** Per-phase accounting with the baseline costed over every
  token and the oracle over decode tokens only.
- **R7 "100% precision, +28.6 s saved".** Filtered candidates against the
  previous token's 8 experts instead of the real 66-slot cache, crediting
  prefetches of experts already resident — the same class of error that produced
  a wrong break-even figure earlier in this project.
- **Two confounds in my own benchmark**: the control always ran first against a
  colder cache, and prompts were ordered by length band so "longer" and "later"
  moved together.
- **A fixed artifact filename in E5** let a v4 run silently destroy the committed
  v3 result, which I then briefly quoted back as though it were v3's.

---

## 8. Why the predictor cannot be made to pay: the break-even was wrong

§6 left the project with a target: a perfect predictor is worth +11.0 tok/s, the
model reaches 50.1% precision, the bar is 75%. That framing turned out to be
wrong, and the correction is the most useful thing in this document.

Running the confidence-gated policy through the full environment — which models
installation, eviction, the slot pool and the 2-inserts-per-step throttle —
gives (`artifacts/r7b_gated_replay.json`):

| threshold | issued/tok | precision | break-even says | measured | extra uploads/tok |
|---|---|---|---|---|---|
| 0.00 | 2.71 | 91.7% | +85 µs | **−50 µs** | +0.84 |
| 0.90 | 2.33 | 97.6% | +98 µs | **−30 µs** | +0.64 |
| 0.99 | 1.21 | **99.7%** | +56 µs | **+2 µs** | +0.28 |

Every one of these clears the 75% bar comfortably. Every one of them loses
throughput, except the last, which breaks even. Robust across prefetch depths
2, 4 and 8.

**The break-even formula prices only the wrong prefetches.** `C/(B+C) = 75%`
asks whether a mistaken upload costs more than the miss it might have avoided.
It says nothing about what a *correct* prefetch costs — and a correct prefetch
still evicts a resident expert. In a cache already 93.3% effective, nearly
everything resident is about to be used again, so the eviction causes a re-fetch
later.

The arithmetic: at threshold 0, the policy issues 2.71 prefetches per token at
91.7% precision. Only 0.22 of those are wrong. Yet uploads rise by 0.84 per
token. The missing ~0.6 are re-fetches of experts that the *correct* prefetches
evicted — **eviction damage is about three times the direct cost of being
wrong.**

So the real bar is not 75% precision. It is somewhere near 99.7%, where the
policy finally stops losing money, and at that precision it issues so little
(1.21 per token against the ~24 misses per token available) that it captures
**0% of the +11.0 ceiling**.

### What this means

The predictor does not ship, and now the reason is structural rather than a
matter of model quality. It is not that this model is not good enough. It is
that prefetching into a near-optimal LRU cache has to displace something useful
to insert something useful, and at 93% residency there is nothing cheap left to
evict. A better model would have to be nearly perfect *and* high-volume at once,
and §6's oracle is the only thing in that class.

For the record the model does see novelty — 52.2% recall@8 on non-resident
experts against 6.25% for chance (`artifacts/r7c_why.json`). My first hypothesis
was that it was blind to exactly the experts that mattered; that is measurably
false and is recorded as such. It is weaker on absent experts than resident ones
by 22.5 points, but the reason prefetching fails is eviction, not blindness.
