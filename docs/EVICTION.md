# The lever was eviction, not prediction

## How this line of work started

The prefetcher was closed by a specific diagnosis, not by giving up: a *correct*
prefetch still evicts, and at 93.3% residency the evicted expert is usually
needed again. Eviction damage measured about three times the direct cost of
being wrong (`docs/REAL-DATA-RESULTS.md` §8).

That diagnosis says where to look next, so eviction got measured.

## P1 — how much room is there?

Four arms, real decode traces, same engine
(`artifacts/p1_eviction_headroom.json`):

| arm | decode tok/s | vs LRU | uploads/token |
|---|---|---|---|
| LRU, no prefetch | 120.7 | — | 20.5 |
| LRU, oracle prefetch | 131.7 | +11.0 | 22.4 |
| **Belady, no prefetch** | **174.0** | **+53.3** | **8.0** |
| Belady, oracle prefetch | 184.3 | +63.6 | 8.3 |

Perfect eviction is worth **five times** perfect admission, and it works by
cutting uploads 61%. LRU keeps discarding experts that immediately come back.
Belady reads the future, so this is a bound, not a policy.

## P2 — the cheap options first

A counter in the C++ cache beats a model, so the classical policies were
measured before anything was trained (`artifacts/p2_eviction_policies.json`):

| policy | vs LRU | of Belady |
|---|---|---|
| LFU | −8.5 | −16% |
| LFU, 64-token window | −22.2 | −42% |
| LRU-2 | +1.5 | 3% |
| frequency/recency score | +0.5 | 1% |

Frequency-based eviction is actively harmful here. The best implementable policy
gets 3% of the bound. So the gap is real and needs a model.

## P3/P4 — the model

**Target.** Multi-hot over 128 experts: *used at this layer within the next H
tokens*. Predicting exact next-use time would spend most of its loss on
distinctions the policy never makes — between an expert returning in 400 tokens
and one in 900, when both are equally evictable.

**Features.** The same four expert-set blocks the admission model uses, so this
adds no feature plumbing to the engine, and the comparison between the two tasks
stays honest.

**Two traps, both guarded.** The target was first written in the 8-slot format
the admission model uses, which truncated 99.6% of rows, since a 4-token horizon
touches ~19 of 128 experts; it is packed multi-hot now. And the corpus stores
positions in contiguous blocks of 8, so a horizon running past a block end would
call a row safe on evidence never recorded — rows without their full horizon
captured are excluded.

**Result on LOCKED TEST groups** — transformers, numpy, mbpp,
stackoverflow-python, devops, none of them trained on:

| policy | decode tok/s | vs LRU | uploads/token |
|---|---|---|---|
| LRU (ships now) | 94.2 | — | 32.1 |
| **learned eviction** | **119.4** | **+25.2** | 21.4 |
| Belady (bound) | 143.0 | +48.8 | 14.4 |

**52% of the perfect-eviction bound**, +27% throughput, on held-out projects.

## The loop, and where it stopped

Each round was judged by measured throughput, never by loss — the offline
rank-accuracy saturates at 99.9% for every variant and cannot tell them apart.

| round | change | result |
|---|---|---|
| R1 | linearctx, H=4 | +23.8 (49%) |
| R2 | linear / lowrank | +23.8 / +24.6 — cheapest ties best |
| R3 | horizon 2 / 4 / 8 | 46% / 50% / **52%** |
| P5 | sum of three horizons | +0.4 over best single, at 3× MAC — rejected |
| P6 | blend with recency | every λ > 0 is worse — rejected |

P5 and P6 are both negative and both informative. The ensemble says ordinal
information about *how soon* is not the binding constraint. The recency blend
says the model has already subsumed what LRU knows — so the remaining 48% needs
information Belady has by definition and the model cannot get: the future.

P6 took two attempts. The first normalised the cache's absolute tick counter by
the token count, but that counter advances once per observation rather than once
per token, so every non-zero λ was testing near-pure LRU and the sweep was
meaningless. Rank-normalising within the resident set fixed it.

## What ships

`models/evictor-real.bin`, 12.3 MB, 47 layers, 4,096 MAC — the same shape and
cost class as the admission predictor the C++ MOEP loader already runs, and
verified against it: **2.27e-06 max difference, 100% top-8 agreement** on 2,000
LOCKED TEST cases.

It is not yet wired into the engine. The simulated gain is large and measured on
held-out projects, but it is simulated: the environment's cost model is
calibrated (its LRU decode baseline of 120.7 tok/s sits near the 126.2 measured
end-to-end) and that is still not the same as an end-to-end measurement. Wiring
the victim choice into `llama-moecache.cpp` and re-running the R3 benchmark is
the next step, and the number to trust when it exists.

---

## Wired into the engine, and measured: no improvement

The simulated +25.2 tok/s does not survive contact with the real engine.

`patches/0005-eviction-model.patch` adds the evictor to
`llama-moecache.cpp`: the model is loaded from `LLAMA_MOE_EVICTOR`, scored once
per layer per step from the four expert-id lists the cache already holds, and
the victim becomes the lowest-scoring resident expert with recency breaking
ties. Both eviction sites are covered.

**End to end, real prompts, 3 interleaved rounds, n=69 per arm**
(`artifacts/r8_evictor_end_to_end.json`):

| config | decode tok/s |
|---|---|
| LRU | 127.53 ± 8.12 |
| learned eviction | 127.25 ± 8.27 |

Paired over the 23 prompts: **−0.28 tok/s**, 95% CI **[−1.32, +0.77]**,
t = −0.52, and the evictor is faster on 10 of 23 prompts. That is a coin flip.
With `LLAMA_MOE_EARLY_ISSUE=1` as well: 115.5 against 116.4, also nothing.

### The model is working; it is just not worth anything here

It is not a wiring failure. With the evictor loaded the cache measurably
changes: hit rate 93.9% → 94.2%, uploads scheduled 15,646 → 14,947 over the same
deterministic completion. The victims are different and slightly better. 0.3
points of hit rate is worth a few tenths of a tok/s, which is exactly what was
measured and is far below the noise floor.

### Where the simulation went wrong

The environment's LRU baseline matched reality well — 130.2 tok/s simulated
against 127.5 measured — which is why the gain looked credible. The baseline was
right for the wrong reason.

The engine's observation callback returns early for batches, so **the real cache
never sees prefill at all**. The simulation replayed every row through it,
including prefill, where every layer touches all 128 experts. That is a
pathologically hostile access pattern, and it made LRU look far worse than it is
in the engine. Re-running with the cache fed decode rows only, as the engine
does:

| cache is fed | LRU | model | Belady |
|---|---|---|---|
| all rows (what P1–P4 did) | 40.9 | +3.9 | +15.1 |
| decode only (what the engine does) | 133.8 | **+5.2** | +19.5 |

The faithful configuration predicts +5.2, not +25.2, and its LRU baseline of
133.8 at 94.2% hit is the closest match to the engine's 127.5 at 93.9%.

A gap remains: +5.2 predicted against −0.28 measured, and the paired test is
sensitive enough (sem 0.53) that +5.2 would have been obvious. So the
environment is still wrong about something after the prefill correction, and I
have not isolated what. Candidates not yet eliminated: uploads in the engine are
asynchronous and published a step or more later, while the environment installs
them synchronously; and in-flight slots are not evictable, so the real model
often chooses from a smaller candidate set than the simulation gives it.

### What this leaves

Perfect eviction is still worth +19.5 tok/s in the faithful simulation, so the
lever is real even if this model does not reach it. What is not real is the
+25.2 figure, and the honest summary of this work is that a simulated result
was believed one step further than the measurement supported. The end-to-end
number is the one that counts, and it is zero.

The code stays in the tree, off unless `LLAMA_MOE_EVICTOR` is set, because the
next attempt should not have to rebuild the wiring — only the model.
