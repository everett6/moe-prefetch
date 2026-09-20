# Phase 2 result: spending the same VRAM better is worth +3.7 tok/s

PLAN-NEXT.md's Phase 2: every layer gets the same slot budget, but per-layer
decode miss rates vary a lot, so uniform allocation spends identical VRAM on
layers with very different needs.

## Caveat, stated up front

Phase 0 (an engine trace replayed through `PrefetchEnv` to prove the simulator
matches measured throughput within ±1 tok/s) was never done, and the last
project to trust this simulator's absolute numbers without it produced a
confident +25.2 tok/s that measured −0.28 in the engine. Everything in this
phase is built on that same simulator, so nothing here was trusted until it
was checked end to end -- see the "measured" section below, not the
"simulated" one, for what actually counts.

What is defensible without Phase 0: layers never share state in `PrefetchEnv`
(independent resident sets, independent free pools, independent recency
clocks), so whatever bias the simulator carries applies identically to every
layer. A *relative* ranking across layers should survive that bias even
though an *absolute* tok/s number should not be trusted. That's what N2.1/N2.2
use the simulator for; N2.4 is the real test.

## N2.1 -- per-layer marginal value of a slot

`experiments/p7_slot_allocation.py`. Layers are independent, so one full
corpus pass at a given uniform capacity gives every layer's hit rate at that
capacity in a single run -- not one pass per layer. Swept capacity 16..88,
decode rows only (prefill has nothing to predict -- established earlier this
project).

Found and fixed a real gap while building this: the capture corpus records a
row for layer L only when L predicts a "below" target at L+1, so the
model's physically last layer (47 of 48, confirmed via `qwen3moe.block_count`
in the GGUF) never appears as a source row in `data/index-v4-replay.npz` at
all -- zero rows, not low-sample noise. Every prior simulation in this repo
silently never modelled that layer either; it never mattered before because
uniform-vs-uniform comparisons are blind to a layer that behaves the same in
both arms. It matters here: a naive greedy allocator read "zero measured
misses" as "needs nothing" and tried to strip that layer down to the capacity
floor. Fixed by freezing any layer with zero corpus rows at the uniform
baseline, excluded from reallocation entirely.

At the measured layers, baseline (uniform 56): **6.7x spread** in per-layer
decode miss rate (worst 2.354 experts-of-8, best 0.350) -- in the same range
as PLAN-NEXT's originally quoted 5.2x from a different capacity and
methodology.

## N2.2 -- greedy re-allocation

Concave marginal-value curves (confirmed by the sweep) make greedy optimal for
a fixed total. Total budget held at 48 layers x 56 slots = 2688 (not
PLAN-NEXT's 72 -- see the VRAM note below). 34 of 48 layers moved off
uniform; layers 0 and 1 (the highest-miss layers) went from 56 to 88, layer 30
(near-zero miss at baseline) dropped to 32.

Simulated effect of the reallocation, same environment, same Phase-0 caveat:
**+4.5 tok/s** (103.3 -> 107.8).

## Why 56, not PLAN-NEXT's 72

This GPU is sharing VRAM with a live desktop session. A manual check found
uniform 72 measures 11715/12227 MiB in use -- about 512 MiB before cuBLAS's
own handle allocation, which is not enough: `cublasCreate_v2` reproducibly
failed with "the resource allocation failed" on the very first decode step,
100% of the time, in both the uniform-72 and the (higher-per-layer-max)
reallocated-72 arm. Uniform 56 (9040 MiB) has run stably all session. The
budget optimised here is 56; the method is unchanged.

## N2.4 -- measured, real engine, paired

`r3_real_bench.py`, `WITH_SLOTS=1`, 3 rounds, real prompts, n=69 per arm:

| config | decode tok/s | vram |
|---|---|---|
| UNIFORM-56 | 113.8 ± 10.99 | 9040 MiB |
| REALLOC | 117.5 ± 11.60 | 9068 MiB |

**+3.7 tok/s (+3.3%) at equal VRAM** (9068 vs 9040 MiB -- the 28 MiB
difference is noise, not a real cost; total slot count is identical, only its
distribution across layers differs).

Consistent across all 3 rounds individually: +5.7, +1.8, +3.5.

Also a sanity check on the Phase-0 caveat: the simulator predicted +4.5, the
real engine measured +3.7. Same direction, same order of magnitude, on a
completely independent (real-prompt, real-server) measurement. That does not
retroactively validate the simulator's absolute numbers elsewhere in this
repo, but it is evidence the *relative* ranking this phase leaned on was
sound.

## Conclusion

**Acceptance criterion (≥ +3 tok/s end to end at equal or lower VRAM): met.**
Per-layer slot allocation is a real, if modest, win. Shipped as
`LLAMA_MOE_SLOT_PROFILE="il:slots,il:slots,..."` (env var, comma-separated,
any layer not listed keeps the CLI default) -- already wired in
`llama-moecache.cpp` (`slots_for()`, `load_slot_profile()`), the profile used
for this measurement is baked into `r3_real_bench.py`'s `WITH_SLOTS` config
and archived in `artifacts/p7_slot_allocation.json`.
