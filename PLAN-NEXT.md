# Plan N: what to do next, and why

**Status.** Phase 1: done, falsified (docs/PHASE1-RESULT.md) -- batched
uploads cut the instrumented copy cost 91.1 -> ~80 µs but moved end-to-end
throughput +0.2 tok/s on n=69 paired, inside the noise band. Ships anyway
(strictly not worse, off by an env var). Phase 2: done, confirmed
(docs/PHASE2-RESULT.md) -- per-layer slot reallocation, same total VRAM,
measured +3.7 tok/s end to end on n=69 paired (113.8 -> 117.5), meeting the
+3 tok/s acceptance bar. Phase 0 (trace-validated environment) was still never
done; Phase 2's N2.1/N2.2 leaned on the simulator for a *relative* ranking
across layers only, and N2.4's real-engine measurement is what actually
decided it -- see the caveat at the top of docs/PHASE2-RESULT.md before
trusting any other simulated number in this repo.

## Where the time actually goes

Everything below is ordered by this table, not by what is interesting to build.
From the fitted cost model and the measured cache state (93.9% hit, 13.0
uploads/token) at the shipping configuration:

| term | ms/token | share | what it is |
|---|---|---|---|
| A, fixed | 4.18 | **59%** | compute: 48 CPU-offloaded MoE layers + GPU work |
| C, uploads | 1.85 | **26%** | 13.0 uploads/token at 142 µs |
| B, misses | 1.08 | 15% | 22.9 missed experts/token at 47 µs |

All cache work shares **2.92 ms, 41% of the token**. Eliminating every miss and
every upload would give 239 tok/s against the 127.5 measured. That is the
ceiling on everything in this plan, and it is worth stating before proposing
anything.

One calibration problem is already visible: the cost model predicts 140.7 tok/s
for the measured cache state and the engine delivers 127.5. It is 10%
optimistic. Phase 0 deals with that.

---

## Phase 0 — trust the instrument (blocks everything else)

**Why this is first.** The last piece of work produced a confident +25.2 tok/s
from simulation and measured −0.28 in the engine. The cause was found (the
environment replayed prefill through a cache that the engine never exposes to
prefill) and the corrected environment still predicts +5.2 against a measured
−0.28, with a paired test sensitive enough (sem 0.53) to have seen +5.2 easily.
So the environment is still wrong about something, and until it is fixed no
simulated number in this repo should be believed, including the ones this plan
relies on.

**N0.1 — trace capture in the engine.** Add `LLAMA_MOE_TRACE=<path>` to
`llama-moecache.cpp`: one record per step per layer — resident set, in-flight
slots, free-pool size, the victim chosen, the ids observed, and which uploads
were scheduled and published. Binary, fixed-width, no allocation in the hot
path.

**N0.2 — replay the engine's own trace.** Feed that trace to `PrefetchEnv` and
diff step by step: first divergence in resident set, in upload count, in hit.
This localises the disagreement instead of guessing at it.

**N0.3 — fix the environment.** Two known-missing mechanisms, both of which bias
the simulation optimistic:
  - uploads are **asynchronous**: scheduled, copied by a worker, published at a
    later step boundary. The environment installs them synchronously, so a
    simulated miss is repaired sooner than a real one.
  - **in-flight slots are not evictable**, so the real victim chooser picks from
    a smaller candidate set than the environment offers it.

**Gate.** The environment must reproduce, within ±1 tok/s, (a) LRU throughput,
(b) the measured evictor delta of −0.28, and (c) the hit rate and uploads/token
of both arms. Until all three hold, Phases 2 and 3 do not start.

**What would falsify the approach.** If the environment cannot be made to match
even after N0.3, then simulation is the wrong tool for this cache and every
later phase must be measured directly in the engine, which is slower but honest.

---

## Phase 1 — upload cost (26% of the token, and the best-evidenced lever)

**Why.** An upload moves 3.06 MB in 142 µs: **21.6 GB/s on a PCIe 5.0 x16 link
whose theoretical ceiling is ~63 GB/s**. The transfer is running at about a
third of the link. Uploads are 1.85 ms of a 7.11 ms token, so halving their cost
is worth roughly 0.9 ms, about **+17 tok/s** — larger than anything the
prediction work ever offered, and it needs no model.

This phase does not depend on Phase 0, because it is measured end to end.

**N1.1 — find out where the 142 µs goes.** Instrument `upload_slice`: time the
`cudaMemcpyAsync` itself against the surrounding work. Separate transfer time
from per-call overhead. Three expert slices (up, gate, down) are copied
per expert — check whether they are three separate calls and three separate
launches.

**N1.2 — batch the copies.** If the three slices are three calls, coalesce them.
If several experts are uploaded in one step, coalesce those too: one transfer of
n×3.06 MB should approach link bandwidth where 3n small ones do not.

**N1.3 — check the source buffer.** The expert weights live in `CUDA_Host`
(already verified). Confirm the copy is actually from pinned memory and is not
being staged through a bounce buffer.

**N1.4 — stream and overlap.** Check whether uploads use a dedicated stream and
actually overlap compute, or serialise against the graph.

**Measurement.** `experiments/e1_upload_cost.py` re-run after each change to
re-derive C, plus the full R3 real-prompt benchmark (3 rounds, n≥69, paired) for
the end-to-end number. C must fall AND throughput must rise; either alone is a
red flag.

**Acceptance.** ≥ +5 tok/s end to end, paired, outside the noise band.

**Falsification.** If `cudaMemcpyAsync` already runs at link speed and the 142 µs
is dominated by synchronisation the engine needs anyway, this phase stops and
says so.

---

## Phase 2 — spend the same VRAM better

**Why.** Every layer gets exactly 72 slots. Per-layer decode miss rates vary
**5.2×** — L0 misses 1.354 experts of 8, L17 misses 0.261 — and the 12 worst
layers (26% of layers) carry 36% of all misses. Uniform allocation is therefore
spending identical VRAM on layers with very different needs.

VRAM is genuinely capped: 11.1 GB of 12.2 GB in use, and a slot costs 47 ×
2.92 MiB ≈ 137 MB across all layers, so uniform growth beyond ~78 is impossible.
Reallocation is not capped.

**N2.1 — marginal value of a slot, per layer.** For each layer, sweep its slot
count alone in the (fixed) environment and record the miss reduction. This gives
a per-layer marginal-value curve.

**N2.2 — solve the allocation.** Total slots fixed at the current VRAM budget;
maximise total miss reduction. Greedy on marginal value is almost certainly
enough; the curves are concave.

**N2.3 — per-layer slot counts in the engine.** `llama_moe_cache_init` currently
takes one `n_slots`. Extend to a per-layer vector, defaulting to uniform so the
shipped behaviour is unchanged.

**N2.4 — measure.** R3 benchmark, paired, plus hit rate and uploads/token per
layer to confirm the mechanism moved in the predicted direction.

**Acceptance.** ≥ +3 tok/s end to end at equal or lower VRAM.

**Falsification.** If the marginal-value curves are nearly flat across layers
once the cache is warm, the uniform allocation was already near-optimal and this
phase is worth nothing. The 5.2× spread in miss rate is suggestive but it is not
the same as a 5.2× spread in the value of one more slot — that is the thing N2.1
actually measures.

---

## Phase 3 — eviction, revisited only if Phase 0 passes

**Why it is not dead.** In the corrected environment perfect eviction is still
worth +19.5 tok/s, and the trained model measurably improves the real cache
(hit 93.9% → 94.2%, uploads 15,646 → 14,947). The gain is real and too small to
matter. The question is whether the gap to +19.5 is model quality or environment
error, and Phase 0 answers that.

**N3.1** Retrain against the corrected environment, with the reward being
simulated throughput rather than a horizon-classification loss — the offline
rank-accuracy saturates at 99.9% and cannot distinguish models that differ by
20 tok/s.

**N3.2** Feed the model the state it currently cannot see: in-flight slots, free
pool size, and how long ago each resident expert was loaded rather than last
used.

**N3.3** Measure end to end, paired.

**Acceptance.** ≥ +3 tok/s end to end. Anything less and the eviction line
closes for good, with the +19.5 bound recorded as unreachable by this approach.

---

## Phase 4 — close out

- **N4.1** Re-run the full regression set: 26 tests, C++/Python agreement,
  quality grading (22/24 both arms), and the three-config R3 benchmark.
- **N4.2** File the five upstream bugs in `docs/UPSTREAM-BUGS.md`, which are
  documented and still not submitted.
- **N4.3** Re-measure `--no-mmap`. It is worth +6.8 tok/s and the mechanism is
  still unknown; with the trace capture from N0.1 it may finally be explainable.
- **N4.4** Final report, and a single honest headline number.

---

## Rules this plan runs under

1. **The engine's number is the only number.** Simulated results are hypotheses
   until measured end to end, paired, outside the noise band. This plan exists
   because that rule was broken once.
2. **Mechanism before throughput.** Every change must move hit rate or
   uploads/token in the predicted direction. A throughput win with no mechanism
   is a measurement error until proven otherwise.
3. **Check the feature is live.** Four of the five upstream bugs found here were
   silent no-ops, and so was the first evictor wiring. Any new flag prints what
   it did, loudly.
4. **Cheap before clever.** Phase 1 needs no model. Phase 2 needs no model.
   Phase 3 needs a model and comes last for that reason.
5. **Record refutations.** Negative results are kept in the docs with their
   numbers. The most useful findings in this repo so far have been the wrong
   ones, corrected.

## Ordering, and why

Phase 1 first: largest evidenced headroom, no model, no dependency on the
simulator. Phase 0 runs alongside it because it blocks 2 and 3 but not 1.
Phase 2 next: same VRAM, measured spread, small engine change. Phase 3 last, and
only if the instrument can be trusted.
