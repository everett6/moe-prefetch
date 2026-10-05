# Fused CPU expert rows

The first implementation stage of design C lives in patch
`0008-fused-cpu-moe-rows.patch`, against engine revision
`e67a8e3b56eed6fdb1ee518234b32aac9e529ac3` (0006 and 0007 already applied).
It is opt-in. The first full paired run FAILED the continuation gate; the
existing engine and launcher default remain unchanged.

`GGML_OP_MOE_ROWS` dispatches the CPU expert chain once. It uses the existing
`mul_mat_id` kernels for up, gate, and down and the existing vectorized SwiGLU
implementation, with explicit team barriers and graph-plan-owned scratch.
Cache hits produce zero CPU rows, and the normal gate routing observation is
retained. Fully resident layers bypass quantization and the matrix kernels. The GPU expert chain and weighted reduction stay unchanged.

Only the existing cache's separate up/gate/down, unclamped SwiGLU path uses
this operation. Clamped activations retain their original CPU graph. CPU
rows can be computed for multiple tokens, but the engine cache remains
single-token; this patch does not enable speculative verification batches.

## Build and run

```bash
experiments/build_fused.sh
ENGINE="$PWD/engine" LLAMA_MOE_FUSED_CPU=1 ./serve.sh exact
```

The build script clones into `engine/` if needed and applies 0008. It does not
change the existing engine. `SOURCE_ENGINE` overrides the source checkout;
`BUILD_JOBS` defaults to 4. No model is downloaded.

## Verification

`cpp/test-moe-rows.cpp` passed 810 bit-identical comparisons against the
original CPU graph: F32, F16, Q4_K, Q5_K, Q6_K and Q8_0 up/gate weights,
Q6_K down weights, 1/2/8 CPU threads, 1/2/8 tokens, all hits, all misses,
mixed hits, duplicate ids, absent cache tables, differing up/gate dot types, and repeated execution with changing inputs.
Every graph execution also checks that routing is observed exactly once.

Run the paired engine benchmark with GPU access:

```bash
python3 experiments/x3_fused_bench.py
```

It alternates arm order across rounds, compares generated output hashes,
uses the existing server PID/VRAM guards, and records NVIDIA telemetry once
per second. Each run writes a uniquely named `artifacts/x3_fused_*.json`,
including partial results on failure. `ROUNDS`, `N_BENCH`, `N_PREDICT`,
`BENCH_BIN`, `MODEL`, `SLOTS`, and `PORT` can be overridden.

The continuation gate is zero output mismatches and a paired gain of at least
4 tok/s. This stage does not implement the upload arbiter, graph-resident
sidecar, or batched speculative decoding. Those stages remain gated on the
measurement; the mapped-memory probe does not establish a supported channel.

## Measurements from this implementation session

The CUDA server built successfully. The GPU smoke test generated identical
output in five prompt pairs: 79.79 -> 82.09 tok/s, +2.29 +/- 1.38 tok/s SEM.
Its 64-token generations are not comparable to the historical headline.

The full 23-prompt, three-round benchmark generated 200 tokens per request:

| round | original CPU chain | fused CPU chain | tok/s gain |
|---|---:|---:|---:|
| 0 | 89.81 | 91.93 | +2.12 |
| 1 | 109.53 | 114.50 | +4.97 |
| 2 | 114.90 | 119.59 | +4.69 |
| all 69 pairs | 104.75 | 108.67 | +3.93 +/- 0.62 SEM |

Four pairs diverged in round 0; all 46 pairs in rounds 1 and 2 matched.
The artifact is `artifacts/x3_fused_1790922036167939250.json`. Both the zero
mismatch gate and the +4 tok/s gate failed; later rounds cannot erase that.

Three fresh baseline-only runs on the four affected prompts produced
identical hashes across all repeats. The control artifact is
`artifacts/x3_output_control_1790922761518294106.json`. This does not establish
repeatability under every cache state or load condition.

GPU telemetry during the full run peaked at 10,430 MiB used, with at least
1,343 MiB free; maximum temperature was 48 C and power 166.66 W at a 175 W
cap. There was enough VRAM for testing. During a later slow model reload,
host memory had 6.6 GiB of swap in use, so GPU headroom alone does not establish
that the host is suitable for stable throughput measurements.

`LLAMA_MOE_CACHE_FREEZE=1` is a diagnostic switch: it disables new expert
uploads and heat-file writes while retaining a preloaded hot set. The
`x3_frozen_control.py` experiment uses it to compare old and fused execution
with identical mixed CPU/GPU placement. It is not a performance configuration.

The upload arbiter, graph-resident sidecar, and speculative batch support are
not implemented. Resolve output divergence and pass the first gate before
building them. In particular, bit-identical CPU rows do not by themselves
prove bit-identical live-cache output when changing execution timing changes
which upload completions are published at a token boundary.

The fixed-placement GPU control PASSED: all four affected prompts generated
identical output with the original and fused CPU chains. Its artifact is
`artifacts/x3_frozen_control_1790923329660597428.json`. Three prompts differed
between the original chain's live-cache and frozen-cache runs, demonstrating
that expert placement itself can affect greedy output. This supports cache
publication timing as the next thing to investigate; it does not establish a
complete explanation of every live-cache mismatch.

Final checks: CUDA server build, 810 exact CPU comparisons, patch reverse-
application check, Python compilation, shell syntax, and six existing
predictor-format tests passed. Benchmark-owned servers were stopped after
all runs. No commits, upstream checkout changes, driver changes, downloads,
or power-limit changes were made.

## Cache publication investigation (2026-10-02)

The patch now provides these opt-in diagnostics:

- `LLAMA_MOE_MAP_RECORD=/absolute/path`: records the initial published mapping
  and the mapping after every decode step, including empty slots. The header
  fixes the layer, expert, and slot dimensions.
- `LLAMA_MOE_MAP_REPLAY=/absolute/path`: disables uploads, validates that header
  and step sequence, and restores the recorded expert weights and tables at
  each boundary. It synchronizes the graph before overwriting physical slots.
  Replay is for arithmetic diagnosis, not throughput measurement.
- `LLAMA_MOE_ROUTE_TRACE=/absolute/path`: logs routing IDs and their published
  slots per layer and decode step.
- `LLAMA_MOE_DETERMINISTIC=1`: synchronizes the completed graph, waits for every
  already scheduled upload, and publishes completions in layer/slot order.
  A 10-second diagnostic timeout aborts on stalled uploads. It reports average
  publication wait time every 512 steps. This mode may sacrifice throughput;
  it is a candidate fix, not a verified replacement for asynchronous publication.

`cpp/moe-logits.cpp` records every float32 vocabulary logit and greedy token,
then compares all logits with the fused implementation under the recorded
cache placements and teacher-forced baseline token prefix. This keeps later
comparisons meaningful if an earlier token would have diverged. Its JSON
reports the first differing step, unequal logit count, and maximum absolute
numerical difference; bit equality is checked even for signed zero.

Run after building, with the GPU free:

```bash
python3 experiments/x3_logits_replay.py
BENCH_DETERMINISTIC=1 python3 experiments/x3_fused_bench.py
```

The replay script targets the four previously affected prompts and retains
maps, routing traces, token IDs, logs, and `result.json` in a uniquely named
artifact directory. Temporary full-logit files are removed after comparison.
The paired benchmark now retains generated token IDs, reports the first token
mismatch, and samples host available memory and swap-in/out counters alongside
NVIDIA telemetry. Deterministic benchmark results are labeled separately.

The diagnostic CUDA build, standalone logit-runner build, 810 CPU comparisons,
Python compilation, and shell syntax checks passed. **The new map replay,
deterministic mode, and full-logit comparisons have not yet run on the GPU.**
At the GPU check in this session, LM Studio held approximately 8.45 GiB and
only 1.9 GiB was free, below the headroom needed by the measured MoE configuration.
The unrelated LM Studio process was left running. No new performance or
correctness gate result is claimed. The earlier failed gate remains in force.

## Re-run, pre-registered 2026-10-04 (written and pushed before any of it ran)

Why the gate changes. The first run's output gate ("zero mismatches on live
runs") turned out to test the live cache's timing, not the fused op: the
frozen control above shows placement alone changes greedy output. The
correctness gate therefore moves to bit-identical logits under a replayed
placement (spec section 6.6, corrected the same day). The speed gate is
unchanged. The first run's failure stays recorded above; if this run passes,
both runs are reported.

Each test runs once, in this order, on the same build:

| # | test | gate |
|---|---|---|
| T1 | `experiments/build_fused.sh`: build plus the 810 CPU comparisons | all 810 bit-identical |
| T2 | `experiments/x3_logits_replay.py`, the four prompts that diverged, 200 tokens, three arms: RECORD (original, live), REPLAY-0 (original, replayed map), REPLAY-1 (fused, replayed map) | REPLAY-0 identical to RECORD (otherwise T2 is void, not failed) and REPLAY-1 identical: zero unequal logits and identical routing traces |
| T3 | `experiments/x3_fused_bench.py`, live cache, 3 rounds × 23 prompts, 200 tokens, arm order alternating | paired mean gain over all 69 pairs ≥ +4 tok/s; no round dropped. Live output mismatches recorded, not gated |
| T4 | `BENCH_DETERMINISTIC=1 experiments/x3_fused_bench.py`, same protocol | zero output mismatches between arms (tests that deterministic publication removes the timing dependence). Speed recorded, not gated |

**Step 1 passes if T1, T2 and T3 pass.** T4 decides whether deterministic
publication is usable as the exactness harness for step 2. Conditions: power
limit 175 W; no other compute process on the GPU (desktop only); RAM and swap
sampled each second.
