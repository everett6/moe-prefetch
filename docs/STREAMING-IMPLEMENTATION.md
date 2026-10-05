# Persistent CPU team and just-in-time expert streaming (spec steps 1 to 3)

Plan: `docs/superpowers/plans/2026-10-04-team-and-streaming.md`. Spec:
`docs/superpowers/specs/2026-10-04-just-in-time-expert-streaming-design.md`.
Patch: `patches/0010-team-and-streaming.patch`, on top of 0008 and 0009.

Two switches, both off by default (off = the 0009 engine exactly):

- `LLAMA_MOE_SIDECAR_TEAM=1`. The sidecar computes a missed layer with a
  persistent team of `LLAMA_MOE_SIDECAR_THREADS` (8) threads: one OpenMP
  region opened per batch, with the serve thread as member 0. Before this
  change, each miss started a fresh one-node ggml graph. Same op, same
  per-node code path, same rows. `LLAMA_MOE_SIDECAR_PIN=<first cpu>` pins the
  team; it is unpinned by default.
- `LLAMA_MOE_STREAM=1`. This needs the sidecar, `LLAMA_MOE_EARLY_ISSUE=1` and
  `LLAMA_MOE_BATCH_TABLES=1`. After each layer L's `MOE_POST`, a new GPU op
  `MOE_AHEAD` applies the routers of layers L+1 and L+2 to L's normed input,
  rescaled by g_T / g_L (that is rmsnorm(residual) · g_T, the predictor P10
  measured). It posts the top-8 experts that are not in VRAM. A host mover
  thread copies them earliest-deadline-first on its own copy stream, in
  512 KiB chunks, and drops any copy that would land after its layer reads its
  table. On that same stream it then writes the 4-byte table word that makes
  the expert visible. Other settings: `LLAMA_MOE_STREAM_PER_LAYER` (4) is the
  cap per layer per token, and `LLAMA_MOE_STREAM_RESERVE` (2) is the number of
  extra free slots kept per layer.

Correctness rests on the same rule as the in-token path. A table word flips
only from the dummy to a slot whose copy has completed, into a slot that no
table references. The host computes exactly the experts the GPU's table marked
missed.

## Tests, pre-registered 2026-10-04 (written and pushed before any of them ran on the GPU)

They run after the sidecar queue (S3, S5, S4 in `SIDECAR-IMPLEMENTATION.md`),
once each, in order. A GPU test runs only if the earlier GPU tests passed,
except J4, which is a measurement. Conditions: 175 W power limit and no other
compute process on the GPU (the harness refuses otherwise). Speed tests use
the S3 protocol: 3 rounds × 23 prompts × 200 tokens, a fresh server per arm,
alternating order, paired by prompt and round.

| # | test | gate |
|---|---|---|
| J0 | CPU only: `cpp/test-moe-team.cpp` (team output byte-identical to `ggml_graph_compute` over 6 weight types × 3 thread counts × 500 jobs, observation count equal); `cpp/test-moe-planner.cpp` | PASS |
| J1 | `experiments/x6_team_logits.py`: four prompts, 200 tokens; RECORD (sidecar), REPLAY-S (sidecar, replayed map), REPLAY-T (sidecar + team, replayed map) | both replays have zero unequal logits, and routing is identical to RECORD |
| J2 | `experiments/x6_team_bench.py`: sidecar vs sidecar + team | **team arm ≥ +4 tok/s paired** (spec step 1), no sidecar error |
| J3 | `cpp/test-moe-sidecar` with the new cases (AHEAD candidates against a CPU reference; copy-queue order with the table word written last) | PASS |
| J4 | P11, from J6's stream arm logs: mean lead from a prediction's arrival to its target layer's post, d = 1 and d = 2 | reported; spec threshold d = 1 ≥ 90 µs |
| J5 | `experiments/x7_stream_logits.py`: RECORD (sidecar + team + stream; the recorded map adds, for each token, the streamed experts the GPU used in time, in the slots they used), REPLAY (sidecar + team, replayed map) | zero unequal logits, routing identical, **and** RECORD used ≥ 1 streamed expert in time (otherwise the test is vacuous and fails) |
| J6 | `experiments/x7_stream_bench.py`: sidecar + team + early issue vs the same + stream | **≥ +6 tok/s paired**; stream arm's counters: **≥ 50% of would-be misses served from VRAM** (in time / (in time + misses left)), **≥ 45% of landed copies used in time**; no sidecar error |
| J7 | `experiments/x7_stream_soak.py`: one server with team + stream, ≥ 100,000 decode tokens | zero channel errors |

If J2 fails, the 40 µs fixed cost is not where spec section 7 put it; the
next step is a timeline of one missing layer, not tuning. If J6 fails, J4
and the counters (offered, sent, landed, in time, late, unused, dropped)
show whether prediction, lead time or bandwidth is short. Tuning (spec step
4) is pre-registered separately, one paired change at a time.

## Status, 2026-10-04: built, CPU tests pass, nothing run on the GPU yet

The user asked for no GPU testing for now, so the sidecar queue (S3, S5, S4)
was stopped before it started, and J1 to J7 wait. Patch 0010 builds for
sm_120. 0008 + 0009 + 0010 applied to e67a8e3 reproduce `engine/` exactly
(identical git tree).

**J0: PASS.** Run once:
- `test-moe-team`: 9,000 team jobs byte-identical to `ggml_graph_compute`
  (F32, F16, Q4_K, Q5_K, Q6_K, Q8_0 × 1, 2 and 8 threads × 500 jobs, two team
  regions each), with observation counts equal.
- `test-moe-planner`: passes.
- `test-moe-rows`: still passes (810 comparisons).

`test-moe-sidecar` with the new J3 cases compiles; it needs the GPU.

Where the build departs from the spec's wording, and why:

- **`MOE_AHEAD` is its own op**, right after `MOE_POST`, rather than part of
  it. A layer's misses are announced first, and the ~3 µs prediction does not
  delay the CPU's start on them.
- **The planner runs in the mover thread**, not the serve thread. The serve
  thread spends its time computing misses, and a prediction should not wait
  behind one. The serve thread tells the mover which layers have read their
  tables, through a single-producer queue.
- **The predictor's input** is x_L ⊙ (g_T / g_L), where x_L = rmsnorm(residual_L) · g_L
  is the layer's router input. This equals rmsnorm(residual_L) · g_T, the
  quantity P10 measured, without plumbing the residual into the op. A pair
  whose g_L has a zero weight, or whose router is not F32, is skipped loudly.
- **Recency of a streamed expert:** if its layer routed to it, it is promoted
  like a hit; if not, it gets the oldest recency, so it is the first to go at
  the boundary. The existing speculative probation, which protects a slot for
  one boundary, is not used. Protecting an unused guess would keep it longer.
- **Exactness of streaming (J5).** A token-boundary map cannot express a
  mid-token flip. So with streaming, the recorded frame for a boundary is
  written one step late, with the next token's in-time streamed experts placed
  in the slots they used. Replaying that map without streaming must give the
  same logits. With streaming on, the route trace prints the slot the GPU
  used. Without streaming, that is the host's slot.
- **Dedupe with the upload workers.** A worker copy that is between the
  workers' queues is invisible to the check made when a job is offered. If the
  same expert both streams in and is published by a worker, adoption keeps the
  worker's slot and gives the streamed slot back. The layer's table is
  rewritten from the mirrors at that step's flush.
