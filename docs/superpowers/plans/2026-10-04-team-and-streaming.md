# Persistent CPU team and just-in-time expert streaming: implementation plan

> For agentic workers: executed natively in the session that wrote it; steps
> use checkbox syntax for tracking.

**Goal:** build spec steps 1 and 3 of
`docs/superpowers/specs/2026-10-04-just-in-time-expert-streaming-design.md`
(the persistent CPU team, and router-lookahead streaming with a
stream-ordered table flip), plus the timestamps step 2 reads, as patch 0010
on the isolated engine. Build and CPU-only tests now; every GPU run is
pre-registered in `docs/STREAMING-IMPLEMENTATION.md` and waits for the user.

**Architecture:** the team is one long-lived OpenMP region inside ggml-cpu
(`ggml_cpu_team_run`), so `ggml_barrier` and the chunk counters work
unchanged and member 0 is the serve thread. Streaming adds one CUDA op
(`GGML_OP_MOE_AHEAD`) after `MOE_POST`, a second ring in the mapped channel,
a copy queue exported by the CUDA backend, a pure host planner, and a mover
thread in `llama-moecache.cpp`.

**Tech stack:** C/C++17, CUDA (sm_120), ggml, OpenMP (libgomp).

**Spec:** `docs/superpowers/specs/2026-10-04-just-in-time-expert-streaming-design.md`, sections 6 to 8.

## Global constraints

- Engine work only in `engine/` (own git); export as `patches/0010-team-and-streaming.patch` = `git -C engine diff <0009 head> HEAD`. `~/llama.cpp-build` untouched.
- No GPU process is started while the user has said "no GPU testing"; CUDA code is compiled, not run.
- Both features off by default: `LLAMA_MOE_SIDECAR_TEAM=1`, `LLAMA_MOE_STREAM=1`. Off means the 0009 graph and threads exactly.
- Output exactness is unchanged: the host computes exactly the experts the GPU's table marked missed; a table word flips only dummy → slot after that slot's copy completes, into a slot no table references.

## Review focus

1. A streamed copy still in flight at the token boundary: the slot must not return to the pool or be flushed into a table until the copy queue is synchronised.
2. An expert both streamed and demand-uploaded in the same token (two slots, one expert): the host must dedupe against stream jobs.
3. A record observed before the mover adopts its landed job: the observe path must not count it as pending demand.
4. Non-F32 router weights, or a zero RMS-norm weight (the ratio trick divides by it): lookahead must disable itself for that layer, loudly.
5. A non-OpenMP build: the team must report itself unavailable and the sidecar fall back to the graph path.

---

### Task 1: persistent team in ggml-cpu

**Files:** modify `engine/ggml/src/ggml-cpu/ggml-cpu.c`, `engine/ggml/include/ggml-cpu.h`; create `cpp/test-moe-team.cpp`.

**Produces:**
- `struct ggml_cpu_team * ggml_cpu_team_new(int n_threads, size_t work_size, int first_cpu)`; NULL when not built with OpenMP. `first_cpu >= 0` pins member i to CPU first_cpu + i.
- `void ggml_cpu_team_free(struct ggml_cpu_team *)`.
- `void ggml_cpu_team_run(struct ggml_cpu_team *, struct ggml_tensor * (*next)(void * ud), void (*done)(void * ud, struct ggml_tensor * t), void * ud)`: the caller becomes member 0. `next` (member 0 only) returns the next one-op node or NULL to end the region. Every member computes the node with the same per-node path as `ggml_graph_compute`, then `done` runs on member 0 after a barrier.

- [x] Test `test-moe-team`: the test-moe-rows shapes (F32/F16/Q4_K/Q5_K/Q6_K/Q8_0, threads 1/2/8), 500 jobs per team region with random ids and miss tables (all-hit, all-miss, mixed). Every output must be byte-identical to `ggml_graph_compute` on the same node, with the observe callback count equal.
- [x] Implement, build `ggml-cpu`, and run the test (CPU only) until it passes.

### Task 2: the sidecar uses the team

**Files:** `engine/src/llama-moecache.cpp`.

- [x] When `LLAMA_MOE_SIDECAR_TEAM=1`: at `sidecar_init`, size the work buffer as the maximum `ggml_graph_plan(graph[il], n_threads).work_size` over the layers and create the team (`LLAMA_MOE_SIDECAR_PIN` = first CPU, default unpinned). `sidecar_serve` enters `ggml_cpu_team_run` once per activation. `next` spins on the ring, handles routing-only records itself, and returns `graph[il]->nodes[0]` for a miss. `done` answers and accounts, exactly as the graph path does. Team creation failure logs and falls back.
- [x] The engine builds; `test-moe-rows` and `test-moe-team` still pass.

### Task 3: channel, lookahead op, copy queue (CUDA)

**Files:** `ggml-moe-sidecar.h`, `ggml.h`, `ggml.c`, `ggml-cpu.c` (reject), `ggml-cuda/moe-sidecar.cu/.cuh`, `ggml-cuda/ggml-cuda.cu`, `ggml-rpc.h` (op count 105, patch version 3).

**Produces:**
- Channel additions: `struct ggml_moe_sc_ahead { uint32_t seq, layer, n[2]; int32_t target[2]; uint64_t t_ns; int32_t ids[2][GGML_MOE_SC_MAX_AHEAD]; }`, `ctl->ahead_head`, `ctl->ahead[GGML_MOE_SC_RING]`; `dev->ahead_seq`. `GGML_MOE_SC_MAX_AHEAD 8`. `MOE_POST` records also carry `t_ns`.
- `ggml_moe_ahead(ctx, x, w[2], r[2], tbl[2], int n_targets, int layer, int target_il[2], int dummy[2], int k, host_dev, dev_state)`: dst is F32 [n_expert, 2] score scratch. CUDA: a matvec kernel (one warp per expert row, s = W_T·(x⊙r_T)), then a one-block kernel that takes the top k of s, keeps those with `tbl_T[e] == dummy_T`, writes the record, fences, and bumps `ahead_head`.
- Copy queue: `void * ggml_backend_cuda_moe_mover_new(int device, int max_pending)`, `int64_t ..._mover_push(void *, void ** dst, const void ** src, const size_t * n, int count)` (async H2D on its own non-blocking stream, then an event; returns a ticket, or -1 when full), `int64_t ..._mover_done(void *)` (highest ticket landed, non-blocking), `void ..._mover_sync(void *)`, `void ..._mover_free(void *)`, `bool ..._is_pinned(const void *)`.
- [x] Add the cases to `test-moe-sidecar.cpp` (written, not run): AHEAD candidates against a CPU reference for random x/W/r/tables; mover push/done/sync ordering, with a flip word written last.
- [x] Build the engine (compiles for sm_120).

### Task 4: planner (pure host logic)

**Files:** create `engine/src/llama-moe-planner.h`; create `cpp/test-moe-planner.cpp`.

**Produces:** `class moe_planner` with:
- `moe_planner(size_t chunk_bytes)`, `void reset()`.
- `int offer(int target, int32_t expert, size_t bytes, uint64_t deadline_ns)`: the job id, or -1 for a duplicate (target, expert). A duplicate offered with an earlier deadline lowers the deadline (the d = 1 refresh of a d = 2 job).
- `void posted(int target)`: layer `target` has read its table, so its unfinished jobs are dropped.
- `bool next(uint64_t now_ns, double bytes_per_ns, int hold_after, chunk & out)`: earliest deadline first among live jobs. A job that cannot finish by its deadline (now + remaining / bandwidth) is dropped, not sent. While `hold_after >= 0`, jobs with target > hold_after wait. `chunk{job, offset, len, last}` with len ≤ chunk_bytes, and a job's chunks in order.
- `const job & get(int id)`; job fields `target, expert, bytes, sent, deadline_ns, dropped`.
- [x] Tests: dedupe and deadline lowering; EDF order across targets; 2.7 MiB in 512 KiB chunks; drop on lateness and on `posted`; hold; reset. Run them (CPU only).

### Task 5: streaming integration

**Files:** `engine/src/llama-moecache.cpp/.h`, `engine/src/llama-graph.cpp`.

- [x] `llama_moe_cache_layer` gains `ahead_n`, `ahead_il[2]`, `ahead_w[2]`, `ahead_r[2]`, `ahead_tbl[2]`, `ahead_dummy[2]`. `stream_init(mc, model)` requires the sidecar, early issue and batched tables, and none of replay, record, freeze, deterministic or in-token mode; it refuses loudly otherwise. It computes r = g_T / g_L on the host from the `ffn_norm` weights, disables a pair with any |g_L| < 1e-12 or a router that is not F32, uploads r to device tensors, and starts the mover. The graph inserts `ggml_moe_ahead` after `MOE_POST` and expands it.
- [x] Mover thread, active with the sidecar: reads the ahead ring; for each candidate checks residency, worker jobs and stream jobs under `mc->mtx`; takes a free slot, keeping `max_inserts` for demand and at most `LLAMA_MOE_STREAM_PER_LAYER` (default 4) per layer per token; offers it to the planner. The deadline is arrival + d·τ − 5 µs, with τ an EMA of consecutive records' GPU timestamps (start 103 µs). Issues chunks while under 1 MiB is outstanding; the last chunk of an expert is pushed together with its 4-byte flip. Hold rule: while the serve thread computes a miss, targets beyond the next layer wait. When the sidecar goes inactive: drop the queue, synchronise, mark what landed, skip the rest of the ring, go idle.
- [x] Serve thread: on each record, `posted(il)` to the planner (through an atomic per layer); for stream-marked ids, count in time (slot ≠ dummy) or late, and mark the job used. The observe path skips stream-marked ids for pending demand.
- [x] `step()`: wait for the mover to be idle, then adopt landed jobs (tables via mirrors; recency `++clock` if routed, else 1, so an unused one goes first) and return the slots of jobs that did not land. Pool target += `LLAMA_MOE_STREAM_RESERVE` (default 2).
- [x] Counters every 512 steps and at shutdown: offered, sent, landed, in time, late, unused, dropped late, dropped posted, mean lead per d (P11).
- [x] Build; CPU tests pass.

### Task 6: export, harness, pre-registration

**Files:** `patches/0010-team-and-streaming.patch`, `experiments/build_fused.sh`, `experiments/x6_team_logits.py`, `x6_team_bench.py`, `x7_stream_logits.py`, `x7_stream_bench.py`, `docs/STREAMING-IMPLEMENTATION.md`.

- [x] Export the patch; verify that 0008 + 0009 + 0010 on e67a8e3 reproduce `engine/` exactly. The build script applies 0010 and builds the three test programs.
- [x] Scripts modelled on x4/x5 (same 3 × 23 × 200 protocol); written, not run.
- [x] Commit and push the plan and pre-registration before any GPU run.
