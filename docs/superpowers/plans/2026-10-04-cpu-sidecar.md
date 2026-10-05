# CPU Sidecar (design C, step 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A decode token runs as one GPU graph; missed experts are computed from host RAM by a CPU serve thread while the GPU carries on with the cached experts, exchanging data through mapped pinned memory.

**Architecture:** Two CUDA-only ggml ops replace the CPU expert split of each cached layer: `MOE_POST` writes the layer input and routed ids to mapped host memory and announces them; `MOE_JOIN` runs after the cached chain, waits for the host's rows if the layer missed, and adds them. A serve thread in `llama-moecache.cpp` reads the announcements, observes routing, and computes missed rows with the step-1 op `GGML_OP_MOE_ROWS` on a per-layer CPU graph. Uploads and publication are unchanged.

**Tech Stack:** C/C++17, CUDA 13.2 (sm_120), ggml, llama.cpp fork at `e67a8e3` + patch 0008, Python 3 harness.

**Spec:** `docs/superpowers/specs/2026-10-01-graph-resident-decode-design.md`, sections 6 and 10 (step 2).

## Global Constraints

- Off unless `LLAMA_MOE_SIDECAR=1`; with it unset the graph and behaviour are byte-for-byte patch 0008's.
- Applies only where the step-1 fused op applies: cache path, `n_tokens == 1`, unclamped SwiGLU.
- Exactness: logits bit-identical to the fused path under a replayed placement map (spec 6.6).
- A host answer is awaited at most ~1 s (`%globaltimer`); on timeout the GPU writes `err`, uses zero rows, and the decode call returns an error. Wrong output is never returned silently.
- Start-up self-test of the channel: 10,000 exchanges; on any failure the sidecar is disabled and the fused path used, with a loud log line.
- Serve thread computes with 8 CPU threads (spec 4.2).
- Engine work happens in `moe-prefetch/engine/` only; `~/llama.cpp-build` is not touched. Delivered as `patches/0009-moe-sidecar.patch` on top of 0008.
- GPU runs only at the 175 W power limit with no other compute process on the card.

## Review Focus

1. **A layer that misses nothing** must not wait for the host; the serve thread must still observe its routing exactly once (cache recency depends on it). Test: Task 3's logit replay on all-hit-heavy prompts plus the per-token observation count check.
2. **The host falling behind** (OS deschedules the serve thread) must stall, not corrupt. Test: Task 2's delayed-echo case (host answers after 2 ms) gives correct rows and no error.
3. **A dead serve thread** must surface as an error within about a second, not a hang. Test: Task 2's no-echo case sets `err` and returns.
4. **CUDA graph replay** reuses kernel arguments every token, so sequence numbers must come from device memory. Test: Task 2 runs the same captured graph 1,000 times and checks every sequence.
5. **Prefill and multi-token batches** must keep the old path. Test: Task 3 prompts are processed in batches before decode; logits must still match.

---

### Task 1: The two ops, the shared channel layout, CPU rejection

**Files:**
- Create: `engine/ggml/include/ggml-moe-sidecar.h`
- Modify: `engine/ggml/include/ggml.h`, `engine/ggml/src/ggml.c`, `engine/ggml/include/ggml-rpc.h`, `engine/ggml/src/ggml-cpu/ggml-cpu.cpp` (supports_op), `engine/ggml/src/ggml-cpu/ggml-cpu.c` (n_tasks)

**Interfaces:**
- Produces: `GGML_OP_MOE_POST`, `GGML_OP_MOE_JOIN` (after `GGML_OP_MOE_ROWS`; `GGML_OP_COUNT == 104`; RPC patch version 2).
- Produces: `ggml_tensor * ggml_moe_post(ggml_context *, ggml_tensor * x /*F32 [n_embd,1,1] contiguous*/, ggml_tensor * ids /*I32 [n_used,1]*/, ggml_tensor * slots /*I32 [n_used,1]*/, int32_t layer, int32_t dummy, void * host_dev, void * dev_state)` — result is a copy of `x`, `src = {x, ids, slots}`.
- Produces: `ggml_tensor * ggml_moe_join(ggml_context *, ggml_tensor * cached /*F32 [n_embd,n_used,1]*/, ggml_tensor * slots, int32_t layer, int32_t dummy, void * host_dev, void * dev_state)` — result `[n_embd,n_used,1]`, `src = {cached, slots}`.
- op_params layout for both: `[0]=layer, [1]=dummy, [2..3]=host_dev, [4..5]=dev_state` (pointers split into two int32).
- Produces in `ggml-moe-sidecar.h` (C, header-only): constants `GGML_MOE_SC_MAX_LAYERS 128`, `GGML_MOE_SC_RING 256`, `GGML_MOE_SC_MAX_USED 16`; `struct ggml_moe_sc_record { uint32_t seq, layer, miss, pad; }`; `struct ggml_moe_sc_ctl { uint32_t head; uint32_t err; uint32_t pad[14]; struct ggml_moe_sc_record ring[RING]; uint32_t resp[MAX_LAYERS]; int32_t ids[MAX_LAYERS][MAX_USED]; }`; `struct ggml_moe_sc_dev { uint32_t seq; uint32_t posted_seq[MAX_LAYERS]; uint32_t posted_miss[MAX_LAYERS]; }`; inline `size_t ggml_moe_sc_host_bytes(int n_embd)` and pointer helpers `ggml_moe_sc_x(base, n_embd, layer)` (layer input, `n_embd` floats) and `ggml_moe_sc_rows(base, n_embd, layer)` (`MAX_USED*n_embd` floats), laid out after the ctl struct at 256-byte alignment.
- Protocol: POST writes `seq`; `head = seq + 1`; host writes `resp[layer] = seq + 1` after the rows.

- [ ] **Step 1:** Add the enums, names, symbols, constructors with shape asserts, RPC bump; CPU `supports_op` returns false and `ggml_get_n_tasks` returns 1 for both ops (never scheduled there).
- [ ] **Step 2:** Build `llama-server` in `engine/`; expected: compiles, `test-moe-rows` still prints PASS.
- [ ] **Step 3:** Commit in moe-prefetch (patch regenerated in Task 4, not yet).

### Task 2: CUDA kernels, mapped allocation, self-test

**Files:**
- Create: `engine/ggml/src/ggml-cuda/moe-sidecar.cu`, `engine/ggml/src/ggml-cuda/moe-sidecar.cuh`
- Modify: `engine/ggml/src/ggml-cuda/ggml-cuda.cu` (dispatch, supports_op, proc address)
- Test: `cpp/test-moe-sidecar.cpp`

**Interfaces:**
- Consumes: Task 1 ops and header.
- Produces (proc address on the CUDA reg): `bool ggml_backend_cuda_moe_sc_alloc(int device, int n_embd, void ** host, void ** host_dev, void ** dev_state)` (cudaHostAlloc mapped + zeroed, cudaMalloc + zeroed); `void ggml_backend_cuda_moe_sc_free(void * host, void * dev_state)`; `int ggml_backend_cuda_moe_sc_selftest(int device, int n)` returning the number of failed exchanges (0 = pass, <0 = setup error).
- POST kernel (one block, 256 threads): read `seq` from `dev_state`, `miss = any(slots[i] == dummy)`; write ids to `ctl->ids[layer]`; if miss write `x` to `ggml_moe_sc_x`; copy `x` to the output; write the ring record; every thread `__threadfence_system()`, `__syncthreads()`, then thread 0 stores `posted_seq/posted_miss[layer]`, `dev_state->seq = seq+1`, and volatile `ctl->head = seq+1`.
- JOIN kernel (one block per used expert, 256 threads): if `posted_miss[layer]`, thread 0 spins on volatile `ctl->resp[layer] == posted_seq[layer]+1` with a 1 s `%globaltimer` limit; on timeout sets `ctl->err = 1`. Then `out[e] = (slots[e] == dummy && answered ? rows[e] : 0) + cached[e]`, reading rows with volatile loads after `__threadfence_system()`.

- [ ] **Step 1: Write `cpp/test-moe-sidecar.cpp`** — builds a CUDA-backend graph `post → scale(post) → join` for `n_embd=2048, n_used=8, dummy=56`, with a host echo thread that, for each record with `miss`, writes rows `f(seq, e, i)` for missed experts and then `resp`. Cases, each asserting every element of `join` equals `rows + cached` (missed rows) or `cached` (hit rows) bit for bit:
  - `all_hit` (no slot == dummy): host never answers; output equals cached; `err == 0`.
  - `mixed` (slots 0,3,7 == dummy), graph run 1,000 times (CUDA graphs on): every run correct, sequences 0..999 seen in order.
  - `delayed` (host sleeps 2 ms before answering): correct, `err == 0`.
  - `dead_host` (host never answers a miss): returns within 3 s, `err == 1`.
  - `selftest`: `ggml_backend_cuda_moe_sc_selftest(0, 10000) == 0`.
- [ ] **Step 2:** Run it before the kernels exist; expected: link or "op not supported" failure.
- [ ] **Step 3:** Implement the kernels, dispatch (`case GGML_OP_MOE_POST/JOIN`), `supports_op` true for F32/I32 inputs as specified, alloc/free/selftest, proc-address entries.
- [ ] **Step 4:** Run; expected: `PASS` for all five cases.
- [ ] **Step 5:** Commit.

### Task 3: Serve thread, graph branch, decode hooks

**Files:**
- Modify: `engine/src/llama-moecache.cpp`, `engine/src/llama-moecache.h`, `engine/src/llama-graph.cpp`, `engine/src/llama-context.cpp`
- Test: `experiments/x4_sidecar_logits.py` (three arms through `x3_logits_replay.py`'s runner, generalised to take arm definitions)

**Interfaces:**
- Consumes: Task 2 proc addresses; `ggml_moe_post/join`; `GGML_OP_MOE_ROWS` (`ggml_moe_rows(ctx, up, input, ids, table, gate, down, dummy)`).
- Produces in `llama-moecache.h`: `llama_moe_cache_layer` gains `void * sc_host_dev = nullptr; void * sc_dev_state = nullptr;` (null when the sidecar is off); `void llama_moe_cache_begin_ubatch(uint32_t n_tokens);` and `bool llama_moe_cache_sidecar_ok();`. `llama_moe_cache_requires_sync()` also returns true when the sidecar is on.
- Init (when `LLAMA_MOE_SIDECAR=1`): alloc via proc address, self-test 10,000 (fail → sidecar off, log `moe-cache: sidecar DISABLED (self-test failed: N)`), CPU backend via `ggml_backend_init_by_type(CPU)` with 8 threads, one prebuilt graph per cached layer: `ggml_moe_rows(up_src, x_view, ids_view, host_table, gate_src, down_src, n_slots)` whose input, ids and output tensors point into the mapped buffer (`ggml_moe_sc_x`, `ctl->ids[layer]`, `ggml_moe_sc_rows`). Log `moe-cache: sidecar ENABLED (self-test 10000/10000)`.
- Serve thread: idle on a condition variable; `begin_ubatch(1)` wakes it; it spins on `head` (acquire), and for each record: miss → `ggml_backend_graph_compute(cpu, layer_graph)` (observes routing inside) then release-store `resp[layer] = seq+1`; no miss → call the routing observer directly with an ids tensor over `ctl->ids[layer]`. `llama_moe_cache_step()` (after the synchronize) waits until consumed == `head`, idles the thread, and latches `err`.
- Graph: when `mcache->sc_host_dev`, build `post = ggml_moe_post(mc_inp, selected_experts, mc_slot_ids, il, n_slots, …)`, feed `post` to the cached up/gate mul_mat_id, and set `experts = ggml_moe_join(down_g, mc_slot_ids, il, n_slots, …)` instead of building any host-weight node or the `ggml_add`.
- Decode: call `llama_moe_cache_begin_ubatch(ubatch.n_tokens)` just before `process_ubatch`; after `llama_moe_cache_step()`, if `!llama_moe_cache_sidecar_ok()` log and `return -3`.

- [ ] **Step 1: Write `experiments/x4_sidecar_logits.py`** — arms on the four prompts, 200 tokens: RECORD (fused, live, records map), REPLAY-F (fused, replay), REPLAY-S (`LLAMA_MOE_SIDECAR=1`, replay). Gate: REPLAY-F and REPLAY-S both zero unequal logits and identical routing traces against RECORD; REPLAY-S log contains `sidecar ENABLED` and no channel error.
- [ ] **Step 2:** Run with the unpatched build; expected: REPLAY-S fails (banner missing).
- [ ] **Step 3:** Implement as specified above.
- [ ] **Step 4:** Rerun; expected: gate passes. Check the split count with `GGML_SCHED_PROFILE=128` on one prompt: CUDA splits per graph ≤ 2.
- [ ] **Step 5:** Commit.

### Task 4: Measure, soak, ship

**Files:**
- Create: `experiments/x4_sidecar_bench.py` (from `x3_fused_bench.py`: arms FUSED vs SIDECAR), `patches/0009-moe-sidecar.patch`, `docs/SIDECAR-IMPLEMENTATION.md`
- Modify: `serve.sh` (`SIDECAR=1` adds `LLAMA_MOE_SIDECAR=1 LLAMA_MOE_FUSED_CPU=1`), `patches/README.md`, `README.md`, spec section 10 status

**Interfaces:** Consumes everything above.

- [ ] **Step 1:** Pre-register in `docs/SIDECAR-IMPLEMENTATION.md` and push: S1 Task 2 test PASS; S2 Task 3 logit gate; S3 paired benchmark, 3 rounds × 23 prompts × 200 tokens, gate **≥ 145 tok/s mean on the sidecar arm** and zero channel errors (spec step 2); S4 soak ≥ 100,000 decode tokens with zero channel errors. Falsified below 145 → profile, do not tune.
- [ ] **Step 2:** Run S3 and S4 once each; record results whatever they are.
- [ ] **Step 3:** Export the patch (`git -C engine diff > patches/0009-moe-sidecar.patch` relative to 0008), update docs, commit, push.
