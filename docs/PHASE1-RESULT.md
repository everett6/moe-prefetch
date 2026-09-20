# Phase 1 result: batched uploads reduce copy time, not throughput

PLAN-NEXT.md's Phase 1 asked whether the 142 µs/upload cost could be cut by
coalescing the three per-expert slice copies (up, gate, down) into one
synchronisation instead of three. It can -- and it does not matter.

## What changed

`upload_slice` in `llama-moecache.cpp` called `ggml_backend_tensor_set` once
per slice, and each call did its own `cudaMemcpyAsync` + `cudaStreamSynchronize`
(`ggml_backend_cuda_buffer_set_tensor`). Three slices, three synchronisations,
per expert.

Added `ggml_backend_cuda_memcpy_h2d_batch` (`ggml-cuda.cu`), exposed through
`ggml_backend_cuda_reg_get_proc_address`: N async `cudaMemcpyAsync` calls on
`cudaStreamPerThread`, one `cudaStreamSynchronize`. The cache worker calls it
when available (`LLAMA_MOE_BATCH_UPLOADS=1`) and falls back to the three-call
path otherwise, so the change is zero-risk to ship.

## Measured

Per-expert copy time, live under decode load (`LLAMA_MOE_UPLOAD_STATS=1`):

| | per-expert |
|---|---|
| three syncs (baseline) | 91.1 µs |
| one sync (batched) | 73.3–83 µs |

A genuine 9–20% cut in the instrumented cost, reproduced across multiple runs.

End to end, paired, real prompts, 3 rounds, n=69 (`r3_real_bench.py`,
`WITH_BATCH=1`):

| config | decode tok/s | vs SYNC-PER-SLICE |
|---|---|---|
| SYNC-PER-SLICE | 112.0 ± 10.84 | -- |
| SYNC-PER-EXPERT (batched) | 112.2 ± 10.94 | +0.2 |

sem ≈ 1.3 tok/s on each arm; the difference is noise.

## Conclusion

**Falsified, as the plan itself allowed for.** PLAN-NEXT.md's own falsification
clause: *"If `cudaMemcpyAsync` already runs at link speed and the 142 µs is
dominated by synchronisation the engine needs anyway, this phase stops and
says so."* That is what happened: the copy got faster and the token didn't.
The most likely reason is that upload latency was already hidden behind other
per-step synchronisation the engine performs regardless (the same
`cudaStreamSynchronize` cost shows up elsewhere in the step), so shaving it
here doesn't shorten the critical path.

The code ships anyway -- `LLAMA_MOE_BATCH_UPLOADS` defaults **on** (set it to
`0` to force the old three-call path; that is what the SYNC-PER-SLICE arm
above did) -- because it is strictly not worse and the instrumented number is
real, but it should not be sold as a throughput win.

Acceptance criterion from the plan (≥ +5 tok/s end to end, paired, outside the
noise band): **not met.**
