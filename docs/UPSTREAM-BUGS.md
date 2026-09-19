# Bugs found upstream

Five, found while trying to make [llama.cpp PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
(MoE expert cache) go fast on a 12 GB card. Patches in [`patches/`](../patches/),
against `bccbacd`. Four of the five are *silent* — the feature appears enabled
and does nothing — which is why they cost so much to find.

## 1. `--moe-expert-cache` silently does nothing

`patches/0001` · **severity: the flag is inert**

`llama_moe_cache_init` guards with a one-shot global:

```c
if (g_init_done) return;
if (n_slots <= 0) { g_init_done = true; return; }   // latches
```

llama.cpp builds a **dry-run context for memory estimation before the real one**,
carrying default params (`n_moe_cache_slots = 0`). That latches the guard, so the
real context returns at the first line. The branch logs nothing: the flag parses,
plumbs correctly all the way through, allocates nothing, and throughput is
unchanged. Fix: do not latch on that path.

## 2. The default insert rate makes the cache slower than no cache

**severity: 33 tok/s against 45 with the feature off**

A slot leaves service when an upload is scheduled and returns only when the
upload is published a step later. At the default `--moe-expert-cache-inserts 2`
the scheduler outruns the uploader and the queue is unstable:

| inserts | tok/s | hit rate | resident/layer | worker backlog |
|---|---|---|---|---|
| 1 | 105.7 | 84.9% | 65.8 / 66 | 10 |
| **2 (default)** | **33.1** | **15.7%** | **3.4 / 66** | **3,001** |
| 4 | 31.4 | 3.1% | 0.3 / 66 | 3,149 |

It is a positive feedback loop: an empty cache misses more, which schedules more,
which deepens the backlog, which keeps the cache empty. Measured on
Qwen3-30B-A3B Q4_K_M, `-ncmoe 48`, RTX 5070.

Two fixes, either sufficient: default the insert rate to 1, or issue uploads
during the graph out of slots freed at the previous step boundary so a slot is
never out of service across a whole step (`patches/0002`, which reaches 118 tok/s).

## 3. A cache too large to allocate silently becomes no cache

**severity: silent 2.5x slowdown**

`--moe-expert-cache 84` (and 96, 104) allocates nothing and runs on: VRAM sits at
the no-cache figure and throughput is the no-cache figure. 78 fails loudly at
load; 84 and above do not. The allocation failure path warns via `LLAMA_LOG_WARN`,
which does not reach `llama-server`'s stderr, so in practice there is no signal
at all.

## 4. `cudaHostRegisterReadOnly` is assumed supported

`patches/0003` · **severity: pinned host memory silently unavailable**

`ggml_backend_cuda_register_host_buffer` calls

```c
cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
```

On an RTX 5070 with driver 595.91.07 that returns `operation not supported` for
**any** size, while the identical call without `ReadOnly` succeeds at 16 GiB.
`ReadOnly` is a hint, so falling back to plain `Portable` keeps pinned memory
working rather than giving it up. Measured difference between the two paths:
86.6 µs versus 188.8 µs per 2.92 MiB transfer idle, and 94.7 versus 297.4 while
the CPU is streaming weights from the same DIMMs.

(Related, not a bug so much as dead code: `ggml_backend_register_host_buffer` is
exported by proc address and **nothing in llama.cpp ever looks it up**, so
`GGML_CUDA_REGISTER_HOST` cannot take effect through any code path.)

## 5. `-ot` cannot target host buffer types

`patches/0004` · **severity: pinned placement unreachable from the CLI**

`parse_tensor_buffer_overrides` builds its list from
`ggml_backend_dev_buffer_type(dev)` only, so `-ot ...=CUDA_Host` fails with
"unknown buffer type". A tensor can be placed on the device or in ordinary
pageable CPU memory, never in pinned host memory. Adding
`ggml_backend_dev_host_buffer_type(dev)` to the same map is four lines.

## An observation, not a bug

`--no-mmap` is worth **+6.8 tok/s** (111.4 ± 2.5 → 118.2 ± 1.7, n=15 each) on
this workload with the cache on. The mechanism is **not established**: the expert
weights report the `CUDA_Host` buffer in both cases, so the obvious explanation —
that mmap denies them pinned memory — is ruled out. Resident set differs by
0.55 GB, which hints at page-cache overhead, but that is a guess and is recorded
as one.
