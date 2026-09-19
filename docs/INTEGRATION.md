# Integration spec: putting this into llama.cpp

The measurements are done and the verdict cleared its bar (129 tok/s at Q4_K_M
against a 110 bar, 1.66x over today's 77.9). This is what remains, written so
that whoever has root can treat it as a defined job rather than a research
question.

## Correction: this never needed root, and it is already done

This document said three times that the CUDA toolkit needs a root install.
**That was wrong.** `cuda-nvcc` is available through conda, which is
user-writable here, and the whole toolchain installs into an isolated
environment without touching the system:

```bash
conda create -y -n cudabuild -c pkgs/main \
    cuda-nvcc=13.2.86 cuda-cudart-dev cuda-cccl libcublas-dev cuda-nvtx
```

13.2.86 matches the driver (595.91.07 reports CUDA 13.2) and targets `sm_120`.
Verified by compiling and running a trivial kernel on the 5070 before anything
larger was attempted.

(PyPI's `nvidia-cuda-nvcc-cu12` looks like an alternative and is not one — it
ships `ptxas` and `nvvm` for JIT use, not the `nvcc` compiler driver.)

### The build, which works

```bash
export CUDA_HOME=$HOME/miniconda3/envs/cudabuild
export PATH=$CUDA_HOME/bin:$PATH
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCMAKE_EXE_LINKER_FLAGS="-L$CUDA_HOME/lib -Wl,-rpath,$CUDA_HOME/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L$CUDA_HOME/lib -Wl,-rpath,$CUDA_HOME/lib"
cmake --build build -j 16
```

The two linker flags are not optional. Without them `libggml-cuda.so` compiles
fine and then every executable fails to link, because `libcudart.so.13` and
`libcublas.so.13` live in the conda env and are not on the linker's search path.
The errors look like missing CUDA symbols and read as a broken toolchain; they
are a missing `-L`.

Run anything built this way with
`LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH`.

### Baseline confirmed on the new binary

```
| qwen3moe 30B.A3B Q4_K_M | 17.28 GiB | CUDA | ngl 999 | n_cpu_moe 22 | tg128 | 78.23 ± 0.48 t/s |
```

**78.23 tok/s against the 77.9 this project has used as its baseline throughout**
— an independent build reproducing the number the whole verdict rests on. The
110 bar and the 133 projection are measured against a figure that now has two
independent sources.

## What remains

## The shape of the change

Three pieces. The first two exist here in Python and are mechanical to port; the
third is the only real engineering.

### 1. The expert cache — `src/prefetch_engine.py` → C++

A per-layer slab of `capacity` expert slots in VRAM plus an expert-id → slot map
with LRU eviction. **66 experts/layer is the operating point**: 9.03 GB of slabs
plus ~1.8 GB of non-expert weights fits 11.94 GB, while 80/layer needs 12.75 GB
and does not.

Refill on a dedicated `cudaStream_t` from pinned host memory with
`cudaMemcpyAsync`. All three matter — a pageable source forces a synchronising
staging copy, and the default stream serialises against compute. Record a
`cudaEvent_t` per in-flight copy; an expert whose event has not completed when
the layer runs is **not** a hit and must fall back to the CPU path.

### 2. The predictor — `src/predictor.py` → C++

Per-layer ridge probes over a shared PCA basis. Ships as one `.npz`: a
`2048 x 256` PCA matrix, a 256-vector of scales, and 47 matrices of `256 x 128`.
**2.06 M parameters, ~8 MB in fp32** — small enough to keep resident.

Inference per layer is one 256x128 matvec on the projected hidden state: ~33K
multiply-adds against the 2.92 MiB fetch it decides. It must run on `ffn_inp-L`
*before* layer L's FFN executes, so the copy has that compute to hide behind.

Prefetch **top-12 to top-16**, not top-8. The engine demands ~8–10 GB/s of the
53.66 available, and the extra depth is worth +3 to +5 tok/s.

### 3. The MoE graph — the actual work

This is where llama.cpp's structure fights back, and it is worth being explicit
about why.

A layer's 128 experts live in **one tensor** (`blk.N.ffn_{gate,up,down}_exps.weight`),
and `ggml_mul_mat_id` indexes into it. A per-expert cache means the gather has to
read from cache slabs for resident experts and from host memory for the rest,
within one op. That is the same problem llama.cpp PR #27861 is solving, and its
open issues are the honest preview of what this costs: duplicate dummy slot IDs
breaking batched `mul_mat_id` at `n_tokens > 1`, expert tables mutating
mid-prefill, and the cache being a per-process singleton.

**Start from PR #27861 rather than from mainline.** It already has the cache and
the graph surgery; this project's contribution is the predictor feeding it, which
is additive. Reimplementing its cache to re-derive its bugs would be a poor use
of the time.

## Two things that will differ from the numbers here

1. **Graph-split and backend-scheduling overhead** sits on top of the measured
   `FIXED = 10.5 us`. The robustness margin covers it: the verdict survives up to
   `FIXED = 121 us`, twelve times measured.
2. **Speculation.** PR #27861 currently disables multi-token decode, and n-gram
   speculation is multi-token decode. On this stack that is not fatal — AI2
   measured speculation at **0.99x** on write-from-scratch prompts, where it has
   nothing to draft — but it costs the +8% that drafting is worth on code edits.

## How to know it worked

Run AI2's harness against the patched binary:

```
MODELS=q4_k_m python3 experiments/model_quality_eval.py     # 414 graded problems
python3 experiments/rerun_original_benchmark.py             # five-prompt decode
```

Two acceptance criteria, and the second matters as much as the first:

- **Speed:** Q4_K_M above **110 tok/s** decode. The projection is 129.
- **Quality: byte-identical output under greedy decoding.** Prefetching changes
  *where weights live*, never which experts are selected or what is computed. Any
  divergence is a bug, not a trade-off — and note this is a stricter test than
  speculation could ever pass, since n-gram drafting is *not* bit-identical
  (`AI2/experiments/spec_determinism.py`).
