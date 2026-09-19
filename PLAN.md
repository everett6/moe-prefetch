# Plan: land it in llama.cpp

*Written 2026-09-18. Phases A, B and C are closed — see
[`docs/PLAN-abc-closed.md`](docs/PLAN-abc-closed.md). This covers the one thing
left, which is no longer blocked on anything.*

## State

| | |
|---|---|
| projected speed | **133 tok/s** at Q4_K_M (1.71x over 77.9, +23 over the 110 bar) |
| predictor | 68.9% recall@8, per-layer ridge on `h_norm` + prev + cur experts |
| engine | working, PyTorch, real async copies and CUDA-event deadlines |
| toolchain | **built and verified** — conda `cuda-nvcc` 13.2.86, no root |
| baseline | **reproduced on the new binary**: 78.23 ± 0.48 t/s vs 77.9 |
| what's left | the MoE graph surgery |

**PR #27861 is a good starting point and better than expected.** Open, draft,
updated 2026-09-18, and **+645/−0 across 12 files** — purely additive. Nothing to
un-merge, and our predictor is additive to *it*.

## D1 — measure PR #27861 alone (do this first, it is decisive)

Build the `moe-expert-cache` branch with the toolchain that now works, run
`--moe-expert-cache` on Q4_K_M, measure.

This is the highest-value step in the plan and it writes no code. Our simulation
says an LRU cache alone is worth **113 tok/s**; the predictor adds +20 on top. D1
tests the 113 — the larger and more load-bearing half — using someone else's
implementation.

- **If it lands near 113:** the cost model is validated end to end and the only
  open question is whether our predictor adds its +20. Proceed to D2.
- **If it lands far below:** our model of what a cache is worth is wrong, and
  that is worth far more than the predictor is. Stop and find out why before
  writing anything.
- **If it does not build or run:** fall back to D1b.

**D1b, fallback.** Patch mainline minimally to force a known number of a layer's
experts onto the CPU, and time it. That measures `FIXED` — the CPU-hop cost — in
llama.cpp's real graph rather than in the PyTorch proxy, which is the one
constant the 133 figure rests on (measured 10.5 µs; the verdict survives up to
121 µs). Smaller than D1 and it answers the most important question on its own.

## D2 — export the predictor for C++

`src/predictor.py` holds a PCA basis (2048×256), a scale vector, and 47 ridge
matrices (256×128) — **2.06 M parameters, ~8 MB fp32**. Write it as a flat binary
with a small header and a ~40-line loader. No dependency, no format library.

Include the A2 feature layout in the header, because it is the part that will
silently disagree: the input is `[PCA(h_norm) | prev-token experts at L+1 |
this layer's experts]`, in that order, with the training-time scaling baked in.
Get the order wrong and it still runs, just worse.

## D3 — hook it into the cache

Find PR #27861's admission point and call the predictor at `ffn_inp-L`, before
layer L's FFN, inserting the top-8 predictions for layer L+1.

Two things to respect, both already measured here:

- **`E_L` is a predictor input**, so prediction must happen *after* layer L's
  router and *before* its FFN. That window is the bulk of a layer and is enough
  (measured: 4.6–7.8% late arrivals in the PyTorch engine).
- **Top-8, not top-16.** With the A2 predictor those are within noise of each
  other on speed (133.2 vs 133.3) and top-8 moves 30% less traffic.

## D4 — validate, on two criteria

1. **Speed:** Q4_K_M above **110 tok/s**, measured with `llama-bench -ncmoe`
   against the 78.23 now confirmed on this binary.
2. **Byte-identical output under greedy decoding**, against the same build with
   the cache off. Prefetching changes *where weights live*, never which experts
   are selected — so any divergence is a bug, not a trade-off. This is a stricter
   bar than n-gram speculation can pass, which is why speculation must be off
   while testing it.

Then AI2's `model_quality_eval.py` (414 graded problems) as a backstop, since
byte-identity should make it redundant — and if it is not redundant, criterion 2
was not really met.

## Known hazards

- **PR #27861 disables multi-token decode**, and n-gram speculation is
  multi-token decode. Not fatal here — AI2 measured speculation at 0.99x on
  write-from-scratch prompts — but it costs the +8% drafting is worth on code
  edits. Measure both ways.
- **Its cache is not counted in `--fit`**, and launch-time VRAM fitting is how
  this stack chooses a split at all. Our 66/layer needs 9.03 GB; something has to
  reconcile those.
- **Open upstream bugs**: duplicate dummy slot ids breaking batched `mul_mat_id`
  at `n_tokens > 1`, expert tables mutating mid-prefill. Both hit prefill, not
  single-token decode, which is what we measure — but they will hit real use.
- **It is a draft PR.** It can be rebased or abandoned under us. D1 gets a
  measurement out of it before that matters.

## Order

**D1 → (D1b if needed) → D2 → D3 → D4.**

D1 first because it is the only step that can invalidate the rest, and it costs a
build and a benchmark. Everything after it is implementation.

**Rule this repo has now earned four times:** when something looks blocked, check
the block. "Needs root" was wrong about the toolkit, "needs a CUDA build" was
wrong about `FIXED`, the shared-probe collapse was the test not the signal, and
`admit()`-without-copy flattered the baseline. Three of those four were nearly
reported as findings.
