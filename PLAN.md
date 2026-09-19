# Plan: close the last unknown, then decide

## Where this stands

Milestone 4 left one number between this project and a verdict:

> **91.1 – 125.7 tok/s** against a **110** bar. The 35-point spread is entirely
> one unknown — whether a layer with a single missing expert pays the whole CPU
> round trip or just that expert's share.

Everything else is measured: hit rates on real traces (86.5% expert, 66.9%
all-8-on-time), copies on real hardware, deadlines on real CUDA events, 4.64%
late. The prefetcher works. What is unknown is what its hit rate is *worth*.

## The insight that unblocks this

Milestone 3 said measuring the partial-miss cost needs a llama.cpp CUDA build,
because `--n-cpu-moe` places whole layers and `-ot` cannot split the experts
inside one — a layer's 128 experts live in a single tensor
(`blk.N.ffn_{gate,up,down}_exps.weight`). That is true, and it is also the wrong
way to get the number.

The cost of a layer with `m` missing experts decomposes:

```
    layer_cost(m)  =  FIXED  +  m x PER_EXPERT
```

- `FIXED` — the hidden state crossing to the CPU and back, plus the
  synchronisation that forces. Paid once if `m >= 1`, not at all if `m == 0`.
  **This is measurable directly with PyTorch, today**: it is a round trip of a
  2048-float tensor with a real dependency stall, on this GPU and this PCIe slot.
- `PER_EXPERT` — CPU compute for one expert. Falls out of AI2's measured
  `SLOPE = 0.3208 ms` for a fully host-resident layer (`m = 8`):
  `PER_EXPERT = (SLOPE - FIXED) / 8`.

Which resolves the two bounds that have straddled the bar since milestone 3:

- `FIXED ≈ SLOPE` → the pessimistic bound is right, any miss costs a whole layer
- `FIXED ≈ 0` → the optimistic bound is right, cost is proportional to misses

No toolkit, no build, no root. It was measurable all along.

## Steps

**M5a — measure `FIXED`.** Round-trip a 2048-float hidden state GPU→CPU→GPU with
the synchronisation a real MoE layer would need (the layer cannot proceed until
the CPU's result is back, so the stall is real, not amortised). Report against
`SLOPE = 0.3208 ms`.

**M5b — collapse the range.** Recompute projected tok/s from the measured
`FIXED` and the *measured* milestone 4 rates (86.5% expert hit, 66.9% all-8),
rather than the two bounds. This produces a single number and a verdict against
the 110 bar.

**M5c — spend the bandwidth headroom.** The engine demands ~7 GB/s of the 53.66
available; there is 7x of slack and only 4.64% lateness. Prefetching the probe's
top-16 instead of top-8 should convert some of that slack into hit rate. The
oracle says +39 points of all-8 are on the table where we take +9.7, so this is
the cheapest place to look for more.

**M5d — decide, and write it down either way.** If the collapsed number clears
110, produce the llama.cpp integration spec so the CUDA build is a well-defined
job for someone with root. If it does not, record that `ud-q3_k_xl` was the
better answer and stop — the same conclusion AI2's Phase E reached, but arrived
at with the mechanism actually built and measured rather than estimated.

## Not doing, and why

- **No llama.cpp patch yet.** It needs the CUDA toolkit as root and it is the
  expensive step. M5b decides whether it is worth anyone's afternoon.
- **No larger model yet.** Option B from `docs/SCOPE.md` was always gated on the
  mechanism proving out on 30B first.
