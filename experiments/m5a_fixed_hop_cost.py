"""
M5a: what does a CPU round trip cost, independent of how many experts it computes?

This is the number that has straddled the bar since milestone 3. The cost of a
layer with `m` of its 8 experts missing from VRAM decomposes as

    layer_cost(m) = FIXED + m * PER_EXPERT

where FIXED is the hidden state crossing to the CPU and back plus the stall that
forces, paid once if m >= 1 and not at all if m == 0. AI2 measured the m = 8 case
directly: SLOPE = 0.3208 ms per fully host-resident expert layer. So measuring
FIXED gives PER_EXPERT = (SLOPE - FIXED) / 8 and collapses the whole range.

Milestone 3 assumed this needed a llama.cpp CUDA build, because --n-cpu-moe
places whole layers and a layer's 128 experts share one tensor. True, and beside
the point: FIXED is a property of the PCIe round trip and the synchronisation,
not of llama.cpp's graph, and both are measurable here.

What makes it expensive is the dependency, not the volume. 2048 floats is 8 KiB
and would take ~0.15 us at 53 GB/s; what costs is that the GPU cannot proceed
until the CPU's answer is back, so the round trip is a latency stall rather than
a bandwidth transfer. That is why the measurement below forces a real
synchronisation each way instead of pipelining.

Four variants, because the honest answer depends on how the real implementation
is written:

  d2h_sync        GPU -> pinned host, synchronised. Half a round trip.
  round_trip      GPU -> host -> GPU, synchronised both ways. The naive version.
  round_trip_ev   the same using CUDA events rather than full device sync, which
                  is what a careful implementation would do.
  launch_only     an empty kernel launch plus sync, to separate the stall that
                  belongs to PCIe from the stall that belongs to driver overhead.

**This is a lower bound on FIXED.** llama.cpp's real path also pays ggml graph
split overhead, backend scheduling, and quantisation/dequantisation at the
boundary. If even the lower bound is a large fraction of SLOPE, the pessimistic
model is right and the argument is over.
"""
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "m5a_fixed_hop_cost_result.json")
SLOPE_MS = 0.3208          # AI2: ms/token per fully host-resident expert layer
HIDDEN = int(os.environ.get("HIDDEN", "2048"))
ITERS = int(os.environ.get("ITERS", "2000"))
WARMUP = 200


def bench(fn, iters=ITERS):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples), statistics.mean(samples)


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA unavailable")
    dev = torch.device("cuda")
    g = torch.randn(HIDDEN, device=dev, dtype=torch.float32)
    h = torch.empty(HIDDEN, dtype=torch.float32, pin_memory=True)
    back = torch.empty(HIDDEN, device=dev, dtype=torch.float32)
    tiny = torch.zeros(1, device=dev)

    def d2h_sync():
        h.copy_(g, non_blocking=True)
        torch.cuda.synchronize()

    def round_trip():
        h.copy_(g, non_blocking=True)
        torch.cuda.synchronize()          # CPU must actually have the data
        back.copy_(h, non_blocking=True)
        torch.cuda.synchronize()

    ev = torch.cuda.Event()

    def round_trip_ev():
        h.copy_(g, non_blocking=True)
        ev.record()
        ev.synchronize()
        back.copy_(h, non_blocking=True)
        ev.record()
        ev.synchronize()

    def launch_only():
        tiny.add_(0)
        torch.cuda.synchronize()

    rows = []
    for name, fn in (("d2h_sync", d2h_sync), ("round_trip", round_trip),
                     ("round_trip_ev", round_trip_ev), ("launch_only", launch_only)):
        med, mean = bench(fn)
        rows.append({"variant": name, "median_ms": med, "mean_ms": mean})
        print(f"  {name:<15} median {med * 1000:7.1f} us   mean {mean * 1000:7.1f} us",
              file=sys.stderr, flush=True)

    print(f"\n=== M5a: FIXED COST OF A CPU HOP ({HIDDEN} floats, {ITERS} iters) ===\n")
    print("%-16s %12s %14s %18s" % ("variant", "median", "as % of SLOPE", "implied PER_EXPERT"))
    for r in rows:
        frac = r["median_ms"] / SLOPE_MS
        per = (SLOPE_MS - r["median_ms"]) / 8
        print("%-16s %9.1f us %13.0f%% %15.1f us"
              % (r["variant"], r["median_ms"] * 1000, 100 * frac, per * 1000))

    best = min(r["median_ms"] for r in rows if r["variant"].startswith("round_trip"))
    per_expert = (SLOPE_MS - best) / 8
    print(f"\nSLOPE (a fully host-resident layer, measured by AI2) = {SLOPE_MS * 1000:.0f} us")
    print(f"FIXED (best round-trip variant)                       = {best * 1000:.0f} us"
          f"  ({100 * best / SLOPE_MS:.0f}% of SLOPE)")
    print(f"PER_EXPERT = (SLOPE - FIXED) / 8                       = {per_expert * 1000:.0f} us")

    print("\nCost of a layer by number of missing experts:")
    print("   misses   cost (us)   vs full layer")
    for m in (0, 1, 2, 4, 8):
        c = 0.0 if m == 0 else best + m * per_expert
        print(f"   {m:6d}   {c * 1000:9.1f}   {100 * c / SLOPE_MS:12.0f}%")

    verdict = ("PESSIMISTIC model is right -- one miss costs nearly a whole layer"
               if best > 0.5 * SLOPE_MS else
               "OPTIMISTIC model is closer -- cost scales with the number of misses")
    print(f"\nVerdict: {verdict}.")
    print("Lower bound: llama.cpp also pays graph-split and backend scheduling "
          "overhead\non top of this, so the real FIXED is somewhat higher.")

    json.dump({"slope_ms": SLOPE_MS, "hidden": HIDDEN, "iters": ITERS,
               "fixed_ms": best, "per_expert_ms": per_expert, "rows": rows},
              open(OUT, "w"), indent=1, default=float)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
