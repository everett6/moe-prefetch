"""
E1: where the 141.6 us of an expert upload actually goes.

The cost model fitted to measured throughput says one upload costs ~3x the cache
miss it prevents, which is the whole reason speculative prefetching makes this
system slower. That conclusion is only worth acting on if the cost is real rather
than an artifact of how the upload is done, so this measures the components.

What the engine does today, per expert, is three calls to
`ggml_backend_tensor_set`, and each one is:

    cudaMemcpyAsync(dst, src, size, HostToDevice, cudaStreamPerThread)
    cudaStreamSynchronize(cudaStreamPerThread)

with `src` inside the mmap'd GGUF -- ordinary pageable memory. Two things there
could be costing more than they need to: pageable transfers cannot DMA directly
and are staged by the driver at roughly half the bandwidth of pinned memory, and
there is a full stream synchronisation per tensor rather than per expert.

So the arithmetic to check is simple. One expert is ~3.06 MB across the three
projections. At PCIe 4.0 x16, pinned runs near 25 GB/s and pageable near half
that, which predicts ~122 us pageable and ~61 us pinned. If the measurement
lands near 122, the fitted 141.6 us is mostly physics and the conclusion stands.
If it lands far below, the cost is implementation and the conclusion does not.

Also measured: the same transfers while the CPU is running a memory-bound
workload, because during decode the host is streaming expert weights out of the
same DIMMs. An upload that is free on an idle machine is not free there.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "artifacts", "upload_cost.json")
EXPERT_MIB = 2.92
EXPERT_BYTES = int(EXPERT_MIB * 1024 * 1024)
N = int(os.environ.get("N", "300"))


def time_copies(src, dst, n, sync_every, stream=None):
    """sync_every=1 reproduces ggml (sync per tensor); 3 syncs once per expert."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ctx = torch.cuda.stream(stream) if stream is not None else _null()
    with ctx:
        for i in range(n):
            dst.copy_(src, non_blocking=True)
            if (i + 1) % sync_every == 0:
                torch.cuda.synchronize()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def memory_pressure(stop_flag, nbytes=1 << 30):
    """A memory-bound CPU workload, standing in for the expert FFN streaming
    weights out of DIMMs while the upload runs."""
    a = np.empty(nbytes // 8, dtype=np.float64)
    a[:] = 1.0
    while not stop_flag[0]:
        a.sum()


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA unavailable")
    third = EXPERT_BYTES // 3
    res = {"expert_bytes": EXPERT_BYTES, "per_tensor_bytes": third, "n": N}

    pageable = torch.empty(third, dtype=torch.uint8)                 # like the mmap'd GGUF
    pinned = torch.empty(third, dtype=torch.uint8).pin_memory()
    dev = torch.empty(third, dtype=torch.uint8, device="cuda")
    s = torch.cuda.Stream()

    def report(name, us_per_tensor):
        per_expert = us_per_tensor * 3
        gbps = EXPERT_BYTES / (per_expert / 1e6) / 1e9
        res[name] = {"us_per_tensor": us_per_tensor, "us_per_expert": per_expert,
                     "gb_per_s": gbps}
        print(f"  {name:<34} {us_per_tensor:7.1f} us/tensor  "
              f"{per_expert:7.1f} us/expert  {gbps:5.1f} GB/s")

    print(f"\n=== E1: what an expert upload costs ({EXPERT_MIB} MiB in 3 tensors) ===\n")
    print("idle machine:")
    report("pageable, sync per tensor (ggml)", time_copies(pageable, dev, N, 1))
    report("pageable, sync per expert", time_copies(pageable, dev, N, 3))
    report("pinned,   sync per tensor", time_copies(pinned, dev, N, 1))
    report("pinned,   sync per expert", time_copies(pinned, dev, N, 3))
    report("pinned,   dedicated stream", time_copies(pinned, dev, N, 3, s))

    import threading
    stop = [False]
    threads = [threading.Thread(target=memory_pressure, args=(stop,), daemon=True)
               for _ in range(8)]
    for t in threads:
        t.start()
    time.sleep(1.0)
    print("\nwith 8 memory-bound CPU threads (standing in for the expert FFN):")
    report("pageable, sync per tensor, loaded", time_copies(pageable, dev, N, 1))
    report("pinned,   sync per expert, loaded", time_copies(pinned, dev, N, 3))
    stop[0] = True
    for t in threads:
        t.join(timeout=2)

    fitted = 141.6
    ggml = res["pageable, sync per tensor (ggml)"]["us_per_expert"]
    best = min(res[k]["us_per_expert"] for k in res if isinstance(res[k], dict)
               and "us_per_expert" in res[k])
    print(f"\nThe cost model fitted to end-to-end throughput says {fitted:.1f} us/upload.")
    print(f"The path the engine uses measures {ggml:.1f} us/expert here.")
    print(f"The best available path measures {best:.1f} us/expert "
          f"({100 * (1 - best / ggml):.0f}% less).")
    res["fitted_us"] = fitted
    res["ggml_path_us"] = ggml
    res["best_path_us"] = best
    res["reducible_fraction"] = 1 - best / ggml
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
