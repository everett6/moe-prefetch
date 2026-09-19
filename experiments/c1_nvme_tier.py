"""
C1: price the NVMe tier before downloading 45-60 GB.

`docs/SCOPE.md` Option B is a genuinely larger MoE -- 80B+ at Q4_K_M, 45-60 GB,
which does not fit in this machine's 29 GB of RAM plus 12 GB of VRAM. The only
way it runs is a third tier: NVMe -> RAM -> VRAM, with RAM as an LRU cache over a
model that does not fit in it.

The arithmetic that decides this is brutal and worth doing first. A miss that
reaches disk is not like a miss that reaches RAM:

    expert computed on the CPU (measured, M5a)          39 us
    expert fetched from RAM over PCIe (measured, AI2)   57 us
    expert fetched from NVMe                            measured below

If NVMe is in the milliseconds, a single disk miss costs more than an entire
token does today, and the RAM tier has to absorb essentially everything. This
computes what "essentially everything" means numerically.

The NVMe measurement uses **random** reads of exactly one expert's size
(2.92 MiB), because that is the access pattern -- a predictor asking for
scattered expert ids, not a sequential model load. It also writes and reads a
file larger than free RAM, so the page cache cannot serve the reads and flatter
the result; without that this measures memory, not disk.
"""
import json
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "c1_nvme_tier_result.json")
SCRATCH = os.environ.get("SCRATCH", "/home/everett/moe-prefetch/state/nvme_probe.bin")
EXPERT_MIB = 2.92
EXPERT_BYTES = int(EXPERT_MIB * 1024 * 1024)
N_READS = int(os.environ.get("N_READS", "300"))

# measured elsewhere in this project
CPU_EXPERT_US = 38.8            # M5a
RAM_FETCH_US = 57.1             # AI2 PREFETCH_FEASIBILITY
FLOOR_MS = 5.876                # all experts GPU-resident, 30B Q4_K_M
K = 8


def free_ram_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def make_file(path, size_gb):
    if os.path.exists(path) and os.path.getsize(path) >= size_gb * 1e9:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"writing {size_gb:.0f} GB probe file (larger than free RAM, so the page "
          "cache cannot serve the reads)...", file=sys.stderr, flush=True)
    chunk = os.urandom(8 << 20)
    with open(path, "wb") as f:
        written = 0
        while written < size_gb * 1e9:
            f.write(chunk)
            written += len(chunk)
        f.flush()
        os.fsync(f.fileno())


def measure_random_reads(path, n_reads):
    size = os.path.getsize(path)
    rng = random.Random(0)
    fd = os.open(path, os.O_RDONLY)
    try:
        times = []
        for _ in range(n_reads):
            off = rng.randrange(0, max(size - EXPERT_BYTES, 1))
            t0 = time.perf_counter()
            os.pread(fd, EXPERT_BYTES, off)
            times.append((time.perf_counter() - t0) * 1e6)      # us
    finally:
        os.close(fd)
    return times


def main():
    free = free_ram_gb()
    size_gb = max(free * 1.5, 12.0)
    print(f"{free:.1f} GB RAM available -> {size_gb:.0f} GB probe file", file=sys.stderr)
    make_file(SCRATCH, size_gb)
    times = measure_random_reads(SCRATCH, N_READS)
    med = statistics.median(times)
    p90 = sorted(times)[int(0.9 * len(times))]
    gbps = EXPERT_BYTES / (med / 1e6) / 1e9

    print(f"\n=== C1: NVMe AS A THIRD TIER ({N_READS} random {EXPERT_MIB} MiB reads) ===\n")
    print(f"per-expert read from NVMe:  median {med:.0f} us,  p90 {p90:.0f} us "
          f"({gbps:.2f} GB/s effective)")
    print(f"  vs computing it on the CPU:   {CPU_EXPERT_US:.0f} us   "
          f"({med / CPU_EXPERT_US:.0f}x)")
    print(f"  vs fetching it from RAM:      {RAM_FETCH_US:.0f} us   "
          f"({med / RAM_FETCH_US:.0f}x)")

    # What RAM hit rate keeps a token acceptable?
    # A token touches n_layers * K experts. Those not in VRAM come from RAM
    # (cheap) or NVMe (expensive). Assume the VRAM cache performs as measured
    # on 30B: 90.8% expert hit. The remaining 9.2% must be served by RAM or disk.
    vram_hit = 0.908
    print(f"\nWith the VRAM cache performing as measured on 30B ({100 * vram_hit:.1f}% "
          "expert hit),\n{:.1f}% of expert lookups fall through to RAM or NVMe."
          .format(100 * (1 - vram_hit)))

    rows = []
    for n_layers in (48, 62, 80):
        lookups = n_layers * K
        fall = lookups * (1 - vram_hit)
        print(f"\n  a {n_layers}-layer model: {lookups} expert lookups/token, "
              f"{fall:.0f} fall through")
        print("   RAM hit    disk reads/token   added ms/token   tok/s")
        for ram_hit in (0.90, 0.99, 0.999, 1.0):
            disk = fall * (1 - ram_hit)
            add_ms = (disk * med + fall * ram_hit * RAM_FETCH_US) / 1000
            floor = FLOOR_MS * n_layers / 48
            tok = 1000 / (floor + add_ms)
            rows.append({"n_layers": n_layers, "ram_hit": ram_hit,
                         "disk_reads": disk, "added_ms": add_ms, "tok_s": tok})
            print(f"   {100 * ram_hit:6.1f}%   {disk:14.2f}   {add_ms:14.2f}   {tok:6.1f}")

    print("\nRead the table by what it takes to stay useful, not by the best row.")
    ok = [r for r in rows if r["n_layers"] == 80 and r["tok_s"] >= 30]
    need = min((r["ram_hit"] for r in ok), default=None)
    if need is not None:
        print(f"For an 80-layer model to hold even 30 tok/s, the RAM tier must hit "
              f"{100 * need:.1f}% of the fall-through -- i.e. NVMe may serve at most "
              f"{80 * K * (1 - vram_hit) * (1 - need):.2f} experts per token.")
    else:
        print("No RAM hit rate in the table keeps an 80-layer model above 30 tok/s.")

    json.dump({"expert_bytes": EXPERT_BYTES, "median_us": med, "p90_us": p90,
               "effective_gbps": gbps, "cpu_expert_us": CPU_EXPERT_US,
               "ram_fetch_us": RAM_FETCH_US, "vram_hit_assumed": vram_hit,
               "rows": rows}, open(OUT, "w"), indent=1, default=float)
    print(f"\nsaved {OUT}")
    try:
        os.remove(SCRATCH)
        print("removed the probe file", file=sys.stderr)
    except OSError:
        pass


if __name__ == "__main__":
    main()
