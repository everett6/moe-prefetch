# D1 result: the cache alone hits 112.8 tok/s, against a predicted 113.0

Measured 2026-09-18 on llama.cpp PR #27861 (`moe-expert-cache`, +645/−0, draft),
built with the userspace CUDA toolchain, Q4_K_M, `-c 4096`, server-reported
timings, 175 W cap.

| config | tok/s |
|---|---|
| `-ncmoe 22`, no cache (today's baseline) | 78.8 |
| `-ncmoe 48`, no cache (all experts host-resident) | 45.6 |
| `-ncmoe 48`, `--moe-expert-cache 32` | 37.3 |
| `-ncmoe 48`, `--moe-expert-cache 48` | 80.6 |
| **`-ncmoe 48`, `--moe-expert-cache 66`** | **112.8** |
| `-ncmoe 48`, cache 48, `--moe-expert-cache-inserts 8` | 32.6 |
| `-ncmoe 48`, cache 48, `--moe-expert-cache-inserts 16` | 32.6 |

**`m3_simulate_speedup.py` predicted 113.0 tok/s for "LRU cache alone" at
capacity 66. The real implementation delivers 112.8.**

That agreement is the point of D1. The whole 133 tok/s projection rests on a
cost model built from measured constants (`FIXED` 10.5 µs, `PER_EXPERT` 38.8 µs,
hit rates replayed from real traces). The larger and more load-bearing half of
that projection has now been confirmed against an independent implementation,
to within 0.2%.

Two secondary findings:

- **The `--moe-expert-cache-inserts` throttle hurts badly here.** Default 2 is
  best; 8 and 16 both collapse to 32.6 tok/s, *below* the no-cache baseline.
- **A small cache is worse than none** — 32 slots/layer gives 37.3 against 45.6
  with the cache off. It thrashes. 66 is the operating point, as the simulation
  said, and it is also the largest that fits.

## The bug that had to be fixed first

Out of the box the flag did nothing at all. `--moe-expert-cache 66` parsed, was
plumbed correctly through `common_params` → `llama_context_params` → 
`llama_moe_cache_init`, and then silently no-oped: 45.6 tok/s with the cache
"on", no log line, no effect.

The cause is a one-shot global:

```c
void llama_moe_cache_init(const llama_model & model, int32_t n_slots, ...) {
    if (g_init_done) return;
    if (n_slots <= 0) { g_init_done = true; return; }   // <-- latches
```

llama.cpp constructs a **dry-run context for memory estimation before the real
one**, and that context carries default params — `n_moe_cache_slots = 0`. It
latched `g_init_done`, so the real context returned at the first line and the
cache never initialised. The branch logs nothing, so there was no symptom beyond
a flag that did not work.

The fix is to not latch on that path
([`patches/0001-moe-cache-dont-latch-on-dry-run-context.patch`](../patches/0001-moe-cache-dont-latch-on-dry-run-context.patch)):
a later context with real params can then initialise. The other two `g_init_done`
sites (allocation failure, successful init) are correct and untouched.

**Worth reporting upstream.** Anyone enabling this flag on a build that does
memory estimation first would measure "the cache does nothing" and conclude the
approach does not work — the same conclusion, from the same silent failure, that
this project nearly reached twice for other reasons.
