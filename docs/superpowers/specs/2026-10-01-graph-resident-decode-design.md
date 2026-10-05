# Graph-resident decode: a token that does not wait for the host

**Status: design, for review. 2026-10-01. Nothing in sections 6 to 8 is
implemented.** Sections 2 to 4 are measurements taken on this machine on
2026-09-30 and 2026-10-01; every number says whether it is *measured*, a
*probe* (a standalone program, not the engine), a *replay* (trace simulation)
or an *estimate*. The engine in this repo has been wrong about its own numbers
often enough that the labels are not decoration.

Machine: RTX 5070 12 GB capped at 175 W, Ryzen 9 7950X, 30 GB DDR5, PCIe 5.0.
Model: Qwen3-30B-A3B Q4_K_M unless a row says otherwise.

## In one paragraph

The cache made the *uploads* asynchronous and left the *token* synchronous:
48 times per token the GPU stops, the layer is handed to the CPU, and the GPU
waits. Measured, that hand-over costs 1.7 ms of an 8.8 ms token before a single
missed expert is computed, and the GPU sits idle for a third of every token.
This design keeps the whole token on the GPU as one captured graph and turns
the CPU into a sidecar that answers misses through shared memory while the GPU
carries on with the experts it already has. A probe shows the exchange works
on this card and driver, costs 4.4 µs instead of about 35, and loses no data in
20,000 exchanges per case. The estimate is **about 160 tok/s with unchanged
output (range 150 to 170), against 113 today**, and about 170 with the
cache-aware routing option that measured as free of quality cost. Until it is
built that is an estimate. What is measured and usable today is in section 2:
119.3 tok/s exact, 130.1 with that routing option.

## 1. What this is for

What was asked: research other ways to raise token speed, and design
"something like asynchronous MoE prefetching" from the ground up, on a 30 to
40B model as well.

What I took that to mean, so it can be corrected:

- The goal is decode tokens per second on this 12 GB card, clearly ahead of the
  other engines that run these models (colibri, ds4, the llama.cpp cache forks).
- Output stays exactly Q4_K_M by default. Anything that changes output is
  opt-in and ships with a quality number.
- "Asynchronous prefetching" is the means, not the end. If measurement says the
  token waits on something other than expert transfers, the design follows the
  measurement.

Not in scope: prefill speed (the cache is decode-only and stays so), models
larger than RAM, multi-GPU, anything needing a retrained router.

## 2. What the 2026-09-30 run established

All on real prompts (23 prompts, 3 rounds, n = 69 per arm), arms interleaved,
differences paired by prompt. `artifacts/r3_real_throughput_*.json`.

| switch | tok/s | paired difference | verdict |
|---|---|---|---|
| batched table writes | 112.3 → 113.0 | +0.66 ± 0.25 (t 2.7) | keep |
| admission gate, heat rule | 113.9 → 110.9 | −3.00 ± 1.13 (t −2.7) | **rejected** |
| admission gate, second miss within 8 tokens | 113.9 → 117.8 | +3.88 ± 0.96 (t 4.0) | keep |
| slot profile by count (+28 MiB VRAM) | 113.9 → 116.9 | +2.95 ± 0.63 (t 4.7) | superseded |
| slot profile by bytes (−8 MiB VRAM) | 113.9 → 116.5 | +2.56 ± 0.66 (t 3.9) | keep |
| **the three kept, stacked** | **112.9 → 119.3** | **+6.41 ± 1.06 (t 6.1)** | new exact default |
| cache-aware routing, δ = 0.005 | 113.7 → 123.6 | +9.87 ± 0.84 | changes output |
| **cache-aware routing, δ = 0.01** | **113.7 → 130.1** | **+16.37 ± 1.37 (t 12.0)** | changes output |
| cache-aware routing, δ = 0.02 | 113.7 → 137.0 | +23.35 ± 1.88 | changes output |

Quality of the routing option, scored one token at a time through the decode
graph (24 texts, 3,072 tokens, paired by text; `artifacts/q1_cache_prior_quality.json`):

| δ | perplexity change | misses per token |
|---|---|---|
| 0.005 | +0.07% ± 0.26% | 33.3 → 23.9 |
| **0.01** | **−0.10% ± 0.42% (no measurable change)** | **33.3 → 18.3** |
| 0.02 | +0.92% ± 0.54% (t 1.7, borderline) | 33.3 → 12.1 |
| 0.05 | +3.9% ± 0.9% (t 4.3, real cost) | 33.3 → 5.2 |

Two things about how these were obtained. The stack was written as "heat gate
+ profile + tables" before any of it had run; the heat gate then measured
negative, so the stack was changed to the 2-in-8 gate *before* the stack stage
started, and the script says so. And the first profiling stage printed nothing
(it asked for a report every 512 tokens and the model stopped at 467); it was
rerun with a fixed script afterwards.

## 3. Where a token goes

`experiments/y1_token_anatomy.sh`, plain 56-slot cache, last 128 tokens of a
1,024-token generation (*measured*, scheduler split profiler):

| per token | µs | what it is |
|---|---|---|
| GPU working, host waiting | 5,053 | 103 µs a layer |
| CPU expert chain, GPU idle | 1,940 | 14 µs a layer whether or not anything missed, plus 47 µs a miss |
| copying the CPU's answer back | 502 | 10 µs a layer |
| relaunching the GPU | 195 | 4 µs a layer |
| after the last layer (output head, sampling) | ~375 | |

The same profiler on two other configurations separates the pieces:

- No cache, all experts on the CPU: the GPU's share of a layer is **58 µs**
  (attention and router) and the CPU's is 343 µs (eight experts). So the
  cached-expert chain is 103 − 58 = **45 µs of GPU work per layer**.
- Cache with δ = 0.02 routing, about 3 misses a token: the CPU chain still
  costs 17 µs a layer. That is its **fixed cost, about 14 µs**, paid 49 times
  a token for doing nothing.

Adding up what is *not* GPU work and *not* computing a missed expert: two
copies down (about 0.3 ms), the CPU chain's fixed cost (0.7), the copy back
(0.5), the relaunch (0.2). **1.7 ms per token, 35 µs per layer, is the price
of handing the layer over.** `nvidia-smi` agrees from the other side: the GPU
is busy 67% of the time during decode.

## 4. Diagnosis

### 4.1 The cost model this repo has quoted was wrong about uploads

`ms = 4.18 + 0.047 × missed + 0.142 × uploads` was fitted to five points, one
of which was the cache in its broken state (3,134 queued uploads). That point
alone fixed the upload term. The admission A/B is the first measurement to
move uploads and misses in opposite directions on a working cache. Refit on 39
(config, round) observations (`experiments/f1_refit_cost_model.py`):

```
ms/token = 6.79 + 0.039 × missed + 0.026 × uploads        rms 0.06 ms
           (6.46–6.93)  (34–47 µs)      (21–31 µs)        95%, configs resampled
```

The old model's error on the same observations is 0.94 ms rms. At the plain
cache: fixed 6.79 ms (76%), misses 1.47 (17%), uploads 0.64 (7%). Three
things in this repo's README follow from the old numbers and are withdrawn:
"an upload costs 142 µs", "a prefetch pays only above 75% precision", and
"an admission must buy three future hits". An upload costs about a quarter of
that, which is why the stingy heat gate lost and the mild one won.

**No cache policy can pass 147 tok/s in this architecture**, because that is
what 6.79 ms is. The fixed term is the target.

### 4.2 A miss is a memory-bandwidth event

`experiments/x2_cpu_expert_cost.cpp` computes one expert with the engine's own
CPU kernels on weights from the real file, 1.36 GiB of them so that every visit
is cold (*probe*):

| | µs per expert | |
|---|---|---|
| 1 thread | 110 | 26 GB/s |
| 4 threads | 53 | |
| **8 threads** | **47** | **61 GB/s, and flat from here** |
| 16 threads | 49 | |
| 32 threads | 54 | |
| 16 threads, 8 experts at once | 331 (41 each) | matches the 343 µs measured in the engine |
| 16 threads, expert already in the CPU cache | **14.5** | 3.4× faster |
| 8 threads, while another reader streams 30 GB/s | 66 against 54 | an upload in flight costs a miss about 22% |

So the 47 µs is 2.7 MiB crossing the memory bus at 60 GB/s. More threads do
not help. An upload reads the same DIMMs, which is where the upload term comes
from: it does not block the token, it slows the misses beside it.

### 4.3 Asynchronous in the wrong place

Put together: the GPU does about 4.8 ms of work per token and the token takes
8.8. The rest is the GPU waiting for a host that is either doing bookkeeping
(1.7 ms) or streaming a missed expert off the DIMMs (1.5 to 1.9 ms) while the
GPU's own experts for that same layer wait their turn. Nothing in that list is
an expert transfer. Prefetching experts sooner cannot fix a wait that is not
for experts, which is the measured history of this project's predictor.

## 5. Four ways forward

| | idea | evidence | predicted | risk |
|---|---|---|---|---|
| A | Tune the cache that exists | **measured** | 119.3 exact, 130.1 with δ = 0.01. Ceiling 147. | none; it is done |
| B | Overlap inside ggml's scheduler: launch the cached chain, then run the CPU chain, then join | moe-autopilot measured its attempt at −3.4%; the hand-over cost stays | about 130 | medium, for a third of the gain |
| **C** | **One GPU graph per token; the CPU serves misses through shared memory** | **probe passed (section 6.2)** | **about 160 exact (150–170)** | engineering; one undocumented driver behaviour |
| D | No CPU compute; the GPU pulls a missed expert from host memory itself (SeqMoE) | probe: 81 µs per expert, and the GPU cannot hide its own stall | 121 at today's hit rate; needs 97% hits to reach 160 | needs a predictor this project never got to pay |

D is what the newest paper does ([SeqMoE](https://arxiv.org/abs/2609.12978)),
and it is right about the principle: never synchronise with the host. But it
answers a miss by transferring the expert, and on this machine a kernel pulling
2.65 MiB from host memory takes 81 µs (the copy engine, 61 to 80), where
computing the expert on the CPU takes 47 and can run *while the GPU works*. C
takes SeqMoE's rule and Fiddler's observation that at batch 1 the CPU should
compute rather than the bus transfer. The llama.cpp cache RFC went the other
way, CPU in control with the GPU as the helper, after reporting a
GPU-in-control experiment that ran about twice as slow as stock even at a
97 to 99% hit rate and blaming its per-layer synchronisation points. C is
GPU-in-control with no synchronisation points, which is the case that thread
did not have. I did not find this combination in the repos or the papers read
for this; that is a statement about my reading, not a priority claim.

**Recommendation: ship A now, build C.** B is skipped: it is a scheduler
rewrite for a result C makes obsolete.

## 6. Design C

### 6.1 Principle

During a token the decode thread makes no CUDA call and the GPU never leaves
its graph. Everything the host contributes arrives through pinned memory that
both sides can read and write.

### 6.2 The probe

`experiments/x1_mapped_coprocessor.cu`, RTX 5070, driver 595, CUDA 13.2. One
"token" is 48 layers, each with 40 µs of GPU work standing in for the cached
chain, captured as a single CUDA graph (*probe*):

| per layer | µs |
|---|---|
| no host involvement | 41.3 |
| today's protocol: synchronise, two copies down, one up | 58.9 to 63.6 |
| **shared-memory exchange, every layer missing** | **45.7 (1 row back) to 47.4 (8 rows)** |
| exchange kernels present, no layer missing | 43.7 |
| host takes 10, 20, 30, 40 µs to answer | 45.7 each: **fully hidden behind the GPU's work** |
| host takes 60 / 100 µs | 64.9 / 105.1: the GPU waits only for the excess |
| a 2.78 MiB upload on a second stream while the GPU kernel waits | 61 µs, same as on an idle GPU |

Zero corrupted payloads and zero timeouts in 20,160 exchanges per case. A GPU
kernel polls a host word in 0.57 µs. Exporting through captured memcpy nodes
instead of kernel stores works too but costs 7 µs more per direction.

What "undocumented" means here: NVIDIA's documentation says accesses to mapped
pinned memory must be synchronised through streams or events. This design
relies on a running kernel seeing a host write, and the host seeing a kernel's
write, without either. It is observed on this card and driver, and the design
treats it as something to verify at every start rather than assume
(section 6.6).

### 6.3 The graph

Per cached layer today: router → *[leave the GPU]* CPU chain → *[return]*
cached chain → add. In C, all on the GPU:

```
router, top-8, slot lookup                (as now)
post     write the layer input and the missed ids to host memory, if any missed
cached chain over the resident experts    (as now)    <- the CPU works during this
join     if this layer posted: wait for the host's answer, read the missed rows
add, weights, sum                         (as now)
```

Two new ggml operations, CUDA only: `MOE_POST` and `MOE_JOIN`. `MOE_POST`
passes the layer input through, so the cached chain depends on it and the
order is forced. A layer with no miss posts only its eight ids (the host needs
them for recency) and waits for nothing. With no host-memory tensor left in
the graph the scheduler produces one GPU split, and the existing CUDA graph
capture covers the whole token with one launch.

### 6.4 The channel

Pinned, mapped, allocated once:

- a ring of 64-byte records, one per posted layer: sequence number, layer,
  eight ids, eight slot ids;
- per layer, a request buffer (the layer input, 8 KiB) and a response buffer
  (eight rows, 64 KiB);
- three words: `head` (GPU writes), `resp[layer]` (host writes), `err` (GPU
  writes on a timed-out wait).

Ordering, which is the whole correctness argument for the channel:

- The payload is written by one kernel and `head` by the next. A stream runs
  kernels in order, so the payload's stores are issued before the store that
  announces them. The probe checks every element of every payload.
- The host writes the response rows and then `resp`. x86 commits stores in
  program order and device reads are coherent with the CPU cache, so a kernel
  that sees `resp` sees the rows.
- Sequence numbers come from a counter in device memory, not from kernel
  arguments, because a captured graph replays the same arguments every token.
- The device decides what missed (its own table lookup) and tells the host.
  The host never infers it, so there is no window in which the two disagree.

### 6.5 The sidecar

- **Serving.** While a decode is in flight the decode thread itself spins on
  `head` instead of blocking in a synchronise call. For each record it updates
  recency, queues uploads for what missed, and if anything missed computes the
  rows and writes `resp`. The last cached layer's record ends the loop.
- **Computing.** A new exported function in ggml's CPU backend,
  `ggml_cpu_moe_rows`, does what the CPU chain's five graph nodes do now, for
  the missed experts only: quantise the input, one `vec_dot` per row for up and
  gate, the same vectorised SwiGLU, quantise, one `vec_dot` per row for down.
  Same kernels, same order, per row independent of threading. Eight threads
  (section 4.2).
- **Uploading.** The existing worker and its second stream, unchanged, with
  one rule added: an upload copies in chunks of at most 512 KiB and only
  starts a chunk while no miss is being served. Misses own the memory bus.
- **Publishing.** Unchanged: tables change only between tokens, in one batched
  write. Publishing mid-token would be legal here but makes output depend on
  thread timing, so it is left out.

### 6.6 Exactness and failure

The GPU chain is untouched and the host rows come from the same kernels in the
same order, so **for a given placement of experts the output is bit-for-bit
the current cache's output**.

*Corrected 2026-10-04.* This section first said the output would equal the
current engine's, full stop, and made "identical tokens at temperature 0" the
acceptance test. That cannot hold, for the current engine either. An expert
computed on the GPU and the same expert computed on the CPU differ in the last
bits, so the output depends on which experts are resident at each token, and
that depends on which uploads have finished when the token boundary comes.
Anything that changes timing changes placement. Measured on step 1 (patch
0008, `docs/FUSED-CPU-IMPLEMENTATION.md`): with placement frozen, the original
and fused CPU chains gave identical output on all four prompts that had
diverged live, and the original chain alone gave different output on three of
them frozen against live.

So the acceptance test is: **record the live placement map of a run, replay
it, and require every logit at every step to be bit-identical**, teacher-forced
on the recorded tokens (`experiments/x3_logits_replay.py`). The same run
replayed through the unchanged path is the control; if the control differs,
the test says nothing. Live-run token differences are recorded but are not a
correctness signal.

A wait gives up after a bounded number of polls (about one second), sets
`err`, and the join uses zeros; the decode call then returns an error. Wrong
output is never returned silently. At start the engine runs 10,000 exchanges
of the probe's integrity test on the real channel and falls back to the
current path, loudly, if one fails. `LLAMA_MOE_SIDECAR=0` selects the current
path for A/B and for bisecting.

### 6.7 What it should be worth

*Estimate*, from the measurements above:

| per token | ms |
|---|---|
| GPU work (what the host waits for today, less its two copies) | 4.8 |
| output head and sampling | 0.4 |
| exchange: 2.4 µs for a layer that hits, about 5 for one that misses | 0.2 |
| misses the GPU's 45 µs window does not hide | 0.9 |
| **total** | **6.3 → about 160 tok/s** |

The 0.9 comes from a trace replay (`experiments/p9_stall_model.py`): misses are
burstier than chance. 61% of (token, layer) pairs miss nothing, 21% miss one
expert, 18% miss two or more, and layers 0 to 3 hold most of the bursts. One
cold miss (47 µs plus the exchange) just overruns a 45 µs window; each further
miss in the same layer costs its full 47. Overlap hides 42 to 47% of today's
miss time, not all of it. With δ = 0.01 routing the same sum gives about
5.8 ms, 170 tok/s. With no misses at all it is 5.4 ms, 185 tok/s: that is this
card's limit for this model at Q4, and nothing in this design goes past it.

What would falsify the estimate: under 145 tok/s on the benchmark with the
sidecar on, which is the gate on step 2 in section 10. Then the model in
section 3 is wrong somewhere and the next step is the profiler, not tuning.

### 6.8 What changes in the code

| file | change |
|---|---|
| `ggml/include/ggml.h`, `ggml/src/ggml.c` | two op enums, names, constructors |
| `ggml/src/ggml-cuda/` | three small kernels (export, wait, join), the two ops, an allocator for mapped host memory exposed through `get_proc_address` |
| `ggml/src/ggml-cpu/` | `ggml_cpu_moe_rows`, exported |
| `src/llama-graph.cpp` | build post / chain / join instead of the CPU chain when the sidecar is on |
| `src/llama-moecache.cpp` | the channel, the serve loop, the upload arbiter; routing observation moves from the CPU op callback to the serve loop |
| `src/llama-context.cpp` | serve until the token is done, then run the cache step |

The current path stays, selectable, until the new one has beaten it on the
benchmark.

### 6.9 Tests

- `ggml_cpu_moe_rows` against the existing CPU graph on random inputs: rows
  identical. CPU only, runs anywhere.
- The probe as a regression test of the channel on this driver.
- End to end: bit-identical logits under a replayed placement map, against
  the unchanged path under the same map (section 6.6); the graded 24-problem quality set; 100,000 tokens with zero
  channel errors.
- The watchdog: stop the serve loop mid-token and confirm the decode call
  errors within about a second instead of hanging.
- Speed: the same paired benchmark as section 2, sidecar off against on.

## 7. Prefetching, redesigned

Once one miss per layer is nearly free, the quantity to minimise is no longer
misses. It is **time the GPU spends waiting**, which is misses beyond the first
in a layer, and anything that slows the memory bus while a miss is being
served. That changes what each part is for.

- **Admission** stays the 2-in-8 gate, and gets re-measured under C: its
  measured gain today comes from fewer uploads competing with misses, and the
  arbiter removes that competition directly.
- **The arbiter** (section 6.5) is the part of prefetching that was missing:
  not what to move, but when the bus is free to move it.
- **Slot allocation** gets a new objective. The current profile minimises
  total misses. The same greedy, priced in bytes, should minimise
  `Σ max(0, x + m·τ − K)` per layer, which puts slots where misses arrive in
  bursts (layers 0 to 3) rather than where they are merely frequent.
- **Warming the CPU cache** is a kind of prefetch this project never tried: a
  missed expert that is already in the CPU cache costs 14.5 µs, not 47, so
  three of them fit in the window instead of one. It takes no VRAM slot and
  evicts nothing, which is exactly the damage that sank VRAM prefetching here.
  The bound is the whole 0.9 ms. But the cheap predictor for it is poor:
  "missed in this layer within the last 1 to 8 tokens" covers 6 to 17% of
  misses at 23 to 6% precision (*replay*), and every wrong guess is 2.7 MiB of
  bus traffic. So it is gated: built only if the trained predictor, replayed
  against real traces, is worth 2 tok/s or more.
- **Uploading predicted experts into VRAM** stays off. Every measurement in
  this repo says so, and C removes the last reason to want it.

The honest summary: the asynchrony that was missing was the token's, not the
prefetcher's. What is left for prediction is small and has to earn its place.

## 8. Other ways to more tokens per second

Measured here, available now:

| | gain | note |
|---|---|---|
| exact stack (2-in-8 gate, byte-priced slots, batched tables) | 112.9 → 119.3 | no output change |
| cache-aware routing δ = 0.01 | 113.7 → 130.1 | output changes; no measurable perplexity change |

Not yet measured, cheap, exact:

| | why it should help | size |
|---|---|---|
| one fused CPU op for the host chain | the chain costs 14 µs a layer doing nothing (section 3); one node instead of five. It is also the function C needs | 0.5 ms, about +6 tok/s |
| upload arbiter in the current engine | section 4.2 | 0.2 to 0.6 ms |
| 8 CPU threads, not 16 | section 4.2; harness mode `WITH_OMP` is written | small |

Later, larger:

- **Small batches through the cache** (n ≤ 8 tokens). The cache only exists
  for single-token graphs, so n-gram drafting, which this stack measured at
  +8% on code edits, switches it off. Upstream's PR has since added this.
- **Stall-aware slots, CPU-cache warming**: section 7.

Looked at and not taken:

| | why not |
|---|---|
| heat admission gate (colibri's rule) | measured −3.0 tok/s |
| device-initiated expert loads (SeqMoE) | 81 µs a miss against 47 overlapped (section 5) |
| low-rank surrogates for missed experts (SPICE) | 3 to 3.5 points of GSM8K accuracy, needs shared experts and calibration |
| low-precision copies of missed experts (MoE-APEX, HOBBIT) | VRAM here does not hold a second copy of 6,144 experts next to a useful cache |
| draft model or MTP speculation | this stack measured a 0.5B draft at 0.96×; colibri reports −32% |
| unified-memory paging by the driver | a page fault costs far more than a CPU-computed miss |
| a fused GPU kernel for the whole expert layer (MonoMoE) | llama.cpp already fuses up, gate and activation; the chain is 45 µs of mostly memory traffic |
| the 250 W power limit | about 5%, and this card falls off the bus there |

## 9. Qwen3.6-35B-A3B on this card

The 30 to 40B model asked for. `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf` (bartowski,
20.75 GiB, SHA-256 verified): 41 expert layers of 256 experts, 8 routed plus a
shared one, 1.69 to 1.95 MiB an expert (3.19 in the last layer, which is
Q8_0). `experiments/gguf_preflight.py` says every layer is cache-eligible, and
it ran on the engine unmodified.

Real prompts, 3 rounds, paired by prompt (*measured*,
`artifacts/r3_real_throughput_model-qwen36-35b.json`). 22 of the 23 prompts:
one makes this model emit end-of-text at once and is dropped from every arm.

| configuration | VRAM | tok/s | paired against no cache |
|---|---|---|---|
| no cache, `-ncmoe 26` | 9,054 MiB | 84.0 | — |
| **cache, 96 slots a layer** | 9,526 MiB | **108.0** | **+23.93 ± 1.04 (t 23, 22 of 22)** |
| cache + 2-in-8 gate + batched tables | 9,526 MiB | 108.0 | +23.95 ± 1.01 |

Hit rate 82.9% with 37.5% of experts resident: 55 misses and 33 uploads a
token. The gate that helped Qwen3-30B does nothing here (it cuts uploads to 13
and adds 4 misses, and the two cancel). The slot profiles are fitted to the
other model and were not used.

The cache arm holds 472 MiB more than the baseline, so the baseline was also
given more than that: `-ncmoe 25` puts one more layer's experts on the card
and takes 10,226 MiB, 700 above the cache arm. On the single profiling prompt
it reaches 85.9 tok/s against 83.5 at `-ncmoe 26` and 112.3 for the cache
(`artifacts/anatomy_qwen36-*.txt`). The gap is the cache, not the memory.

Cache-aware routing on this model (*measured*, same 22 prompts, 2 rounds,
`artifacts/r3_real_throughput_prior-qwen36-35b.json`; quality on the same 24
human texts as section 2, `artifacts/q1_cache_prior_quality-qwen36-35b.json`):

| δ | tok/s | paired against δ = 0 | hit | misses a token | perplexity against δ = 0 |
|---|---|---|---|---|---|
| 0 | 108.0 | — | 82.9% | 55.2 | 7.065 |
| **0.01** | **127.0** | **+18.91 ± 1.00 (22 of 22)** | 95.1% | 15.9 | −0.38% ± 0.68% (t −0.55) |
| 0.02 | 130.4 | +22.32 ± 1.09 (22 of 22) | 96.9% | 10.0 | +2.03% ± 0.98% (t 2.04) |

δ = 0.01 is usable here on the same terms as on Qwen3-30B: no change that
this test can see, and the test would not see one under about 1%. δ = 0.02 is
not: twice the cost it has on the 30B model for 3 tok/s more.

Other engines on this same model, for scale and not as a ranking, since none
is on this hardware: colibri reports 16.6 tok/s on an 8 GB RTX 3070;
moe-autopilot 128.3 on an RTX 5090 (its own baseline 116 to 118); the
llama.cpp RFC branch 100.2 and another fork 88.9 on RTX 3090-class cards.

The profiler shows the same anatomy as the 30B model with larger numbers per
layer (131.6 µs of GPU work, since each layer also has a shared expert):
41 hand-overs a token instead of 48, misses of about 32 µs instead of 47.
*Estimate* for design C on this model, same arithmetic as section 6.7: 135 to
145 tok/s against 108, exact. It is wider than the 30B estimate because the
miss clustering it depends on was replayed on the 30B model only.

## 10. Build order, each step gated

| step | what | gate to continue |
|---|---|---|
| 0 | **done**: the measurements in sections 2 to 4, both probes | — |
| 1 | `ggml_cpu_moe_rows`, used as one fused CPU op in the *current* scheduler | rows identical to the CPU graph; logits identical under replayed placement; ≥ +4 tok/s paired |
| 1b | upload arbiter in the current engine | ≥ +1.5 tok/s paired with admit-all; no loss with 2-in-8 |
| 2 | `MOE_POST` / `MOE_JOIN`, the channel, the serve loop, behind `LLAMA_MOE_SIDECAR` | logits identical to step 1 under replayed placement; zero channel errors in 100,000 tokens; **≥ 145 tok/s** paired |
| 3 | stall-aware slot profile | ≥ +2 tok/s paired at equal or lower VRAM |
| 4 | CPU-cache warming | only if a replay of the trained predictor says ≥ +2 tok/s |
| 5 | batches up to 8 tokens through the exchange | n-gram drafting composes with the cache |

Step 1 is worth doing even if step 2 were to fail: it is exact, it needs no
GPU-side change, and it is the function step 2 calls.

The first implementation plan covers steps 1, 1b and 2. Steps 3 to 5 each get
their own, written after step 2 has a measured number.

## 11. Risks

- **The driver behaviour in 6.2 is observed, not promised.** Mitigation: the
  start-up self-test, per-exchange sequence checks, the watchdog, the fallback
  switch. A driver update that breaks it costs the speed-up, not correctness.
- **The estimate rests on a 512-token replay** for how misses cluster, and on
  one prompt for the 45 µs window. Either could move the 0.9 ms by a third.
  The gate in step 2 is set below the estimate for that reason.
- **A descheduled serve thread stalls the GPU.** Mean throughput will not
  show it; the benchmark should report the slowest 1% of tokens as well.
- **Two graph ops and three kernels in a fork of a draft PR.** PR #27861 has
  moved since `bccbacd` (small batches, an admission gate of its own). The
  patch set should be rebased before step 2 rather than after.
- **This is one model on one machine**, again.

## 12. Sources

Read for this design, beyond what `docs/COMPETITORS.md` already lists:

- [SeqMoE: predictive and graph-compatible MoE offloading (arXiv 2609.12978)](https://arxiv.org/abs/2609.12978)
- [SPICE: speculative prefetching with low-rank expert surrogates (arXiv 2608.21240)](https://arxiv.org/html/2608.21240v1)
- [Reproducible evaluation of MoE expert caching (arXiv 2608.07911)](https://arxiv.org/html/2608.07911)
- [Mixture of cache-conditional experts (arXiv 2412.00099)](https://arxiv.org/html/2412.00099v2)
- [MonoMoE: a fused mega-kernel for quantised MoE decoding (arXiv 2609.04244)](https://arxiv.org/pdf/2609.04244)
- [llama.cpp RFC: MoE expert cache with hybrid hit/miss execution (discussion #24528)](https://github.com/ggml-org/llama.cpp/discussions/24528)
- [llama.cpp PR #27861, current state](https://github.com/ggml-org/llama.cpp/pull/27861)
- [JigSawPT/moe-autopilot](https://github.com/JigSawPT/moe-autopilot) and [mbx10br/llama-moe-lru-pinned-vision](https://github.com/mbx10br/llama-moe-lru-pinned-vision)
- [CUDA decode optimisation: batched MoE and kernel fusion](https://zolotukhin.ai/blog/2026-06-11-cuda-llm-decode-optimization-batched-moe-kernel-fusion/)
- [Qwen3.6-35B-A3B architecture overview](https://huggingface.co/blog/EXDai/qwen36-35b-a3b-architecture-overview)
