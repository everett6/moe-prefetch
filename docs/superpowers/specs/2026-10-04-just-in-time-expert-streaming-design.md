# Just-in-time expert streaming: send the expert before the GPU asks for it

**Status: approved 2026-10-04. Steps 1 and 3 (and step 2's timestamps) are
built as patch 0010, not yet measured; see `docs/STREAMING-IMPLEMENTATION.md`.** Numbers are labelled *measured* (the engine), *probe* (a
standalone program), *replay* (traces through a model of the engine) or
*estimate*. Machine and model as in
`2026-10-01-graph-resident-decode-design.md`: RTX 5070 12 GB at 175 W, Ryzen
9 7950X, DDR5, PCIe 5.0 x16, Qwen3-30B-A3B Q4_K_M, 48 MoE layers of 128
experts, 8 routed, about 2.7 MiB an expert.

## In one paragraph

The GPU does not wait because the memory bus is full; it waits because each
layer asks for its missing experts at the last moment, and a missing expert
then takes a detour through the CPU. The bytes a token needs from host RAM
take about a seventh of the bus's time over the token, but they are
requested in bursts at exactly the point the GPU needs them. This design
predicts, one and two layers early, which experts each layer will route to,
using that layer's own router applied to the current hidden state; sends the
ones not already in VRAM over the copy engine in deadline order; and flips
each one into the GPU's table the moment its copy lands, with the flip
queued on the copy stream itself. On this model the rule "send what the
next layer's router would pick" catches 75% of the misses a layer ahead with
68% of copies used (*replay*), and a timing replay with the measured costs
removes about 0.6 ms of a 6.8 ms token (*estimate*: 147 to about 160 tok/s,
same output). Removing the sidecar's fixed cost per miss is worth about as
much again and is cheaper; together they reach about 168 tok/s, against
the card's 185 for this model.

## 1. What this is for

**Said:** design an architecture that predictively sends data to the GPU
before it needs it, so it arrives on time when the GPU reads it from VRAM,
to bypass the memory bandwidth limit; research it.

**Taken as given:** it builds on what exists and is measured: the CPU
sidecar (patch 0009, exact, about +31 tok/s in its one clean round), the
fused CPU op, the cache and its admission, the in-token publication path.
It must keep output exact in the sense the engine already is (bit-identical
for a given expert placement). No new model training unless the training-free
predictor proves inadequate.

**Success:** a measured, paired gain over the sidecar of at least +6 tok/s
with zero channel errors, and the mechanism visible in counters: misses
converted to hits on time, copies used, copies late.

## 2. What "bypass the memory bandwidth" can and cannot mean here

| link | bandwidth | what crosses it per token |
|---|---|---|
| host DRAM (CPU reading experts) | 61 GB/s measured, flat beyond 8 threads | every missed expert, 2.7 MiB each |
| PCIe 5.0 copy engine | 45.7 GB/s measured (2.78 MiB in 61 µs) | every uploaded expert |
| GPU VRAM | 672 GB/s (card spec) | every resident expert, about 3 µs each |

A copy reads the same DIMMs a CPU miss does, so prefetching does not reduce
the bytes taken from host RAM; wrong guesses add to them. What it changes is
**when** they move and **who waits**. At 21.4 misses a token (*replay*, 56
slots) a token needs about 57 MiB from host RAM, about 1 ms of the bus in a
6.8 ms token. The bus is mostly idle; the GPU still stalls because the bytes
are asked for in bursts (layers 0, 2, 43 to 47 miss most, figure in
`artifacts/p10_lookahead_probe.json`) and each missing layer pays the CPU
detour: the sidecar's serve log measured **120 µs of CPU per layer that
missed** (S3 run, 1.7 misses each), of which the cached-expert work hides 45.
An expert that is already in VRAM costs the GPU about 3 µs. So the lever is
to move the same bytes earlier, on the copy engine, while the bus is idle,
so that the layer finds them resident. That is the sense in which the design
gets around host bandwidth: the critical path stops touching it.

What actually removes bytes is listed for completeness and is not this
design: more residency (VRAM is full), cache-aware routing (δ = 0.01,
measured, changes output), and batching several tokens through one load
(speculative drafts; spec 2026-10-01 step 5).

## 3. What the literature does

| work | predictor | horizon | reported accuracy | what carries over |
|---|---|---|---|---|
| Fate (2502.12224) | next layer's gate applied to this layer's gate input | 1 layer | 97% with confidence expansion; 77–79% at native budget | **the predictor; no training** |
| SpecPrefetch (2607.24787) | small layer-specific low-rank head | 1 layer | ExpertRecall@8 85.5–89.4% on Qwen3-VL-30B-A3B (Fate 81.2–85.0%) | a trained head is worth +4 points; budget must be bounded by the overlap window (throughput peaks at 8) |
| ProMoE (2410.22134) | learned predictor, "stride" prefetch several layers ahead | multi-layer | 79.6–89.0% R@8 (SpecPrefetch's table) | look further ahead to buy transfer time |
| ST-MoE (2606.15453) | cross-layer co-activation table + last token's set | 1 layer | 85% | temporal signal adds little once a cache holds last token's experts |
| PROBE (2602.00509) | frozen clone of the target router + small MLP | 1 layer | n/a | the target router is the right prior |
| PreScope (2509.23638) | prediction-driven, cross-layer scheduling | 1 layer | n/a | schedule across layers, not per layer; weigh prefetch against on-demand cost |
| SPICE (2608.21240) | draft model, confidence-aware lookahead | adaptive | n/a | CPU executes misses asynchronously, as the sidecar does |
| SeqMoE (2609.12978) | sequence model; GPU loads its own misses | 1 layer | 97% hit at 45% residency | never synchronise with the host |

Three things follow. The router of the target layer, fed an earlier hidden
state, is the strongest cheap predictor (residual streams change slowly).
Every system that gains bounds the number of prefetches by the transfer
window; over-fetching loses. And none of them measures the quantity that
matters on a cached system: how well the predictor finds the experts the
cache would **miss**, which is the hard tail. P10 measures that.

## 4. Probe P10: predicting misses on this model

`cpp/moe-route-dump.cpp` records, during greedy decode, each layer's
residual stream, router input and routed experts. `experiments/p10_lookahead_probe.py`
replays every layer through an LRU cache with the measured slot profile and
scores predictors made d layers early. Six real prompts from the benchmark
set (three code, three decision), 320 tokens each, first 64 discarded for
warm-up: 1,536 decode tokens, 21.4 misses a token. Run on the CPU only
(*replay*). The router applied to the true input reproduces the routed set
99.99% of the time.

Predictor: layer T's router applied to rmsnorm(residual at layer T−d) with
T's norm weight ("renorm-ahead"; using the earlier layer's normed input
instead is a point worse). Candidates are only experts not already resident.

| rule | d = 1 | d = 2 | d = 3 | d = 4 | sent / token (d = 1) |
|---|---|---|---|---|---|
| non-resident in the router's top 6 | 52.6% / 82.9% | 48.1 / 69.0 | 44.9 / 58.4 | 42.6 / 50.8 | 13.1 |
| **non-resident in the router's top 8** | **75.1% / 68.0%** | **66.9 / 55.0** | 61.3 / 46.1 | 57.3 / 39.9 | 22.9 |
| non-resident in the router's top 10 | 86.7% / 48.1% | 78.1 / 40.0 | 72.1 / 34.3 | 67.4 / 30.0 | 37.3 |
| probability ≥ 0.02 | 72.7% / 55.7% | 66.7 / 48.2 | 62.6 / 42.6 | 58.9 / 37.5 | 27.0 |
| last token's experts at that layer | 1–8% recall, under 1% precision | | | | |
| token embedding (layers 0–3 only) | 26% / 32% at one per layer, falling fast | | | | |

(recall of misses / precision of what is sent)

So: **send what the target router would select, minus what is resident.**
Recall falls slowly with distance, by about 5 points a layer. The previous
token's choices are useless for misses by construction: the cache already
holds them. For comparison, the trained predictor this project shipped
reached 61% precision; this one needs no training and is 68–83% precise at
higher recall.

## 5. Timing: can the copy land in time?

A prediction for layer T made at layer T−d is available when layer T−d's FFN
starts, and the copy must finish before layer T's table lookup. With 103 µs a
layer (58 of attention and router, 45 of cached experts) that is about
103·d µs; a copy takes 60 µs on one copy engine. One layer ahead buys room
for about 1.7 copies, two layers ahead 3.4.

Replay of the six prompts with that timeline: one copy engine, earliest
deadline first, a copy that cannot land before its deadline is dropped; a
layer still missing m experts stalls the GPU by max(0, 40 + 5 + 47·m − 45) µs
(the 40 µs fixed part calibrated to the sidecar's measured 120 µs per missing
layer):

| what is sent | GPU stall, ms / token | copies / token | misses served from VRAM | copies used |
|---|---|---|---|---|
| nothing (sidecar today) | 1.01 | 0 | 0 | — |
| top 8, d = 1 | 0.55 | 14.9 | 45.8% | 65.9% |
| top 8, d = 2 | 0.45 | 22.0 | 55.5% | 53.8% |
| **top 8, d = 2 then refreshed at d = 1** | **0.38** | 25.6 | **62.3%** | 52.2% |
| top 8, d = 3, 2, 1 | 0.37 | 31.2 | 62.9% | 43.2% |
| top 6, d = 2 then 1 | 0.50 | 15.7 | 50.1% | 68.4% |

With the fixed 40 µs per missing layer removed (section 7): 0.45 ms with no
prefetch, **0.15** with the top-8 two-horizon schedule
(`artifacts/p10_lookahead_probe_fixed0.json`).

What the replay leaves out, all of which make it optimistic: copies slow the
CPU's remaining misses by up to 22% when they overlap (probe x2); wrong
copies occupy reserve slots and can displace a useful expert at the next
token boundary; layer time is held at 103 µs although stalls stretch it
(which helps the copies, the one pessimistic omission); the engine's own
admission rule is 2-in-8, not admit-on-first-miss.

## 6. Design

```
GPU, layer L (one CUDA graph per token, as with the sidecar)
  attention, router L, slot lookup L
  MOE_POST L  ── posts L's ids/slots (as now)
              └─ LOOKAHEAD: for T = L+1, L+2:
                   s_T = W_T · rmsnorm(residual_L) · g_T      (router T, 1 MiB F32 each)
                   cand_T = top-8(s_T) where dev_table_T[e] == dummy
                   writes cand_T (≤ 8 ids each) into the same ring record
  cached experts L ......................... copy engine streams experts for L+1, L+2
  MOE_JOIN L                                  each copy ends with its table flip
host serve thread
  record L: observe routing; compute L's misses (as now)
            hand cand_{L+1}, cand_{L+2} to the planner
planner (same thread): dedupe against resident / in flight / queued,
            deadline_T from measured GPU timestamps, EDF queue, drop if late
mover (one thread, one copy stream): copies chunks; after the last chunk of
            an expert, enqueues the 4-byte flip dummy → slot on the same stream
```

### 6.1 Predictor, in the graph

`MOE_POST` gains the lookahead: inputs the layer's residual stream, the norm
weights and router weights of layers L+1 and L+2, and their device tables.
It writes up to 8 candidates per target layer, already filtered against the
device table, into the ring record. Two 128 × 2048 matvecs read 2 MiB from
VRAM: about 3 µs a layer, 0.15 ms a token (*estimate*), paid whether or not
anything is sent. It reads the model's own weights and changes nothing the
model computes. Layers 0 and 1 cannot be predicted from earlier layers of the
same token; the embedding is too weak (section 4) and they stay on the cache
and the sidecar.

### 6.2 Planner, on the host

Runs in the sidecar's serve thread, which is already spinning on the ring.
For each candidate: skip if resident, in flight or queued; compute the
deadline from GPU timestamps (`MOE_POST` writes `%globaltimer` into the
record; the planner keeps a running estimate of the time per layer); enqueue
earliest deadline first. A job that cannot start early enough to land before
its deadline is dropped, not sent late. A d = 1 candidate for a layer that
already has a d = 2 job is merged. The number sent per layer is capped by
the reserve pool (6.4).

### 6.3 Mover, one thread, one copy stream

Copies each expert's three slices with `cudaMemcpyAsync` on a dedicated
stream from the pinned host weights, in chunks of at most 512 KiB so a
later-deadline expert cannot hold the engine against an urgent one. After an
expert's last chunk it enqueues, on the same stream, a 4-byte copy of the
slot number into that layer's device table. **The flip lands exactly when the
expert has landed**, with no host round trip in between; the host learns of
it from an event and updates its bookkeeping afterwards. Arbitration with the
sidecar: while the serve thread is computing a miss, chunks for deadlines more
than one layer away wait (they would slow that miss by up to 22%).

### 6.4 Slots

Each layer keeps R free slots at every token boundary (the eviction at the
boundary refills them), so a mid-token arrival never needs an eviction while
a graph runs. R is per layer, sized to its share of misses; R = 2 on average
costs 48 × 2 × 2.7 MiB = 260 MiB of cache. A prefetched expert enters on
probation (exists: `LLAMA_MOE_SPEC_PROBATION`): if its layer does not route
to it this token it is the first to go at the boundary.

### 6.5 Correctness

Unchanged from the sidecar and the in-token path already built: the GPU
decides what missed from its own table and posts the slot ids; the host
computes exactly those rows; a table word flips only from the dummy to a slot
whose copy is complete, into a slot no table referenced. Output is the exact
model's, with the last-bit dependence on placement the cache already has.
Tested the same way: logits bit-identical under a replayed placement map.

### 6.6 What it replaces

The trained next-layer predictor and the worker-side in-token publication
(`LLAMA_MOE_PREDICT_INTOKEN`, built, being measured as S5) stay as the
fallback; this replaces both the prediction (router lookahead instead of the
trained model) and the publication (stream-ordered flip instead of a
synchronous host write after the copy).

## 7. The cheaper lever beside it

The sidecar spends about 40 µs per missing layer that is not expert compute:
waking an OpenMP team and running a one-node ggml graph each time. A
persistent pinned team of 8 threads that spins between misses, calling
`ggml_cpu_moe_rows` directly, removes most of it: 1.01 → 0.45 ms of stall
(*estimate*, same replay). It needs no prediction and should be built first.

## 8. Expected value and build order

| step | what | gate | *estimate* |
|---|---|---|---|
| 0 | done: P10 (predictability and timing replay) | — | — |
| 1 | persistent CPU team in the sidecar | ≥ +4 tok/s paired over sidecar; S2-style logit test | 147 → about 159 |
| 2 | P11 on the GPU: timestamps in `MOE_POST`, measured lead time per layer and copy latency under load | lead ≥ 90 µs at d = 1 | — |
| 3 | lookahead in `MOE_POST`, planner, mover, stream-ordered flip, reserve pool | logits bit-identical under replay; ≥ +6 tok/s paired over step 1; counters: ≥ 50% of misses served from VRAM, ≥ 45% of copies used | about 168 |
| 4 | tune: K (6/8), horizons, reserve size, arbitration | each change paired | — |

Step 1 gets its own implementation plan; steps 2 and 3 share one, written
after step 1 has a measured number. The ceiling for this model on this card
with no misses at all is about 185 tok/s. Two thirds of the remaining gap is in reach of steps 1 and 3 by the
replay; the replay is optimistic (section 5), so the gates are set below it.

## 9. Risks

- **Contention.** 26 copies a token add about 69 MiB of host reads to the 22
  MiB the remaining misses still need; if copies overlapping misses slow
  them more than the 22% measured, the gain shrinks. The arbiter and the
  counters exist for this.
- **Lead time.** If the GPU reaches layer T faster than 103 µs a layer
  (it does when nothing misses), d = 1 copies arrive late more often. P11
  measures it; d = 2 carries most of the value anyway.
- **Reserve slots** come out of the cache and raise the miss rate a little;
  R is a measured trade, not a constant.
- **A copy stream under the sidecar's spinning join** is what probe x1
  section 6 measured (61 µs per 2.78 MiB, same as idle); a driver change
  could alter it, and the self-test does not cover the copy path.
- **Replay on six prompts**, 1,536 tokens, LRU with admit-on-first-miss
  rather than the engine's 2-in-8 rule.

## 10. Sources

- [Fate: accurate expert predictions via cross-layer gate (arXiv 2502.12224)](https://arxiv.org/html/2502.12224v1)
- [SpecPrefetch: parameter-efficient expert prefetching (arXiv 2607.24787)](https://arxiv.org/pdf/2607.24787)
- [ProMoE: proactive caching for MoE serving (arXiv 2410.22134)](https://www.alphaxiv.org/abs/2410.22134.md)
- [ST-MoE: spatio-temporal expert prefetching (arXiv 2606.15453)](https://arxiv.org/html/2606.15453v1)
- [PROBE: real-time predictive prefetching (arXiv 2602.00509)](https://arxiv.org/pdf/2602.00509)
- [PreScope / LayerScope: prediction-driven cross-layer scheduling (arXiv 2509.23638)](https://arxiv.org/html/2509.23638v2)
- [SPICE: speculative prefetching with low-rank surrogates (arXiv 2608.21240)](https://arxiv.org/html/2608.21240v1)
- [SeqMoE (arXiv 2609.12978)](https://arxiv.org/abs/2609.12978)
- [SP-MoE: speculative decoding and prefetching (arXiv 2510.10302)](https://arxiv.org/html/2510.10302v2)
- [HOBBIT: mixed-precision expert offloading (arXiv 2411.01433)](https://arxiv.org/pdf/2411.01433)
