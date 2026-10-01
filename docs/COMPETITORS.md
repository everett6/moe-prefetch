# colibri and ds4: what they do, what was worth taking, what was not

Read on 2026-09-30: [JustVugg/colibri](https://github.com/JustVugg/colibri) at
`ce370e8` and [antirez/ds4](https://github.com/antirez/ds4) at `0aaea5a`, plus
the papers they and this project lean on.

**Nothing in this document has been measured on the engine.** It was written
with the GPU in use by other work. The engine changes compile (syntax and
semantics, three translation units) and are off by default; the one set of
numbers below that is new is a trace replay, labelled as such.
`experiments/next_session.sh` runs every measurement in order.

## Where this project stands against them

They are not built for the same job, so the raw numbers are not a ranking:

| | hardware | model | decode |
|---|---|---|---|
| colibri `qwen36` + VRAM tier | RTX 3070, 8 GB | Qwen3.6-35B-A3B int4 | 16.6 tok/s (their Ollama reference on that box: 20.3) |
| ds4, SSD streaming | M5 Max, 128 GB | GLM 5.3 Flash Q4 / DS4 Flash | 11.9 – 19.3 tok/s |
| this project | RTX 5070, 12 GB | Qwen3-30B-A3B Q4_K_M | 113.8 – 126 tok/s (R3, real prompts) |

colibri and ds4 exist to run models that are far larger than memory (125B to
2.8T parameters) off an SSD; a 30B model on a 12 GB card is a side case for
colibri and out of scope for ds4. On that side case this project is several
times faster in absolute terms, on a faster card with half again the VRAM.
What they have that this project did not is a set of cache-policy ideas, and
that is what was worth reading them for.

## What they do

**colibri** treats VRAM, RAM and disk as one hierarchy.

- *Heat-driven placement.* A per-expert use count, halved every 1024 decode
  ticks, decides who holds VRAM. Promotion needs the candidate to beat the
  coldest resident by 25% + 4 (`tier_should_promote`).
- *The same guard on speculation* (`PILOT_EVICT_GUARD`): a prefetched expert may
  evict a resident only if it is historically hotter. Added after they found
  speculative loads thrashing warm demand-loaded experts (#441/#474) -- the same
  failure this project measured as "a correct prefetch still evicts".
- *Router lookahead* (`PILOT`): apply layer L+1's real router to layer L's
  post-attention state. 71.6% recall of the true top-8 on GLM-5.2, against
  41.3% for "same experts as the last token". No trained predictor.
- *Persisted heat* (`HEAT_FILE`, `.coli_usage`): the second run starts with the
  hot set already placed. Their own table: 44% hit cold, 95% warm.
- *Cache-aware routing* (`CACHE_ROUTE`, off by default): keep the true top-J,
  fill the rest from resident experts inside the top-M. Changes output.
- *Issue / take overlap*: resident experts are issued to the GPU as an async
  group, misses run on the CPU at the same time, and the two are joined.
- *Trunk placement by price*: "bytes saved on the memory bus per token, per byte
  of VRAM" decides whether a dense matrix or a cold expert gets the VRAM.

**ds4** is a narrow engine for a few very large models.

- One *global* slot pool across layers, plain LRU by stamp.
- "Protect every hit first: a miss must not evict a later request's hit."
- "Look-ahead is not evidence of reuse": a prefetched slot is published with the
  lowest recency (`slot.used = 1`) and only a real router hit promotes it.
- A profiled *hotlist* compiled into the binary, preloaded at start.

## The literature the three of us share

- [arXiv 2412.00099](https://arxiv.org/abs/2412.00099), *Mixture of
  Cache-Conditional Experts*: three cache-aware routing rules. The best is
  cache-prior re-ranking -- add a bonus to cached experts' logits. Over 50%
  fewer misses for 0.1 – 3% perplexity; swapping the top-ranked expert is what
  hurts, the tail is nearly free.
- [arXiv 2608.18261](https://arxiv.org/abs/2608.18261), a pre-registered
  negative result with a measurement study **on Qwen3-30B**: training routers for
  locality fails the quality gate, but inference-time rerouting within a
  tolerance cuts misses ~80% for 2 – 3% perplexity.
- [SeqMoE, arXiv 2609.12978](https://arxiv.org/abs/2609.12978): sequence-model
  prediction, probabilistic Belady eviction, synchronisation-free orchestration
  for graph capture. Reports 96.97% hit rate and 80% of full-load speed at 45%
  residency. This project sits at 44% residency (56 of 128) and 94 – 96% hit.
- [SpecPrefetch, arXiv 2607.24787](https://arxiv.org/abs/2607.24787): a small
  adapter predicts next-layer experts for transfer only; +20% on a phone.

## What was taken, and what it became here

### 1. An admission gate (from colibri's guard and ds4's probation)

This engine uploaded every miss. P1 had already shown the cost: Belady reaches
a better hit rate with 61% fewer uploads. The fitted cost model gives the rule
without any reference to the other engines -- a miss costs B = 47 us, an upload
C = 142 us, so **an admission has to buy three future hits to break even**, and
"admit on first miss" assumes the cold tail will deliver them.

`LLAMA_MOE_ADMIT=heat` is colibri's rule (decayed use count, 25% + floor over
the victim); `LLAMA_MOE_ADMIT=window` is the 2Q/ARC ghost rule (second miss
within N tokens). Trace replay, LRU eviction held fixed
(`experiments/p8_admission.py`, `artifacts/p8_admission.json`), 56 slots:

| admission | uploads/token | hit | modelled tok/s |
|---|---|---|---|
| every miss (today) | 27.1 | 90.7% | 103.3 |
| second miss within 8 tokens | 13.0 | 88.7% | 124.0 |
| hotter than the victim, half-life 64 | 9.8 | 89.3% | 133.4 |

**Read the upload column, not the tok/s column.** The simulator is not
trace-validated (Phase 0 was never done) and its tok/s is the cost model's
C term multiplied out. Phase 1 already showed that making an upload *shorter*
changes nothing, so whether upload *count* costs throughput is exactly the
open question -- and this gate is the cheapest way to answer it. If uploads
fall by half and throughput does not move, C is wrong at this operating point
and the cost model has to be refitted. That is a result either way.

P2 found frequency-based *eviction* harmful (LFU −8.5 tok/s). This is
admission with LRU eviction unchanged, and the short half-life is what the
replay prefers: at half-life 1024 the same rule loses 14 points of hit rate.

### 2. Probation for predicted uploads (from ds4)

`LLAMA_MOE_SPEC_PROBATION=1`. A predicted expert used to be published as the
most recently used slot, so a wrong guess held its slot for a full LRU cycle.
Now it enters at the bottom of the order, protected for exactly one token. A
wrong guess costs one slot for one token. Only matters with a predictor loaded.

### 3. Warm start from persisted heat (from both)

`LLAMA_MOE_HEAT_FILE=<path>`. Use counts are written every 512 steps and, on
the next start, the hottest experts are placed before the first token. Refused
by name if the file's dimensions are another model's.

### 4. Cache-prior routing, in the graph (from the paper; colibri does a CPU variant)

`LLAMA_MOE_CACHE_PRIOR=<delta>`. **Changes model output; off by default.**
A device tensor holds `delta` for resident experts and 0 otherwise, and is
added to the router's *selection* probabilities before top-k. The gate weights
are still gathered from the unbiased probabilities. Three things differ from
both references:

- It is one `add` on the GPU, inside the graph. colibri reranks on the CPU,
  which here would cost a device round trip per layer.
- The bound is in probability units: a resident expert displaces a
  non-resident one only if the router ranked them within `delta`. That is a
  statement about how much gate mass can move, which a rank window is not.
- The prior is published in the same transfer as the slot tables, so it can
  never name an expert the tables do not.

Quality cannot be read off `llama-perplexity`: it evaluates in batches and the
cache only exists on the single-token graph, so it would score a model the
prior never touches and report zero cost. `experiments/q1_cache_prior_quality.py`
scores human text one token at a time through the decode graph instead.

## What this project added that neither has

- **Batched table publication** (`LLAMA_MOE_BATCH_TABLES=1`). Every eviction and
  every publication was a synchronous 4-byte device copy on the decode thread:
  two per upload, each ending in a stream synchronise. Now a step edits host
  mirrors and writes the changed layers with one synchronise.
- **Byte-priced slot allocation.** `experiments/gguf_preflight.py` found that
  Q4_K_M is not one quant type on this model: 24 of 48 layers keep a Q6_K
  down-projection, so an expert is 2.92 MiB there and 2.53 MiB elsewhere. The
  Phase 2 profile was allocated by slot count and used +28 MiB over uniform --
  measured, and written off as noise. It is exactly that. The byte-priced
  profile uses −8 MiB (`artifacts/p7_slot_allocation.json`).
- **A scheduler split profiler** (`GGML_SCHED_PROFILE=<n>`). The cost model's
  fixed term is 59% of a token and has never been decomposed. This prints, per
  backend, time waiting for inputs against time computing.
- **A GGUF preflight** that says whether the cache will engage on a file at all
  (a fused `gate_up` tensor silently disables it) and how many slots fit.

## What was not taken, and why

- **Router lookahead (PILOT).** It predicts layer L+1 of the *current* token,
  and this engine can only publish an upload at a step boundary, so the expert
  cannot arrive in time to be used. It would need mid-graph table writes.
- **Global slot pool (ds4).** The right generalisation of Phase 2 -- it adapts
  to the workload instead of being fitted to a corpus -- but it means one
  shared tensor per quant type and a rewritten cache. Not something to write
  without being able to run it.
- **Frequency eviction (colibri's LFRU).** Measured harmful here (P2).
- **MTP speculation.** colibri measures −32% around 85% hit rate; this
  project's own draft-model line was closed the same way.
- **SSD streaming, striping, multi-GPU.** Different problem.

## The structural idea left on the table: issue / take overlap

Every cached layer today runs GPU (attention, router) → CPU (missed experts,
GPU idle) → GPU (cached experts). The cached-expert chain does not depend on
the CPU chain, so the two could run at the same time, which is what colibri's
issue/take does natively. In ggml the scheduler synchronises the whole GPU
backend before copying a CPU split's inputs, so reordering the graph is not
enough. The change that would work:

1. order the graph as router → cached chain → CPU chain → add;
2. force a split boundary before the cached chain;
3. in `ggml_backend_sched_compute_splits`, when the split after next is a host
   split whose inputs are all older than the current one, copy those inputs
   *now*, before launching the current split.

The saving per layer is the smaller of the two chain times. It is a scheduler
change under CUDA graph capture and was not written blind. The profiler is
there to say whether it is worth writing: if the CPU split's compute time is
small against its input wait, there is little to overlap.

## A 30 – 40B model to try next

[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (35.95B, 40
layers × 256 experts, 8 routed). llama.cpp arch `qwen35moe`, present in this
checkout. `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`, Q4_K_M is 20.75 GB. Not
downloaded. Two things to check on the file before benchmarking:

- the cache needs separate gate/up/down tensors; a GGUF converted with
  `--fuse-gate-up-exps` disables it silently (`gguf_preflight.py` reports this);
- 256 experts per layer at the same VRAM means a lower resident fraction than
  the 44% this project runs at, so hit rate will start lower.

The slot profiles and predictor weights are fitted to Qwen3-30B and do not
carry over. The admission gate, batched tables and cache prior do:
`MODEL=… NCMOE=40 SLOTS=… experiments/next_session.sh`.
