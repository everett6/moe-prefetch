# Final results

Qwen3-30B-A3B at **Q4_K_M** on one RTX 5070 (12 GB, 175 W cap), Ryzen 9 7950X,
29 GB RAM. Everything below is measured on this machine unless a row says
*modelled*, and the two are never mixed.

## Headline

| | tok/s (mean ± sd, n=15) |
|---|---|
| AI2's shipped Q4_K_M config (`-ncmoe 22`, no cache) | 85.1 ± 0.97 |
| **This work** (`-ncmoe 48 --moe-expert-cache 72`, early issue) | **116.8 ± 1.60** |
| PR #27861's cache as it ships, same cache size | 35.5 ± 0.61 |

**1.37× over the baseline, 3.29× over the cache as shipped, and the 110 tok/s
bar is cleared by 6.8.** Measured by interleaving the three configurations across
three rounds of five requests, so drift affects all of them equally.

The bar matters because it is what `ud-q3_k_xl` already does at Q4_K_M-equal
accuracy. Below it, quantizing down was the better answer; above it, this is
something that did not previously exist on this hardware.

## Where the speed came from — and where it did not

**Not from prediction.** The project was built to test asynchronous *predictive*
prefetching. Prefetching is switched off in the winning configuration, because
it is measurably worse, and the reason is arithmetic rather than model quality.

A cost model fitted to five end-to-end measurements spanning 33 to 106 tok/s
(`artifacts/cost_model.json`, worst residual 4.9%):

```
ms/token = 4.18 + 0.047 × missed_experts + 0.142 × uploads
```

A cache miss costs **47 µs**; an expert upload costs **142 µs**. A *correct*
prefetch adds no upload — the demand path fetches that expert anyway — so it is
worth +47 µs, while an incorrect one is a wasted 142 µs. Hence

> **break-even precision = C/(B+C) = 75%**

on non-resident experts. The fresh model reaches **61.0%** at the depth the
engine can afford and 71.8% at depth 8. It is not close enough, and the gap is
not the kind that a better model closes.

**It came from fixing the cache.** PR #27861 takes a slot out of service when an
upload is scheduled and returns it only when the upload is published a step
later. At the default `--moe-expert-cache-inserts 2` the scheduler outruns the
uploader: the backlog reaches 3,134 jobs, ~65 of 66 slots per layer sit
permanently in flight, and the cache converges to **0.6 resident experts per
layer** with a 23% hit rate — worse than no cache at all.

Issuing the copy from the observation callback during the graph, out of a pool of
slots freed at the previous step boundary, fixes it:

| | shipped | early issue |
|---|---|---|
| cache hit rate | 23.4% | **91.0%** |
| resident experts/layer | 0.6 / 66 | **64.0 / 66** |
| worker backlog | 3,134 jobs | 1 |
| tok/s | 33.1 | **104.0** |

and re-tuning the cache size on a cache that works takes it to 116.8 at 72 slots.
(78 slots fails to allocate; 84 *silently* falls back to no cache, which is its
own bug.)

## The reset, and the fresh model

The draft corpus and predictor were deleted deliberately
(`artifacts/reset_record.json`): 1.70 GB and 424,029 samples. The draft
predictors are archived out of `models/`, isolated, with nothing permitted to
initialise from them — they exist only as the `DRAFT_OLD` comparison arm.

**Dataset `v3-20260919-fresh`:** 28.20 GB, 6,832,244 rows, 52 shards, 540 prompts
across 16 registers, every shard checksummed and validated. Sequence length,
context size and batch size vary across six session regimes (90 prompts each);
the draft held all three fixed. Every row carries how many of its experts would
miss a 66- and a 32-slot LRU and its overlap with the previous token, so rows
where a prefetcher could actually be paid are separable from rows where the cache
already had everything. All 128 experts appear, imbalance 2.6×.

Splits are by **register**, assigned from a fixed seed before capture and written
into the manifest: train 11, validation 2, **locked test 3** (history, medical,
reasoning). Selection used validation only; the test registers were read once.

**Architecture search under a latency budget.** A layer is 198 µs at the
operating point, so a predictor running once per layer gets ~10 µs — roughly
20,000 multiply-accumulates. That rules out the obvious answers before training:

| config | val recall@8 | params | MAC | budget |
|---|---|---|---|---|
| **linearctx-lr1e2** | **74.26%** | 3.15 M | 4,096 | ok |
| linear | 70.95% | 1.58 M | 2,048 | ok |
| linear-mse | 69.49% | 1.58 M | 2,048 | ok |
| lowrank-r96 | 68.23% | 625 K | 13,824 | ok |
| resmlp-h128 | 67.82% | 452 K | 26,624 | **over** |
| lowrank-r32 | 58.59% | 212 K | 4,608 | ok |

Depth loses to inputs. Every shared-trunk and nonlinear variant is beaten by a
per-layer linear model while costing 3–6× the arithmetic. The two extra context
sets are worth +3.3 points for 2,048 extra MAC, and BCE beats the draft's
squared-error objective by +1.5.

**Locked test, three ways** (`artifacts/final_eval_test.json`):

| arm | recall@8 | recall@8 hard | warm precision@2 | warm precision@8 |
|---|---|---|---|---|
| `DRAFT_OLD` | 64.64% | 65.38% | 55.0% | 67.5% |
| `FRESH_SUPERVISED` | **75.99%** | **76.53%** | **61.0%** | **71.8%** |
| `FRESH_RL` | — | — | — | — (converged to depth 0) |

The fresh model beats the draft by **+11.35 recall points** on registers neither
ever saw. It still does not clear 75%, so the RL stage converged on issuing
nothing, which is the correct answer to the problem as posed.

## The RL stage, and what it is actually for

State: the model's top-8 scores, how many are already resident, free pool slots,
layer index, running hit rate. Action: prefetch depth ∈ {0,1,2,3}. Reward: +47 µs
per correct prefetch, −142 µs per incorrect one, in milliseconds of token time.
REINFORCE with a value baseline, against fixed-depth controls.

It converged to depth 0. The useful output is the sweep:

| upload cost | best fixed depth | modelled tok/s |
|---|---|---|
| 141.6 µs (**measured**) | 0 | 88.0 |
| 70.8 µs | 0 | 107.3 |
| **35.4 µs** | **1** | 120.8 |
| 14.2 µs | 2 | 133.9 |
| 0 | 3 | 146.2 |

**Prefetching starts to pay below ~35 µs per upload, four times cheaper than
measured.** That is the number to engineer against, and it is why the next
section exists.

## The optimisation that is still on the table

`ggml_backend_cuda_buffer_set_tensor` does `cudaMemcpyAsync` from **pageable**
memory followed by a full `cudaStreamSynchronize`, and the cache calls it three
times per expert. Measured directly (`artifacts/upload_cost.json`):

| path | idle | with 8 memory-bound CPU threads |
|---|---|---|
| pageable, sync per tensor — **what ggml does** | 188.8 µs | **297.4 µs** |
| pageable, sync per expert | 135.0 µs | — |
| pinned, sync per expert | 86.6 µs | **94.7 µs** |

Pageable transfers collapse to 10.3 GB/s when the CPU is streaming expert weights
out of the same DIMMs; pinned holds 32.3 GB/s. The demand path alone performs
23.6 uploads per token, so a pinned, batched upload path is worth roughly
1.2 ms/token — about **+15 tok/s** — without any prediction at all. It would not
reach the 35 µs that makes prefetching pay, but it is the largest remaining win
and it is not speculative.

## Correctness

**Byte-identical greedy output: FAILS, for the shipped cache as much as for
anything here.** The control passes — cache-off twice is byte-identical — so
decoding is deterministic and the cache is genuinely changing the output. The
cause is structural: PR #27861 splits one `mul_mat_id` into a device cache chain
and a host chain and sums them, so floating-point addition happens in a different
order. The project's premise that prefetching "changes only where weights live"
is false for this implementation.

Graded accuracy is unaffected: **22/24 with the cache off and 22/24 in the final
configuration**, the same two misses, both grader artifacts rather than model
errors (`artifacts/quality_*.json`).

## Reproducing

```bash
python3 experiments/capture_corpus.py          # 28 GB, resumable, ~2 h
python3 experiments/build_index.py             # -> data/index-v3.npz
python3 experiments/train_deep.py --epochs 40  # architecture sweep, validation only
python3 experiments/e2_precision.py --ckpt artifacts/ckpt/FRESH_SUPERVISED-linearctx-lr1e2.pt
python3 experiments/rl_policy.py  --ckpt artifacts/ckpt/FRESH_SUPERVISED-linearctx-lr1e2.pt
python3 experiments/e3_final_eval.py --fresh artifacts/ckpt/FRESH_SUPERVISED-linearctx-lr1e2.pt --split test
python3 experiments/export_fresh.py && python3 cpp/verify_fresh.py
./experiments/e4_config_sweep.sh               # end-to-end throughput
./experiments/d4_identity.sh                   # correctness
python3 -m pytest tests/ -q
```

llama.cpp changes are in `patches/`, against PR #27861 at `bccbacd`.

## Limitations

- **One model, one machine.** Every constant here — 72 slots, 142 µs, the 75%
  bar — is specific to a 30B-A3B at Q4_K_M on a 12 GB card over PCIe 4.0.
- **Byte-identity is unreachable** with this cache design, so the strict
  criterion is failed rather than met. Graded accuracy stands in for it.
- **The environment is optimistic.** It issues 15.5 uploads/token where the
  engine measures 23.6, so its absolute throughput is high; it is used for
  ranking policies, never quoted as a measurement.
- **PR #27861 is a draft** and can be rebased or abandoned. The patches are
  against one commit.
- **The predictor is not deployed.** It is trained, exported, verified against
  the C++ loader, and switched off, because measurement says it costs more than
  it returns. `LLAMA_MOE_PREDICTOR` enables it for anyone who wants to re-test
  that on different hardware.
