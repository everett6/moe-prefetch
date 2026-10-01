# moe-prefetch

Running a mixture-of-experts model at **Q4_K_M** — the quality reference — on a
12 GB GPU, at speeds previously reachable only by quantizing down to Q2_K.

**Result: 126.2 tok/s on real coding and decision-making prompts** (n=69, sem
1.5), against 78.7 for the shipped Q4_K_M config on the same machine, and a
110 tok/s bar. That is **1.60×**. Method and full numbers in
[`docs/FINAL_RESULTS.md`](docs/FINAL_RESULTS.md); the real-prompt work in
[`docs/REAL-DATA-RESULTS.md`](docs/REAL-DATA-RESULTS.md).

Earlier versions of this README reported 118.2 ± 1.7 tok/s. That was thirty
repeats of a *single prompt I wrote myself*, so its `±` measured repeatability
rather than anything about workloads. Measured across 23 real prompts the figure
is higher and the spread is three times wider — real prompts differ from each
other far more than one prompt differs from itself.

Qwen3-30B-A3B, RTX 5070 (12 GB, 175 W), Ryzen 9 7950X, 29 GB RAM.

## As of 2026-10-01

The 126.2 above was measured at 72 cache slots a layer. With a desktop session
sharing the card, 72 no longer allocates, so everything measured since is at 56
slots, and is lower for that reason.

| at 56 slots, real prompts, n = 69, paired | tok/s |
|---|---|
| the cache as it was | 112.9 |
| + second-miss admission, byte-priced slots, batched table writes (same output) | **119.3** (+6.41 ± 1.06) |
| + cache-aware routing, δ = 0.01 (output changes; perplexity −0.10% ± 0.42%) | **130.1** (+16.37 ± 1.37 over 113.7) |

`./serve.sh` starts the first; `./serve.sh fast` the second.

**Qwen3.6-35B-A3B** (20.75 GiB at Q4_K_M) runs on the same engine unmodified:
108.0 tok/s with the cache against 84.0 for the best split without it, and
127.0 with cache-aware routing at δ = 0.01 (perplexity −0.38% ± 0.68%).
`MODEL=qwen36 ./serve.sh [fast]`.

The same run corrected this project's cost model, and with it the reasoning in
the next section. An upload costs 26 µs, not 142; the fixed part of a token is
76%, not 59%; and that fixed part is the GPU stopping 48 times a token to hand
a layer to the CPU. The design that follows from it, and the probes showing it
is feasible on this card, are in
[`docs/superpowers/specs/2026-10-01-graph-resident-decode-design.md`](docs/superpowers/specs/2026-10-01-graph-resident-decode-design.md).
Its estimate is about 160 tok/s with unchanged output. It is not built.

---

## What this turned out to be about

> **Superseded in part, 2026-10-01.** The cost model in this section was fitted
> to five points, one of them the cache in its broken state, and that point
> alone set the 142 µs upload cost. Refitted on 39 observations it is
> `ms = 6.79 + 0.039 × missed + 0.026 × uploads`. The 75% precision bar and
> the "−0.4 tok/s for a perfect predictor" both rest on the old upload cost and
> are withdrawn. The conclusion survives for a different reason: the token
> waits on the host, not on expert transfers, so moving experts sooner does not
> shorten it. The section is left as written, as the record.

The project was built to test **asynchronous predictive expert prefetching**: if
the experts a token is about to need were copied into VRAM before it needed them,
the cost of keeping them in host RAM would be hidden behind compute.

The predictor works. It is trained, exported, verified against its C++ loader —
and **switched off**, because it makes the system slower. That is the finding,
and it is arithmetic rather than a modelling failure.

A cost model fitted to five end-to-end measurements spanning 33 to 106 tok/s
(worst residual 4.9%):

```
ms/token = 4.18 + 0.047 × missed_experts + 0.142 × uploads
```

A cache miss costs 47 µs. An expert upload costs 142 µs. A *correct* prefetch
adds no upload — the demand path fetches that expert anyway — so it is worth
+47 µs, while a wrong one burns 142. So:

> **A prefetch pays only above 75% precision on non-resident experts.**

The freshly trained model reaches 61% at the depth the engine can afford. Every
number this project reported for a year was a *recall*; recall is not the
quantity that decides.

And the gap is not the point. Replaying the same traces with a **perfect**
predictor — 100% precision, fetching exactly what the next token will use — is
worth **−0.4 tok/s** at the measured upload cost. Its prefetches duplicate what
demand admission does anyway while spending the same slot budget a beat earlier.
So the limit is not prediction quality; there is no headroom above it to reach.
Halve the cost of an upload and a perfect predictor becomes worth +4.6; quarter
it, +12.4. That is the lever, and it is a memory-bandwidth problem.

## What did make it fast

Fixing the cache it was supposed to ride on. [llama.cpp PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
takes a cache slot out of service when an upload is scheduled and returns it when
the upload is published a step later. At the default insert rate the scheduler
outruns the uploader, the backlog reaches 3,134 jobs, ~65 of 66 slots per layer
sit permanently in flight, and the cache drains to **0.6 resident experts per
layer** — *slower than no cache at all*.

| | shipped | early issue |
|---|---|---|
| cache hit rate | 23.4% | **91.0%** |
| resident experts/layer | 0.6 / 66 | **64.0 / 66** |
| worker backlog | 3,134 jobs | 1 |
| tok/s | 33.1 | **104.0** |

Issuing the copy during the graph, out of slots freed at the previous step
boundary, fixes it without ever mutating an expert table mid-graph. Re-tuning the
cache size on a cache that works takes it to 111.4, and `--no-mmap` to 118.2 —
1.49× the shipped config, 3.3× the cache as PR #27861 ships it.

Five upstream bugs found on the way (see [`docs/UPSTREAM-BUGS.md`](docs/UPSTREAM-BUGS.md)); the two that mattered most: `--moe-expert-cache` silently no-ops because
a dry-run context latches its one-shot init guard ([`patches/0001`](patches/0001-moe-cache-dont-latch-on-dry-run-context.patch)),
and the default insert rate is above the stability threshold.

## Layout

| | |
|---|---|
| `docs/FINAL_RESULTS.md` | numbers, method, limitations |
| `docs/D1-RESULT.md` | how a headline measurement was wrong by 3.4× |
| `docs/COMPETITORS.md` | colibri and ds4 read against this engine: what was taken, what was not, and why |
| `docs/UPSTREAM-BUGS.md` | five bugs found in llama.cpp, four of them silent |
| `PLAN.md` | what remains, and the bar it has to clear |
| `experiments/` | capture, training, RL, benchmarks — each a runnable script |
| `cpp/` | the predictor's C++ loader and its agreement test |
| `patches/` | the llama.cpp changes; `patches/README.md` says which to apply (0006 then 0007) and which are history |
| `artifacts/` | frozen baseline, cost model, manifests, experiment log |
| `tests/` | format, leakage and numerical-agreement tests |

## A note on measurement

The 110-tok/s claim this repo originally made for the cache was wrong by 3.4×.
The harness started a server, waited for `/health`, and measured whatever
answered on the port — but `/health` returns 503 *with a body* while the model
loads and `curl -s` exits 0 on that, so the wait returned instantly and a server
left over from the previous configuration answered the requests.

Every benchmark here now verifies that the port was free, that the process it
measures is the one it launched, and that the cache allocated its ~8.8 GB. It is
the fifth time in this project that something which looked like a finding was an
artefact, and the reason the working rule is: *when something looks blocked or
looks finished, check the block.*
