# D1: what PR #27861's expert cache actually does — and the 112.8 that wasn't

**This document supersedes an earlier version that reported 112.8 tok/s for the
shipped cache and called it a 0.2% match to a predicted 113.0. That number was a
measurement error.** The original text is kept at
[`D1-RESULT-superseded.md`](D1-RESULT-superseded.md) rather than deleted, because
how it happened is the most useful part.

## The error

The D1 harness started `llama-server`, waited for `/health`, and measured
whatever answered on port 8099. Three things were wrong with that:

- **`/health` returns HTTP 503 with a body while the model is still loading**,
  and `curl -s` exits 0 on a 503. So the wait returned immediately.
- **Nothing checked that the server being measured was the server just
  launched.** A process that fails to bind the port exits; the port stays held
  by the previous configuration's server, and curl is answered by that one.
- **Nothing checked the cache had allocated.** libllama's `LLAMA_LOG_INFO` does
  not reach `llama-server`'s stderr, so the "MoE expert cache enabled" line is
  invisible and its absence proves nothing either way.

Together those let a reading from one configuration be attributed to another.
AI2 hit exactly this failure — an orphaned `llama-server` holding a port — and
fixed it there with a pre-flight port check; the lesson did not travel to this
repo until it had already produced a headline number.

The harness now verifies all three ([`experiments/d3c_bench.sh`](../experiments/d3c_bench.sh)):
port free before launch, the measured PID is the launched PID, and the process
appears in `nvidia-smi` holding the ~8.8 GB the cache allocates.

## What the cache actually does

Qwen3-30B-A3B Q4_K_M, `-ncmoe 48 -c 4096`, 175 W cap, greedy, `n_predict=200`,
one warm-up discarded, PR #27861 at `bccbacd` plus only the dry-run latch fix:

| config | tok/s | cache hit | resident/layer | worker backlog |
|---|---|---|---|---|
| no cache | 44.9 45.3 44.4 | — | — | — |
| `--moe-expert-cache 66` (default inserts 2) | 36.3 32.9 32.7 | 23% | **0.6 / 66** | **3,134 jobs** |

**The shipped cache is worse than no cache at all**, and the reason is not
policy but arithmetic. A slot is taken out of service the moment an upload is
scheduled and only returns when the upload is published a step later. At 2
inserts per layer per step the scheduler outruns the uploader, the queue grows
without bound, ~65 of 66 slots per layer sit permanently in flight, and the
cache converges to empty. Every expert then misses, which produces more
scheduling, which deepens the backlog. It is a death spiral with a positive
feedback term.

The insert rate is the control, and it is bistable:

| `--moe-expert-cache-inserts` | tok/s | hit | resident/layer | backlog |
|---|---|---|---|---|
| **1** | **104.3 107.0** | 84.9% | **65.8 / 66** | 10 |
| 2 (the default) | 33.4 33.1 | 15.7% | 3.4 / 66 | 3,001 |
| 4 | 31.4 32.0 | 3.1% | 0.3 / 66 | 3,149 |

At 1 the cache fills and holds; at 2 it collapses. Nothing in between was
tested because there is nothing in between — the parameter is an integer.

**This is an upstream bug worth reporting.** The default value of
`--moe-expert-cache-inserts` is above the stability threshold on this hardware,
and the failure is silent: the flag works, the memory is allocated, and
throughput drops by two thirds. The earlier D1 text read the 32.6 tok/s at
`inserts 8` and `16` as "the throttle hurts", which was the right observation
about the wrong mechanism — those rows were the same collapse, already underway
at the default.

## What still holds from the original D1

The dry-run latch bug is real and the patch stands
([`patches/0001`](../patches/0001-moe-cache-dont-latch-on-dry-run-context.patch)).
`--moe-expert-cache` parses, plumbs through `common_params` →
`llama_context_params` → `llama_moe_cache_init`, and then silently no-ops,
because `g_init_done` is latched by llama.cpp's dry-run memory-estimation
context, which carries default params (`n_moe_cache_slots = 0`). Without that
fix none of the measurements above are even reachable.

## What does not hold

The claim that an independent implementation confirmed
`m3_simulate_speedup.py`'s cost model to within 0.2%. It did not: the 112.8 was
never measured. The cost model has not been validated end to end, and the
133 tok/s projection that rests on it should be read as a projection.
