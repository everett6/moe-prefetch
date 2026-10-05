# CPU sidecar: asynchronous expert compute from host RAM (design C, step 2)

Plan: `docs/superpowers/plans/2026-10-04-cpu-sidecar.md`. Spec: section 6 of
`docs/superpowers/specs/2026-10-01-graph-resident-decode-design.md`.

`LLAMA_MOE_SIDECAR=1` (with `LLAMA_MOE_FUSED_CPU=1`). Each cached layer's
host-side expert split is replaced by two GPU ops. `MOE_POST` writes the
layer's routed ids, and its input if an expert missed, to mapped pinned host
memory and announces it; the GPU then runs the cached experts. A host thread
picks up the announcement, observes the routing, and for a miss computes the
missed rows with the step-1 fused op (`GGML_OP_MOE_ROWS`, 8 threads) straight
from the host-resident weights. `MOE_JOIN`, after the cached chain, waits for
the host's answer only if the layer missed, and adds the rows. A decode token
no longer leaves the GPU: the CPU works on misses while the GPU works on hits.

Uploads, admission and publication at token boundaries are unchanged.

## Safety

- At start, 10,000 exchanges through a test channel; any failure disables the
  sidecar loudly and the fused path runs instead.
- A GPU wait gives up after 1 s, sets an error word and uses zero rows; the
  decode call then returns -3. Wrong output is not returned silently.
- `cpp/test-moe-sidecar.cpp`: all-hit, mixed (1,000 runs under CUDA graphs),
  all-miss, a host 2 ms late, a dead host, and the self-test. Passes.

## Tests, pre-registered 2026-10-04 (written and pushed before S2 to S4 ran)

Each runs once, in order; a later test runs only if the earlier ones pass.

| # | test | gate |
|---|---|---|
| S1 | `cpp/test-moe-sidecar.cpp` | PASS (already run: passed) |
| S2 | `experiments/x4_sidecar_logits.py`: the four prompts, 200 tokens; RECORD (fused, live), REPLAY-F (fused, replayed map), REPLAY-S (sidecar, replayed map) | both replays zero unequal logits and routing identical to RECORD |
| S3 | `experiments/x4_sidecar_bench.py`: fused vs sidecar, live cache, 3 rounds × 23 prompts × 200 tokens, alternating order | **sidecar arm mean ≥ 145 tok/s** (spec step 2) and no sidecar error in any server log |
| S4 | `experiments/x4_sidecar_soak.py`: one sidecar server, ≥ 100,000 decode tokens | zero channel errors |

If S3 falls short of 145, the estimate in spec 6.7 is wrong somewhere: the
next step is the split profiler on the sidecar build, not tuning. The paired
gain over the fused path is reported either way. Conditions: 175 W power
limit; no other compute process on the GPU (the harness refuses otherwise).

## Predictive in-token prefetch, pre-registered 2026-10-04 (before it ran)

`LLAMA_MOE_PREDICT_INTOKEN=1`, with `LLAMA_MOE_PREDICTOR` and
`LLAMA_MOE_EARLY_ISSUE=1`. While layer L is observed, the trained predictor
(`models/predictor-real.bin`: the experts layer L+1 used last token and the
experts layer L uses now) names the top 3 experts layer L+1 will want. Their
copies start at once, and the upload worker makes each one usable by the GPU
the moment its copy lands, instead of at the end of the token. That timing
limit is why prediction measured as a loss in this project before: a guess
for layer L+1 could not be used until the next token.

What makes a mid-token table write safe: the predicted expert goes into a slot
that is in no table, and one aligned 4-byte table word flips from the dummy
slot to it. A graph that read the table before the flip sees a miss, one that
reads it after sees a hit. Since this commit the host computes exactly the
experts the GPU's table marked missed (the GPU posts its slot ids), not what
the host's own table says, so a flip can never leave a row uncomputed. The
server log counts predictions used in time, routed but too late, and unused.

| # | test | gate |
|---|---|---|
| S5 | `experiments/x5_predict_bench.py`: arm 0 sidecar + early issue, arm 1 the same + predictor + in-token publication + probation; 3 rounds × 23 prompts × 200 tokens | **≥ +2 tok/s paired** (spec section 7: prediction has to earn its place) and no sidecar error |

Runs after S2 passes, whatever S3 and S4 give. If it fails, the in-time /
late / unused counters say whether the predictor or the timing is at fault.
