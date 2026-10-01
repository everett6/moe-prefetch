# Verification: a real self-retraining predictor and a real async prefetcher

Both existed as claims before this check; this is the evidence that they are
not just wiring that never fires.

## Asynchronous MoE prefetcher

Not new -- the early-issue path plus the worker-thread pool in
`llama-moecache.cpp` (`free_slots`, `slot_in_flight`, per-worker `todo`/`done`
queues) already was the asynchronous prefetcher. What was missing was
confirmation it is actually live and actually draining under real load rather
than silently falling back to synchronous demand fetches.

Live server, real prompt, decode load (`LLAMA_MOE_STATS=1`):

```
moe-cache: steps=2048 hit=96.0% resident=52.1/56 scheduled=31701 copied=31701 published=31701 todo=0 done=0
```

`scheduled == copied == published` and `todo=0, done=0` at a stats checkpoint
means every upload the predictor issued this run was picked up by a worker,
copied, and published with none stuck in the queue -- the pipeline is keeping
up with demand, not just present in the code. 96.0% hit rate at 56 slots is
consistent with (slightly above) prior measurements at this capacity.

## Self-retraining token predictor

`moe_predictor::update()` (`src/moe-predictor.h`/`.cpp`, new this phase):
online logistic-regression update against the true routing outcome, applied
to the ~32 rows and bias the prediction actually scored, gated by
`LLAMA_MOE_ONLINE_LR`. `moe_predictor::save()` writes the updated weights back
out in MOEP v3 format, checkpointed every 200k updates (temp file + rename,
so a crash mid-write can't corrupt the live weights) and on clean shutdown via
`LLAMA_MOE_PREDICTOR_SAVE`.

"Self-retraining" is a claim about behaviour, not just the presence of an
`update()` function, so it was checked with the model actually changing under
its own online updates within a single run, starting from the same frozen
`predictor-real.bin` used elsewhere in this repo:

```
lr=0.02
online updates=20000   recall@8 first20k=0.496  since=0.000   (-0.496)
online updates=40000   recall@8 first20k=0.496  since=0.643   (+0.147)
online updates=60000   recall@8 first20k=0.496  since=0.657   (+0.160)
online updates=80000   recall@8 first20k=0.496  since=0.660   (+0.164)
online updates=100000  recall@8 first20k=0.496  since=0.667   (+0.171)
```

`first20k` is recall@8 measured once, on the first 20k updates, with the
weights the model started with -- a fixed reference point, not something that
moves. `since` is recall@8 on the most recent window, with whatever weights
are live *right now*. It climbs from 49.6% to 66.7% by 100k updates.

**What that number is and is not.** The verification run sent the SAME prompt
five times at temperature 0, so the model produced the same token sequence
five times and the predictor was scored on routing it had already been
updated on. The rise shows that the update path works end to end -- features
are captured, the gradient moves the weights, the moved weights score
differently, and the checkpoint is written. It does NOT show that online
updates make the predictor better at routing it has not seen: on a repeated
sequence, memorising it is enough. That needs held-out prompts, each sent
once, and has not been run. (The `since=0.000` on the first line is the
reporting window being empty, not a recall of zero.)

Also confirmed live in the same run: `slot profile: 4 layers overridden`
(non-uniform slot counts wired and active) and `batched uploads ENABLED` (see
`docs/PHASE1-RESULT.md` for why that doesn't move throughput). Clean shutdown,
5/5 completions succeeded, no crash, no assert.

## What this does and doesn't establish

It establishes the mechanism is real: online updates change the weights, the
changed weights score a repeated sequence better, and the async pipeline drains
without a backlog under decode load. It does **not** establish that
online-updated weights beat the static `predictor-real.bin` on end-to-end
throughput -- the predictor-driven admission path was already shown
(`docs/EVICTION.md`, PLAN-NEXT.md's Phase 3 discussion) to be a small,
uncertain lever compared to eviction, and this check did not re-run that
comparison with online learning enabled. That would be the next question if
this line of work continues.
