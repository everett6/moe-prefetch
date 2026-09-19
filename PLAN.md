# Plan

*Rewritten 2026-09-19 after two things invalidated the previous one: D1's
headline turned out to be a measurement error, and the project was deliberately
reset to a fresh corpus and a fresh model.*

## Where this actually stands

| | |
|---|---|
| **measured best** | **~105 tok/s** at Q4_K_M, `-ncmoe 48 --moe-expert-cache 66` with early issue |
| against | 78.8 tok/s for AI2's shipped Q4_K_M config, 45 with the cache off |
| the bar | 110 tok/s (what `ud-q3_k_xl` already does at Q4_K_M-equal accuracy) |
| **the cache as shipped** | **33 tok/s** — worse than no cache; see [`docs/D1-RESULT.md`](docs/D1-RESULT.md) |
| predictor | being retrained from scratch on a fresh 28 GB corpus |

Two claims from the old plan are withdrawn. The 133 tok/s projection rested on a
cost model that D1 was supposed to have validated and did not. And "the predictor
adds +20" was never true of the mechanism actually available: at PR #27861's
step-boundary admission a *perfect* oracle is worth +4.1 points of all-8-resident,
because by the time the cache admits anything it already knows the answer.

## What the problem turned out to be

Not prediction quality. Bandwidth economics.

A cost model fitted to five end-to-end measurements spanning 33 to 106 tok/s
(`artifacts/cost_model.json`, worst residual 4.9%) gives

    ms/token = 4.18 + 0.047 * missed_experts + 0.142 * uploads

so a cache miss costs **47 µs** and an upload costs **142 µs**. A *correct*
prefetch adds no upload — the demand path would have fetched that expert anyway —
so it is worth +47 µs; an incorrect one is a wasted 142 µs. That sets a hard bar:

> **A prefetch pays only if its precision on non-resident experts exceeds
> C/(B+C) = 75%.**

Every number this project has ever reported was a *recall*, and recall is not the
quantity that decides. Measuring precision against 75% is what the remaining work
is for.

## The sequence

- **R0 — reset.** Done. v1 corpus (173,991 samples) and a superseded partial v2
  capture deleted; draft predictors archived, isolated, in `artifacts/draft_old/`
  for the comparison arm only. `artifacts/reset_record.json`.
- **R1 — fresh corpus, v3.** 616 prompts over 16 registers, sequence length,
  context size and batch size all varied across sessions, ~28 GB. Every row
  labelled with how many of its experts would miss a 66- and a 32-slot LRU, and
  its overlap with the previous token — so "hard rows" can be scored separately
  from rows where the cache already had everything.
- **R2 — supervised.** Architecture search under a **20K MAC** budget derived
  from the 198 µs layer time, not chosen for elegance. Selected on validation.
- **R3 — reward-driven.** Prefetch depth as the action, reward in milliseconds
  from the fitted cost model. The informative output is the sweep over upload
  cost: the break-even tells you which engineering change would matter.
- **R4 — locked test, three ways.** `DRAFT_OLD` / `FRESH_SUPERVISED` / `FRESH_RL`
  under identical conditions, never mixed.
- **R5 — system.** Re-tune the operating point (cache size, insert rate, split)
  now that the cache works, then measure end to end with repetitions.

## Two things that must not slip

**Output identity.** Measured, and it fails — for the shipped cache as much as
for any change of ours. PR #27861 splits one `mul_mat_id` into a device cache
chain and a host chain and sums them, so float addition order changes and greedy
decoding diverges. The control passes (cache-off twice is byte-identical), so
this is the cache, not nondeterminism. The design's premise that prefetching
"changes only where weights live" is false for this implementation, and the
acceptance criterion has to say so rather than be quietly dropped.

**Measurement hygiene.** The harness verifies the port is free, that the PID it
measures is the one it launched, and that the cache allocated its ~8.8 GB. All
three exist because their absence produced a headline number that was wrong by
3.4x.

## The rule this repo keeps re-earning

When something looks blocked, check the block. "Needs root" was wrong about the
toolkit; "needs a CUDA build" was wrong about the hop cost; the shared-probe
collapse was the test, not the signal; `admit()`-without-copy flattered the
baseline; and the 112.8 was an orphaned server. Five for five, and four of them
were nearly reported as findings.
