# moe-prefetch

Asynchronous predictive expert prefetching for MoE inference on a 12 GB GPU,
driven by a self-improving token predictor.

**Goal:** run a mixture-of-experts model at **Q4_K_M** — the quality reference —
at speeds currently only reachable by quantizing down to Q2_K.

This is a follow-on to [AI2](../AI2), which took Qwen3-30B-A3B from 47 to 188
tok/s on this machine but got there partly by *trading quality away*: the default
is Q2_K (10.2 GiB) rather than Q4_K_M (17.3 GiB). Q4_K_M runs at 77.9 tok/s
because 22 of its 48 expert layers have to live in CPU RAM. This project attacks
that directly instead of quantizing around it.

---

## The idea

At Q4_K_M, 22 layers' experts are in host RAM. Every token, the ones it needs get
used from the CPU — which is the whole cost. If the experts a token is *about to*
need were already copied into VRAM before it needed them, that cost would be
hidden behind compute that was happening anyway.

Three things have to be true for that to work. Two are already measured, on this
exact machine, and one is the open question this project exists to answer.

### 1. There must be enough PCIe bandwidth — **measured, yes**

From [`AI2/experiments/PREFETCH_FEASIBILITY.md`](../AI2/experiments/PREFETCH_FEASIBILITY.md):

- **2.92 MiB per expert** (gate+up+down), read from the GGUF tensor table.
- **53.66 GB/s** sustained host→device, measured with many small ~2.92 MB async
  copies on a dedicated stream — this slot has no small-transfer penalty.
- Fetching a token's "new" experts costs **0.255 ms/layer** pipelined.

### 2. Expert selection must be predictable — **measured, yes**

From [`AI2/experiments/EXPERT_TRACE_FINDINGS.md`](../AI2/experiments/EXPERT_TRACE_FINDINGS.md),
over 82K token-pairs: consecutive tokens reuse **44.2%** of the same experts in
the same layer, against a **6.2%** random baseline. That is ~7x signal.

### 3. Something must predict *which* experts — **this is the open question**

And here is the finding that shapes this entire project. AI2 built three expert
predictors and **every one lost to doing nothing**:

| Predictor | Recall@8 (held-out) |
|---|---|
| **naive-repeat** — "same experts as last token", zero training | **47.8%** |
| MLP + learnable residual to the repeat prior | 41.3% |
| MLP (multi-hot + layer embedding) | 35.5% |
| Markov, per-layer pairwise co-occurrence | 26.8% |

The honest reading recorded there: *"the exploitable structure is almost entirely
'experts recur as themselves across adjacent tokens' — not richer cross-expert
transition patterns."*

## Milestone 2 result (2026-09-18): 59.7% on a clean split, +20.2 points over the bar

Milestone 1's 51.3% was a ceiling — ridge strength and ensemble weight were both
chosen on the held-out rows. Milestone 2 removes that and scales the data:
**173,991 rows from 40 prompts across 8 registers**, split three ways *by
register* (5 train / 1 validate / 2 test), every hyperparameter chosen on
validation, test registers scored once.

| predictor | recall@8 | prefetchable? |
|---|---|---|
| **ridge probe on `h_L` + repeat prior** | **59.7%** | **yes** |
| ridge probe on `h_L` alone | 54.1% | yes |
| MLP probe + repeat prior | 52.0% | yes |
| MLP probe on `h_L` alone | 43.7% | yes |
| naive-repeat (the bar) | 39.5% | no |
| random | 6.3% | — |

**+20.2 points over the naive-repeat bar, winning on 47 of 47 layer transitions**
— and unlike the bar, this is prefetchable. Test registers (`science`,
`technical`) were never seen in training or tuning. Features are PCA to 256 dims
fitted on train rows only, retaining 98.8% of variance.

It holds across depth, which is what makes it look structural rather than fitted:

| layers | probe + prior | naive-repeat |
|---|---|---|
| 0–15 | 57.8% | 37.8% |
| 16–31 | 58.0% | 43.3% |
| 32–46 | 63.3% | 37.2% |

Per-layer gain over the bar ranges from +1.0 to +56.0 points; it is never
negative.

**The MLP underperformed the linear ridge** (43.7% vs 54.1%), which is worth
flagging rather than hiding. That is most likely my training setup — plain SGD,
fixed learning rate, 150 epochs, no tuning — and not evidence that nonlinearity
cannot help. It is an open item, not a finding.

**Next (milestone 3):** build the async prefetch engine against this predictor.
The predictor is cheap by construction — a 256x128 matrix per layer on PCA
features, which is ~33K multiply-adds per layer against the ~2.9 MiB expert
fetch it decides.

---

## Milestone 1 result (2026-09-18): green light, via a different mechanism

**A prefetchable predictor reaches 51.3% recall@8, against the 45.8% bar that
beat every trained predictor in AI2.** The route there was not the one this repo
originally proposed, and the first two attempts both failed:

| predictor | recall@8 | prefetchable? |
|---|---|---|
| **per-layer probe on `h_L` + repeat prior (ensemble)** | **51.3%** | **yes** |
| naive-repeat, same layer, previous token | 45.8% | no — needs `h_L(t+1)` |
| per-layer ridge probe on `h_L` alone | 40.6% | yes |
| *one* probe shared across all 47 layer transitions | 9.4% | yes |
| cross-layer expert IDs (`E_L` → `E_{L+1}`) | 6.5% | yes |
| random (8 of 128) | 6.2% | — |

Three things made the difference, in order of how much they mattered:

1. **Predict from the hidden state, not from expert IDs.** `E_L → E_{L+1}` is at
   the random floor. `h_L → E_{L+1}` is not — 2048 floats carry what 8 integers
   cannot.
2. **One probe per layer transition.** A single shared probe scores 9.4%; giving
   each transition its own scores 40.6%. Routing is layer-specific and a shared
   probe cannot represent 47 different gates at once. This was the difference
   between concluding "no signal" and finding it.
3. **Ensemble with the repeat prior.** The probe and naive-repeat are
   *anti-correlated by depth* — where one is weak the other is strong:

   | layers | probe | repeat |
   |---|---|---|
   | 0–15 | 44.4% | 38.8% |
   | 16–31 | 36.6% | 44.8% |
   | 32–46 | 40.6% | 40.5% |

   At layer 0 the probe gets 48.7% where repeat gets 21.9%; at layer 46, 61.4%
   against 27.2%. The probe wins on 24 of 47 layers. Combining them beats both
   by ~10 points — and layer 0 being the probe's *strongest* region is convenient,
   because it is the layer with the least prefetch budget.

**Both ensemble inputs are available before layer L+1 computes**: `h_L` is in hand
one layer early, and the repeat prior `E_{L+1}(t-1)` is known from the previous
token. That is what makes this prefetchable where naive-repeat is not.

### Caveats, stated because they bound the number

- **51.3% is optimistic.** The ridge strength and the ensemble weight were both
  chosen by their held-out score, so that figure is a ceiling, not a clean
  generalisation estimate. A proper validation split is the first job of
  milestone 2.
- **Small data:** 8 prompts, 557 token positions, 2 prompts held out. The
  qualitative finding — that the probe is prefetchable and complementary to
  repeat — rests on a structural pattern across 47 layers rather than on the
  headline average, which is why it is believable at this scale and the exact
  percentage is not.
- **Linear probes only.** No nonlinearity has been tried yet.

---

## Correction (2026-09-18): the originally proposed mechanism does not work

**The token-predictor design described in the next section has a dependency flaw,
and the cross-layer alternative has no signal. Both are recorded here rather than
quietly deleted, because the measurement is the useful part.**

A layer's router takes *that layer's* hidden state `h_L(t)`. To know which experts
token t+1 needs at layer L you need `h_L(t+1)` — which requires having already run
layers 0..L-1 on token t+1. Knowing the token's *identity* early does not give you
its hidden states. So "predict the token, then ask the real router" cannot run
ahead of the computation it is supposed to prefetch for. The idea was sound about
*why* the AI2 predictors failed and wrong about what to do instead.

The standard fix in the literature (Pre-gated MoE, FATE) is to predict **across
layers within a token**: while layer L computes, predict what layer L+1 will
select. That direction genuinely has lookahead. `experiments/m1_prediction_signals.py`
measured whether it has any *information*, on AI2's 255,360-row Q4_K_M trace:

| signal | recall@8 | can it prefetch? |
|---|---|---|
| same-layer, previous token (`naive-repeat`) | **45.8%** | **no** — needs `h_L(t+1)` |
| cross-layer, same token | **6.5%** | yes — `h_L` is in hand one layer early |
| random (8 of 128) | 6.2% | — |

**The signal with lookahead carries essentially no information** (6.5% against a
6.2% floor; the best layer pair, 16→17, reaches only 9.1%). Expert selections at
adjacent layers of the same token are very nearly independent.

So: the informative signal cannot prefetch, and the prefetchable signal is not
informative. That is a real obstacle, not a tuning problem.

**What this does *not* rule out:** the test predicts expert IDs *from expert IDs*.
Pre-gated MoE predicts `E_{L+1}` from the hidden state `h_L`, which carries far
more information than 8 integers — at the cost of modifying the model so the
gate emits next-layer routing, and retraining it. That remains open and is the
only route by which a *trained* predictor beats the trivial one here.

**What still works without any predictor:** prefetching across tokens using
naive-repeat. When token t finishes, `E_L(t)` is known for every L, and token
t+1 does not reach layer L until it has computed L-1 layers — roughly
`L x 0.27 ms` of budget against a 0.24 ms fetch. That is feasible for every layer
but the first few, at a 45.8% hit rate, and it is essentially what llama.cpp
PR #27861's LRU cache already does for +11.7% to +40%.

---

## Why this project expected to beat 47.8% *(superseded — see the correction above)*

**Because all three failed predictors were trying to predict experts from past
experts.** That is a statistical shortcut around the thing that actually decides
the answer.

Expert selection is not stochastic. It is a *deterministic function of the
hidden state* — the router picks top-k from a gate projection. Nothing has to be
learned about "which experts follow which." The router already knows, exactly,
given the token.

So this project predicts **the next token**, not the next experts:

```
  draft model  ──predicts──▶  token t+1  ──run through the REAL router──▶  exact expert IDs
                                                                                │
  naive-repeat (47.8%) ◀── fallback when the draft is wrong ──┐                 │
                                                              ▼                 ▼
                                              prefetch into VRAM, asynchronously,
                                              while token t is still computing
```

When the draft token is **right**, the expert prediction is not a guess — it is
*exact*, because it came from the same router the real token will use. AI2's
measured draft accept rates for this model family were **66–75%**. When the draft
is wrong, fall back to naive-repeat's 47.8%.

A rough composite, to be measured rather than believed: `0.70 x ~100% + 0.30 x
47.8% ≈ 84%` — against the 47.8% ceiling that beat every trained predictor. That
gap is the entire thesis of this repo, and **the first milestone is to measure
it, not to build the prefetcher.**

The token predictor is the self-improving part: it trains online on its own
mismatches, the mechanism AI2 already built in `draft_trainer.py`. Every token it
learns to predict correctly converts a statistical expert guess into an exact one.

## Win condition

Stated up front so the project can fail honestly:

> **Q4_K_M above 110 tok/s**, with output quality unchanged (it must be
> byte-identical under greedy decoding — prefetching changes *where weights live*,
> never which experts are selected or what is computed).

110 tok/s is the bar because that is what `ud-q3_k_xl` already does in AI2 at
accuracy statistically indistinguishable from Q4_K_M. **Below that bar this
project is pointless** — the shelf already has the answer, and AI2's Phase E was
declined for exactly this reason. Above it, this is the fastest Q4_K_M on a 12 GB
card that we are aware of.

## Hardware

RTX 5070 (12 GB, PCIe 5.0 x16), Ryzen 9 7950X, **29 GB usable RAM**, 499 GB NVMe.

The RAM figure is a real constraint on "a larger model": Q4_K_M of a 30B-A3B is
17.3 GiB and fits, but a genuinely larger MoE (80B+ at Q4) is 45–60 GiB and does
not fit in 29 GB of RAM plus 12 GB of VRAM. Going larger means a third tier —
NVMe → RAM → VRAM — which is a different and much harder project. See
[`docs/SCOPE.md`](docs/SCOPE.md).

## Status

Nothing is built yet. This README is the design and the prior art it rests on.
The first milestone is measurement, per the rule AI2 learned the hard way:
**price it before building it.**

## Credit where it is due

The three measurements this design stands on — bandwidth, predictability, and the
negative result on trained expert predictors — were all produced by
[AI2](../AI2). This project would have started in the wrong place without them,
and probably would have built the fourth losing expert predictor.
