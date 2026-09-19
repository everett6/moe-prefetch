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

## Why this project expects to beat 47.8% anyway

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
