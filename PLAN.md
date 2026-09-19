# Plan: close the predictor gap, then make it self-improving

*Written 2026-09-18, after the M5 plan closed at 129.1 tok/s.
The closed plan is kept at [`docs/PLAN-m5-closed.md`](docs/PLAN-m5-closed.md).*

## Where the value is now

The project cleared its bar — 129.1 tok/s at Q4_K_M against a 110 target, 1.66x
over today's 77.9. So the question changes from *does this work* to *what is
still on the table*, and the answer is unusually clear:

| | all-8 on time | tok/s |
|---|---|---|
| LRU cache alone | 57.3% | 113.1 |
| **current: LRU + ridge probe, top-16** | **69.2%** | **129.1** |
| oracle: perfect prediction, same cache | 98.5% | **167.8** |

**A better predictor is worth up to +38.7 tok/s** — more than twice what the
current predictor won over plain LRU (+16). Nothing else available is close to
that, and none of it needs root.

Two other things are unfinished, one of which was in the original brief:

- **"Self-trained / self-improving" was asked for and has not been built.** The
  probe is fitted once, offline, and frozen. That is a static predictor, not a
  self-improving one.
- **The larger model** (`docs/SCOPE.md` Option B) was always gated on the
  mechanism proving out on 30B. It now has.

## Phase A — close the predictor gap (biggest lever, no root)

The current probe is linear, single-input, and one layer ahead. Each of those is
a choice that can be revisited, cheapest first.

**A1. Fix the MLP. — DONE 2026-09-18. Hypothesis confirmed, payoff marginal.**

Proper training (Adam + cosine schedule, BCE-with-logits instead of softmax on a
multi-label target, early stopping on the validation register, hidden width
chosen per layer) moved the MLP **+9.5 points, 43.7% → 53.2%**. So milestone 2's
gap really was the optimiser, not nonlinearity — but the fixed MLP still only
*ties* linear ridge (53.2 vs 54.1) rather than beating it.

| predictor | recall@8 |
|---|---|
| **ridge+MLP blend + prior** | **60.5%** |
| per-layer pick (on validation) + prior | 60.0% |
| ridge + prior (milestone 2's best) | 59.7% |
| MLP + prior | 58.6% |
| ridge alone | 54.1% |
| MLP alone | 53.2% |
| naive-repeat | 39.5% |

The MLP beats ridge on **31 of 47 layers** yet loses on average, i.e. it fails
badly on a few rather than being uniformly worse — which is why blending the two
helps at all. Net gain over milestone 2's best: **+0.8 points of recall@8**,
worth roughly +1 tok/s. Real, but not the +38.7 that is on the table.

**Conclusion: nonlinearity is not the bottleneck.** 3.5K rows per layer against
256 features is not much to fit a nonlinearity on, and the signal that is missing
is not a curved version of the signal already there. Effort moves to A2.

**A1 (original text). Fix the MLP.** It scored 43.7% against ridge's 54.1% in milestone 2, which
almost certainly says more about plain SGD at a fixed learning rate for 150
epochs than about nonlinearity. Redo it properly — Adam, learning-rate schedule,
early stopping on the validation register. *If a tuned MLP still loses to ridge,
that is a real finding and A2 is where the effort goes instead.*

**A2. Richer inputs.** The probe sees only `h_L`. Three additions, each free at
inference time because the model has already computed them:
- `ffn_moe_logits-L` — the router's own 128-d scores for layer L
- the previous token's experts at layer L+1 as *features*, not merely as the
  ensemble prior bolted on afterwards
- layer index and token position

**A3. Predict two layers ahead.** `h_L → E_{L+2}` gives the copy two layers of
compute instead of one. Accuracy will fall; lateness (currently 4.6–5.0%) will
fall too. Worth knowing which way the trade lands, since it costs one experiment.

**A4. Spend the cache asymmetrically.** Capacity is currently uniform at 66/layer.
Milestone 1 found routing predictability varies sharply by depth — the probe got
48.7% at layer 0 and 61.4% at layer 46, against naive-repeat's 21.9% and 27.2%.
Layers where prediction is weak may deserve more cache, and vice versa.

**Target: all-8-on-time above 78%, which is 139 tok/s.** Above 86% would be 149.

## Phase B — make it self-improving (the unfinished half of the brief)

The predictor is wrong about 30% of layers, every token, and it *knows* — the
true experts are revealed microseconds later when the layer runs. That is a free
training signal being thrown away, and it is exactly the mechanism AI2 already
built for its draft model in `draft_trainer.py`.

**B1. Online ridge.** Ridge regression has an exact incremental form (recursive
least squares), so the probe can be updated per token without retraining. Bound
the per-token cost: it must stay far below the 2.92 MiB fetch it informs.

**B2. Measure whether it actually helps.** The honest test is distribution shift,
because that is the only place online adaptation can win: fit on 5 registers,
then run on a *held-out* register with updating on versus off. If online learning
does not beat the frozen probe there, it does not beat it anywhere, and B3 is
dead.

**B3. Only if B2 pays: persistence.** Checkpoint the adapted probe so a session
starts where the last one finished, with a guard against drift on a workload
change.

## Phase C — the larger model (`docs/SCOPE.md` Option B)

Now justified, not before. The mechanism is proven on 30B and Phase A/B improve
it further.

**C1. Re-price it with real numbers.** A cold token needs ~656 MB of experts; a
consumer NVMe does 3–7 GB/s, so a miss that reaches disk costs 100–200 ms against
the 39 µs a CPU-computed miss costs now — **a 3,000x penalty**. The RAM tier has
to absorb essentially everything. Compute the hit rate that requires *before*
downloading 45–60 GB.

**C2. If C1 survives**, pick the model and capture traces. Everything downstream
reuses Phase A and B unchanged.

## Phase D — llama.cpp integration (blocked)

Specified in [`docs/INTEGRATION.md`](docs/INTEGRATION.md). Needs the CUDA
toolkit as root. Start from PR #27861, not mainline. Unblocked the moment someone
with root runs one `apt`/runfile install; nothing in Phases A–C depends on it.

## Order, and why

**A1 → A2 → B2 → A3/A4 → C1.** Phase A first because +38.7 tok/s dwarfs
everything else available. B2 early because it is cheap and it decides whether
Phase B exists at all — and because a self-improving predictor was asked for, so
"we measured it and it did not help" is a legitimate answer but "we never tried"
is not.

**Rule carried over from AI2, which has earned it twice in this repo alone:**
price it before building it, and when a result contradicts the design, suspect
the test before the signal. The one-shared-probe collapse in milestone 1 and the
`admit()`-without-copy bug in milestone 4 were both nearly reported as findings.
