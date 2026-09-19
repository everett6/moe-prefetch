# Does the headline survive real prompts?

## The problem with the old number

Every throughput figure in this repo was measured on one prompt, which I wrote:

> Write a detailed explanation of how a B-tree index works in a relational
> database, including insertion, node splitting, deletion and range scans.

150 characters of fluent English asking for English back. `118.2 ± 1.66 tok/s`
was thirty repeats of that single prompt. The `±` is repeatability, not
uncertainty about workloads, and the two were being read as the same thing.

## Method

Same binary, same flags, same three harness guards. One variable changed: the
prompt, from mine to 23 real ones — stratified across coding and
decision-making and across three length bands, from 9 to 2,695 tokens, drawn
from ten real sources. Three rounds, configs interleaved, n=69 measurements per
config. `experiments/r3_real_bench.py`, `artifacts/r3_real_throughput.json`.

## Result

| config | real prompts (n=69) | control first / mid / last |
|---|---|---|
| AI2 shipped `-ncmoe 22`, no cache | 78.7 | 77.3 / 79.0 / 80.4 |
| this work, with mmap | 120.0 | 103.8 / 108.1 / 113.2 |
| **this work + `--no-mmap`** | **126.2** | 119.1 / 121.1 / 119.3 |

**The headline holds, and is better on real traffic than on my prompt.** 126.2
tok/s against a baseline of 78.7 is **1.60×**, where the synthetic control
measured 1.49×. Nothing about real coding and decision workloads degrades the
result.

## The `±` was the wrong kind of number

| config | between-prompt sd | within-prompt sd | sem of the mean |
|---|---|---|---|
| baseline | 1.12 | 2.38 | 0.23 |
| prev best | 9.88 | 4.84 | 2.06 |
| new best | **7.27** | 2.70 | 1.51 |

Repeating one prompt measures the 2.70. Real workloads vary by 7.27 — nearly
three times as much, and invisible to any single-prompt benchmark. The old
`± 1.66` was not wrong, it was answering a narrower question than it appeared
to. The right figure for a workload mean is the sem, 1.51.

The control prompt's 119.3 sits about one between-prompt standard deviation
below the real-workload mean of 126.2. It is one draw from a wide distribution,
not a slow outlier that needs explaining.

## A prediction of mine, falsified

I expected longer prompts to be *faster* under the cache: decode tok/s is
measured over the decode phase only, so a long prefill is free to the metric but
not to the cache, which it fills. Regressing decode tok/s on log prompt tokens:

| config | pearson | spearman |
|---|---|---|
| baseline | **−0.649** | −0.668 |
| prev best | −0.178 | −0.190 |
| new best | −0.274 | −0.265 |

Longer prompts are *slower*, in every config — and most strongly in the baseline,
which has no expert cache at all. Whatever this is, it is not cache warming; the
shape is what growing attention cost over a longer context would produce. The
hypothesis is recorded as refuted rather than quietly dropped.

## Two confounds I built and then removed

Both were mine, both would have produced a confident wrong claim, and the first
version of this script had both.

**Control placement.** The control ran once, at the start of each pass. The
expert cache is a server-lifetime LRU, so it met a colder cache than the 23 real
prompts after it — "real prompts are faster" and "later prompts are faster"
would have been indistinguishable. Running the control first, middle and last
separates them, and shows the position effect is not even consistent: it rises
9.4 tok/s across a pass on the mmap config and is flat on the `--no-mmap` one.

**Prompt ordering.** The bench set is built stratified by length band, which put
every long prompt late in each pass, so length and recency moved together.
Prompts are now shuffled per round.

The first, confounded run is kept as `artifacts/r3_real_throughput_uncontrolled.json`.
It reported the same direction with a larger gap, which is what a confound in
that direction would do.
