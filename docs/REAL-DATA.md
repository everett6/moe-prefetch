# The corpus, and why the old one had to go

## What the old corpus was

`experiments/corpus_prompts.py` builds 616 prompts by crossing templates I wrote
with topic lists I wrote. Every prompt in it is mine. It was used to train the
predictor, to fit the cost model, and — this is the part that matters most — to
benchmark the whole system.

## What it looks like next to real traffic

Measured by `experiments/r2_characterise.py`, tokenised with the model's own
tokenizer (`artifacts/r2_characterisation.json`):

| | real | synthetic |
|---|---|---|
| prompts | 6,720 | 616 |
| tokens, p50 | 123 | 14 |
| tokens, p90 | 1,049 | 17 |
| tokens, max | 15,133 | 24 |
| tokens, total | 2,689,079 | 8,405 |
| chars per token | 3.46 | 5.02 |
| contains a code fence | 32.7% | **0.0%** |
| contains indented code | 23.5% | **0.0%** |
| contains a stack trace | 7.0% | **0.0%** |
| contains any code at all | 52.5% | **0.0%** |

Vocabulary, on a 400-prompt sample of each: the real corpus uses 13,278 distinct
token types, the synthetic one 848. **3.7%** of the token types in real traffic
appear anywhere in the corpus the predictor was trained on.

The single most damaging line is the one that reads 0.0%. The whole project is
about routing for coding workloads, and not one of the 616 training prompts
contains a line of code. The chars-per-token row is the same fact from another
angle: 5.02 is the signature of plain English, 3.46 of text with identifiers and
punctuation in it.

## What replaced it

6,720 prompts that humans wrote for their own purposes, fetched by
`experiments/real_prompts.py` with licence, upstream URL, fetch timestamp and a
sha256 per record.

### Coding
| source | what it is | groups | n |
|---|---|---|---|
| SWE-bench | real GitHub issues from 12 real Python repos | repo | 2,275 |
| GitHub issues | live issues: llama.cpp, vLLM, transformers, PyTorch, NumPy | repo | 902 |
| MBPP | crowd-sourced programming tasks | 1 | 970 |
| HumanEval | hand-written problems | 1 | 164 |
| Stack Overflow | real questions, python and c++ tags | tag | 319 |

### Decision-making
| source | what it is | groups | n |
|---|---|---|---|
| softwareengineering.SE | architecture and trade-off questions | 1 | 350 |
| devops.SE, security.SE | operational decisions | 2 | 240 |
| no_robots | human-written instructions | category | 800 |
| OpenAssistant oasst1 | real human first turns, English | 1 | 700 |

Prompts are capped at 20,000 characters and the cap is recorded per record;
`truncated` is a field, not a silent edit.

## The split

The unit is the **group** — a repo, a Stack Exchange site, a category — not the
prompt, and groups are stratified by domain so LOCKED TEST always contains both
coding and decision-making groups. A held-out group is unseen code in an unseen
project, which is a much harder question than the old register split asked: that
one held out one half of a template from the other half of the same template.

## What capture had to change

Real prompts broke three things that a 14-token corpus never exercised.

**A fixed 256-token buffer.** `llama_tokenize` was called with a 256-entry array.
It returns a *negative* count when the prompt does not fit, and the return was
not checked — so every prompt over roughly 800 characters would have passed a
negative length to `llama_batch_get_one`. Fine for a corpus whose longest prompt
was 24 tokens; catastrophic and silent for a 15,000-token GitHub issue.

**Position subsampling, done in blocks.** Real prompts are ~30× longer, so a
fixed byte budget buys ~30× fewer of them. Positions inside one prompt are
highly correlated and prompts are not, so the budget goes on prompts: at most
128 positions are stored per prompt. Two constraints on how they are chosen —
labels (`lru_miss_66`, `repeat_hit`) are computed on the **full** trajectory
before anything is dropped, or every row would look like a cache miss; and
positions are kept in contiguous blocks, because the model's strongest feature
is the *previous token's* experts and an evenly spaced sample has no t−1 for
almost any t. That failure would have been silent: −1 is a legal value for that
feature. Blocks of 8 leave 91% of kept rows with their predecessor.

**Context sized to the prompt.** n_ctx is now the smallest rung of
{1024, 2048, 4096, 8192} that holds prompt plus continuation, while n_predict
and n_batch cycle on the prompt id so they stay orthogonal to length rather than
confounded with it.

## Capture is only reproducible at a fixed thread count

`llama_cpp` defaults to 4 threads whatever the machine. On 32 cores that ran the
capture — the long pole of the whole pipeline — at a fraction of its throughput.
Measured on the same six prompts:

| n_threads | time | rows |
|---|---|---|
| 4 (the default) | 127 s | 34,849 |
| 16 | 98 s | 34,849 |
| **24** | **84 s** | 34,849 |
| 32 | 233 s | 34,849 |

24 is the operating point: 1.51× over the default, and 32 is **2.8× slower than
24**, which is oversubscription rather than anything subtle. Worth stating that
the core count does not predict this — the 6× the hardware suggests is not
available, because MoE inference here is bandwidth-bound.

The row counts are identical at every thread count. The **checksums are not**.
Comparing the 4-thread and 24-thread captures row by row:

- every prompt is bit-identical up to one position, and 100% different after it
- the divergence points are late: 578/601, 2158/2181, 256/258, 596/749
- overall 12.5% of rows differ, and 11% of rows route to a different expert set

This is greedy decoding diverging, not numerical noise in the features: a
reduction-order difference flips one sampled token, and from there the sequences
are simply different text. Both continuations are legitimate model behaviour, so
neither dataset is more correct than the other — but capture is reproducible
only at a fixed thread count, which is why `n_threads` is recorded in the
manifest alongside the model hash and the git commit.
