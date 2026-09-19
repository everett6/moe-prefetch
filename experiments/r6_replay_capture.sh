#!/usr/bin/env bash
# A small corpus with position subsampling OFF, for the economics replay only.
#
# The main v4 corpus stores at most 128 positions per prompt, which is right for
# training -- positions inside one prompt are highly correlated, prompts are not,
# and the byte budget should buy prompts. It is wrong for E5, which replays the
# trace through PrefetchEnv in (prompt, position, layer) order: a subsampled
# trace has gaps, and a cache replayed over gaps misses more than the real
# sequence does. The per-row LABELS are computed on the full trajectory and are
# unaffected; the REPLAY is what needs contiguity.
#
# 40 prompts is enough: the round-robin puts the first 30 in 30 distinct groups,
# so coverage is one prompt per repo/site/category before any repeats. Cost is
# compute, not storage, so this is about seven minutes.
set -u
cd "$(dirname "$0")/.."
export PROMPT_SET=real
export PER_REGISTER=1400
export LIMIT=${LIMIT:-40}
export KEEP_POSITIONS=100000        # effectively off
export TARGET_GB=0
export N_THREADS=${N_THREADS:-24}
export OUTDIR=${OUTDIR:-$PWD/data/corpus-v4-replay}
export DATASET_VERSION=v4-20260919-real-replay
export FLUSH_ROWS=${FLUSH_ROWS:-600000}
echo "replay capture -> $OUTDIR  ($LIMIT prompts, all positions)"
exec python3 -u experiments/capture_corpus.py "$@"
