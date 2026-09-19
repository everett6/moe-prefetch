#!/usr/bin/env bash
# R4: capture routing on real prompts -> dataset v4.
#
# Resumable. Killing this and re-running it picks up at the next prompt: the
# manifest records done_prompt_ids after every shard and is written atomically,
# so an interrupted run costs at most one shard.
#
# The knobs and why they are set where they are:
#
#   PROMPT_SET=real     data/prompts/*.jsonl -- real GitHub issues, real Stack
#                       Exchange questions, real crowd-sourced tasks
#   PER_REGISTER=1400   prompts, round-robin across all 30 groups so that any
#                       prefix is balanced if the byte cap stops the run early
#   KEEP_POSITIONS=128  positions STORED per prompt, chosen after labelling.
#                       Real prompts are ~30x longer than the synthetic ones, so
#                       a fixed byte budget would otherwise buy ~30x fewer of
#                       them. Positions within a prompt are highly correlated
#                       and prompts are not, so the budget goes on prompts.
#   TARGET_GB=30        hard cap; the run stops at the next shard boundary
#
# Capture is CPU-only: the eval callback reads tensor data directly, and with
# layers on the GPU t.data is a device pointer -- reading it segfaults rather
# than returning a wrong number.
set -u
cd "$(dirname "$0")/.."
export PROMPT_SET=${PROMPT_SET:-real}
export PER_REGISTER=${PER_REGISTER:-1400}
export KEEP_POSITIONS=${KEEP_POSITIONS:-128}
export TARGET_GB=${TARGET_GB:-30}
export OUTDIR=${OUTDIR:-$PWD/data/corpus-v4}
export DATASET_VERSION=${DATASET_VERSION:-v4-20260919-real}
export FLUSH_ROWS=${FLUSH_ROWS:-400000}
echo "capture -> $OUTDIR  (target ${TARGET_GB} GB, $PER_REGISTER prompts, keep $KEEP_POSITIONS pos/prompt)"
exec python3 -u experiments/capture_corpus.py "$@"
