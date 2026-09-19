#!/usr/bin/env bash
# R5: index the v4 corpus and train a predictor on it from scratch.
#
# From scratch means from scratch: no warm start from the v3 checkpoints, which
# were fitted on a corpus containing no code. Results are tagged FRESH_REAL and
# never pooled with FRESH_SUPERVISED (v3 synthetic) or DRAFT_OLD -- the corpora
# are different enough that a shared table would be meaningless.
set -eu
cd "$(dirname "$0")/.."
CORPUS=${CORPUS:-data/corpus-v4}
INDEX=${INDEX:-data/index-v4.npz}

if [ ! -f "$INDEX" ]; then
  echo "== building index from $CORPUS =="
  python3 -u experiments/build_index.py "$CORPUS" "$INDEX"
fi

echo "== stage A: architecture sweep on real data =="
python3 -u experiments/train_deep.py --index "$INDEX" --tag FRESH_REAL "$@"
