#!/usr/bin/env bash
# R6: re-price the economics on the real-prompt corpus.
#
# The v3 answers were: break-even precision 75%, and a PERFECT predictor worth
# -0.4 tok/s (artifacts/oracle_ceiling.json). Those were computed on traces from
# a corpus with no code in it. Whether they hold on real coding and decision
# traffic is a question about the data, not about the arithmetic -- so it gets
# re-run rather than assumed.
#
# The upload cost itself (e1) is a property of the hardware and the transfer, not
# of the prompts, so it is reused unless --remeasure is passed.
set -eu
cd "$(dirname "$0")/.."
INDEX=${INDEX:-data/index-v4.npz}
CKPT=${CKPT:-artifacts/ckpt/FRESH_REAL-linearctx-lr1e2.pt}

if [ "${1:-}" = "--remeasure" ]; then
  echo "== E1: upload cost =="
  python3 -u experiments/e1_upload_cost.py
fi

echo "== E5: the ceiling -- a perfect predictor on real traces =="
python3 -u experiments/e5_oracle_ceiling.py "$INDEX"

echo
echo "== E2: precision against break-even, real traces, LOCKED TEST =="
python3 -u experiments/e2_precision.py --index "$INDEX" --ckpt "$CKPT" --split test
