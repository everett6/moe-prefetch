#!/usr/bin/env bash
# Start llama-server in the fastest configuration measured on this machine.
#
#   ./serve.sh                 Qwen3-30B-A3B, output unchanged      119.3 tok/s
#   ./serve.sh fast            + cache-aware routing, delta 0.01    130.1 tok/s
#   MODEL=qwen36 ./serve.sh       Qwen3.6-35B-A3B, output unchanged 108.0 tok/s
#   MODEL=qwen36 ./serve.sh fast  + cache-aware routing, delta 0.01 127.0 tok/s
#
# Numbers are real-prompt decode speeds measured on 2026-09-30 / 10-01 (RTX 5070
# at 175 W, 56 or 96 cache slots a layer); method in
# docs/superpowers/specs/2026-10-01-graph-resident-decode-design.md.
#
# "fast" lets a resident expert stand in for a non-resident one the router
# ranked within 0.01 of it. That CHANGES the model's output. Measured
# perplexity change at 0.01: -0.10% +/- 0.42% on Qwen3-30B-A3B, -0.38% +/- 0.68%
# on Qwen3.6-35B-A3B, i.e. none that the test can see. At 0.02 it is +0.9% and
# +2.0%. DELTA overrides the 0.01; a model with no measurement is refused
# unless FORCE_FAST=1.
#
#   PORT (default 8080), CTX (default 4096), extra server flags after the mode.
set -euo pipefail
ENGINE=/home/everett/llama.cpp-build
export CUDA_HOME=${CUDA_HOME:-$HOME/miniconda3/envs/cudabuild}
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
MODE=exact
if [ "${1:-}" = fast ] || [ "${1:-}" = exact ]; then MODE=$1; shift; fi

case "${MODEL:-qwen3-30b}" in
    qwen3-30b)
        FILE=$HOME/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
        NCMOE=48; SLOTS=56; FAST_OK=1
        # byte-priced per-layer slot counts, fitted to this model (artifacts/p7_slot_allocation.json)
        export LLAMA_MOE_SLOT_PROFILE="0:88,1:88,3:72,5:64,6:48,7:40,10:72,11:64,12:64,13:64,14:48,15:48,18:40,19:40,20:40,23:64,24:64,25:64,27:48,30:40,31:40,32:40,34:64,35:64,36:64,37:64,40:48,43:40,44:48,45:48,46:48"
        export LLAMA_MOE_ADMIT=window
        ;;
    qwen36)
        FILE=$HOME/.lmstudio/models/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf
        NCMOE=41; SLOTS=96; FAST_OK=1
        # the admission gate measured 0.0 on this model, so it is left off
        ;;
    *)
        echo "MODEL must be qwen3-30b or qwen36 (run experiments/gguf_preflight.py on anything else first)" >&2
        exit 1
        ;;
esac
export LLAMA_MOE_BATCH_TABLES=1

if [ "$MODE" = fast ]; then
    if [ "$FAST_OK" != 1 ] && [ "${FORCE_FAST:-0}" != 1 ]; then
        echo "'fast' changes model output and its quality cost has only been measured on Qwen3-30B-A3B." >&2
        echo "Run experiments/q1_cache_prior_quality.py on this model first, or set FORCE_FAST=1." >&2
        exit 1
    fi
    export LLAMA_MOE_CACHE_PRIOR=${DELTA:-0.01}
fi

cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
if [ "$cap" -gt 175 ]; then
    echo "GPU power limit is ${cap} W. This card falls off the bus at 250 W; run: sudo nvidia-smi -pl 175" >&2
    exit 1
fi
busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F, '$2+0 > 512' | wc -l)
if [ "$busy" -ne 0 ]; then
    echo "another model is already on the GPU:" >&2
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader >&2
    exit 1
fi
[ -f "$FILE" ] || { echo "model file not found: $FILE" >&2; exit 1; }

echo "serve.sh: $MODE, $(basename "$FILE"), $SLOTS slots a layer, port ${PORT:-8080}" >&2
exec "$ENGINE/build/bin/llama-server" -m "$FILE" -ngl 999 -ncmoe "$NCMOE" --moe-expert-cache "$SLOTS" \
    --no-mmap -c "${CTX:-4096}" -fa on --host 127.0.0.1 --port "${PORT:-8080}" "$@"
