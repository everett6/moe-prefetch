#!/usr/bin/env bash
# The measurement session for patch 0007. Run in full on 2026-09-30; results are
# in docs/superpowers/specs/2026-10-01-graph-resident-decode-design.md, section 2.
#
#   experiments/next_session.sh            # all stages
#   experiments/next_session.sh profile    # one stage: build | profile | tables |
#                                          #   admit | slots | quality | prior | stack | omp
#
# Every stage refuses to start if another model is on the GPU, and every arm
# refuses to be measured if its switch did not print its banner (stale binary).
set -euo pipefail
cd "$(dirname "$0")/.."
ENGINE=/home/everett/llama.cpp-build
export CUDA_HOME=${CUDA_HOME:-$HOME/miniconda3/envs/cudabuild}
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
STAGE=${1:-all}
want() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }
MODEL=${MODEL:-$HOME/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}

cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
if [ "$cap" -gt 175 ]; then
    echo "power limit is ${cap} W -- this card falls off the bus at 250 W. Set 175 W first." >&2
    exit 1
fi

if want build; then
    echo "== build =="
    cmake --build "$ENGINE/build" --target llama-server -j"$(nproc)" | tail -3
fi

if want profile; then
    # Where does a token go? As first written this stage printed nothing: it
    # asked for a report every 512 graphs and the model stopped at 467 tokens.
    # y1_token_anatomy.sh ignores end-of-stream and fails if no block came out.
    echo "== where a token goes =="
    experiments/y1_token_anatomy.sh q4-cache56 "$MODEL" -ncmoe "${NCMOE:-48}" \
        --moe-expert-cache "${SLOTS:-56}" --no-mmap
fi

run() { echo "== $1 =="; env "$1=1" ROUNDS="${ROUNDS:-3}" python3 experiments/r3_real_bench.py; }
if want tables;  then run WITH_TABLES; fi   # exact: batched table writes                 measured +0.66
if want admit;   then run WITH_ADMIT;  fi   # exact: admission gates                      heat -3.00, 2-in-8 +3.88
if want slots;   then run WITH_SLOTS;  fi   # exact: slot profiles                        count +2.95, bytes +2.56
if want quality; then echo "== cache-prior quality =="; python3 experiments/q1_cache_prior_quality.py; fi
if want prior;   then run WITH_PRIOR;  fi   # CHANGES OUTPUT: read with the quality table  0.01: +16.4, ppl unchanged
if want stack;   then run WITH_STACK;  fi   # exact switches that measured positive       +6.41
if [ "$STAGE" = omp ]; then run WITH_OMP; fi  # exact, not yet run: OpenMP wait policy, placement, 8 threads
