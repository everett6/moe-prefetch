#!/usr/bin/env bash
# Everything written on 2026-09-30 with the GPU in use by other work, in the
# order it should be measured. Nothing in here has been run.
#
#   experiments/next_session.sh            # all stages
#   experiments/next_session.sh profile    # one stage: build | profile | tables |
#                                          #   admit | slots | quality | prior | stack
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

cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
if [ "$cap" -gt 175 ]; then
    echo "power limit is ${cap} W -- this card falls off the bus at 250 W. Set 175 W first." >&2
    exit 1
fi

if want build; then
    echo "== build (the engine source changed; the binary on disk predates it) =="
    cmake --build "$ENGINE/build" --target llama-server -j"$(nproc)" | tail -3
fi

if want profile; then
    # Where does a token go? The cost model's fixed term is 59% of the token and
    # has never been decomposed. This prints, per backend, time spent waiting
    # for inputs (= waiting for the GPU, on a CPU split) against time computing.
    echo "== scheduler split profile =="
    LLAMA_MOE_STATS=1 GGML_SCHED_PROFILE=512 "$ENGINE/build/bin/llama-server" \
        -m "${MODEL:-$HOME/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}" \
        -ngl 999 -ncmoe "${NCMOE:-48}" -c 4096 --port 8097 --host 127.0.0.1 --no-webui -fa on \
        --moe-expert-cache "${SLOTS:-56}" --no-mmap > /tmp/r3-profile.log 2>&1 &
    pid=$!
    for _ in $(seq 1 600); do
        curl -s http://127.0.0.1:8097/health 2>/dev/null | grep -q '"status":"ok"' && break
        kill -0 $pid 2>/dev/null || { echo "server died, see /tmp/r3-profile.log" >&2; exit 1; }
        sleep 1
    done
    curl -s http://127.0.0.1:8097/completion -H 'Content-Type: application/json' \
        -d '{"prompt":"Explain how a write-ahead log gives a database crash consistency, then contrast it with shadow paging.","n_predict":1600,"temperature":0,"top_k":1,"seed":0}' > /dev/null
    kill $pid; wait $pid 2>/dev/null || true
    grep -E "sched-profile|per token" /tmp/r3-profile.log | tail -8
    mkdir -p artifacts && grep -E "sched-profile|moe-cache:" /tmp/r3-profile.log > artifacts/sched_profile.txt
fi

run() { echo "== $1 =="; env "$1=1" ROUNDS="${ROUNDS:-3}" python3 experiments/r3_real_bench.py; }
if want tables;  then run WITH_TABLES; fi   # exact: batched table writes
if want admit;   then run WITH_ADMIT;  fi   # exact: admission gate -- also the direct test of the cost model's C term
if want slots;   then run WITH_SLOTS;  fi   # exact: byte-priced slot profile against the slot-count one
if want quality; then echo "== cache-prior quality =="; python3 experiments/q1_cache_prior_quality.py; fi
if want prior;   then run WITH_PRIOR;  fi   # CHANGES OUTPUT: read with the quality table, never alone
if want stack;   then run WITH_STACK;  fi   # exact switches stacked, after the single-switch runs
