#!/usr/bin/env bash
# Where does one token go? Runs ONE server configuration with the scheduler's
# split profiler on, generates a fixed number of tokens, and keeps the profile.
#
#   experiments/y1_token_anatomy.sh LABEL MODEL [server args...]
#
#   LABEL   names the output: artifacts/anatomy_LABEL.txt
#   env     N_PREDICT (default 1024), PROFILE_EVERY (default 128), PORT (8097),
#           other variables pass through to the server (OMP_*, LLAMA_MOE_*)
#
# The first version of this (next_session.sh, stage "profile") asked for 1600
# tokens with a profile every 512 graphs; the model stopped at 467 and nothing
# was printed. This one ignores end-of-stream and fails if no block came out.
set -euo pipefail
cd "$(dirname "$0")/.."
ENGINE=/home/everett/llama.cpp-build
export CUDA_HOME=${CUDA_HOME:-$HOME/miniconda3/envs/cudabuild}
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
LABEL=$1; MODEL=$2; shift 2
PORT=${PORT:-8097}
N_PREDICT=${N_PREDICT:-1024}
EVERY=${PROFILE_EVERY:-128}
LOG=/tmp/anatomy-$LABEL.log
OUT=artifacts/anatomy_$LABEL.txt

cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
[ "$cap" -le 175 ] || { echo "power limit is ${cap} W; set 175 W first" >&2; exit 1; }
busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F, '$2+0 > 512' | wc -l)
if [ "$busy" -ne 0 ]; then
    echo "another model is on the GPU; not starting" >&2
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader >&2
    exit 1
fi
if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "port $PORT already answers; refusing to measure someone else's server" >&2
    exit 1
fi

LLAMA_MOE_STATS=1 GGML_SCHED_PROFILE=$EVERY "$ENGINE/build/bin/llama-server" -m "$MODEL" \
    -ngl 999 -c 4096 --port "$PORT" --host 127.0.0.1 --no-webui -fa on "$@" > "$LOG" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true' EXIT
ok=0
for _ in $(seq 1 900); do
    kill -0 $pid 2>/dev/null || { echo "server died; see $LOG" >&2; tail -5 "$LOG" >&2; exit 1; }
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then ok=1; break; fi
    sleep 1
done
[ "$ok" -eq 1 ] || { echo "server never became healthy" >&2; exit 1; }
vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
util_log=/tmp/anatomy-$LABEL.util
( for _ in $(seq 1 400); do
      nvidia-smi --query-gpu=utilization.gpu,clocks.current.graphics,power.draw --format=csv,noheader,nounits
      sleep 0.5
  done > "$util_log" 2>/dev/null ) &
upid=$!

curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d "{
  \"prompt\": \"Explain how a write-ahead log gives a database crash consistency, then contrast it with shadow paging, then describe how checkpoints bound recovery time.\",
  \"n_predict\": $N_PREDICT, \"temperature\": 0, \"top_k\": 1, \"seed\": 0, \"ignore_eos\": true, \"cache_prompt\": false}" \
  > /tmp/anatomy-$LABEL.json
kill $upid 2>/dev/null || true; wait $upid 2>/dev/null || true
kill $pid; wait $pid 2>/dev/null || true
trap - EXIT

mkdir -p artifacts
{
    echo "# $LABEL"
    echo "# model: $MODEL"
    echo "# args: -ngl 999 -c 4096 -fa on $*"
    echo "# vram after load: ${vram} MiB"
    awk -F, 'NR > 2 {u += $1; c += $2; p += $3; n++} END {if (n) printf "# during generation (%d samples): GPU busy %.0f%%, clock %.0f MHz, %.0f W\n", n, u / n, c / n, p / n}' "$util_log"
    grep -E "moe-cache: (expert weights|batched|admission gate|table writes|slot profile|steps=|per token)" "$LOG" | tail -6 || true
    python3 experiments/y1_profile_delta.py "$LOG" "/tmp/anatomy-$LABEL.json"
} | tee "$OUT"
