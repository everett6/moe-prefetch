#!/usr/bin/env bash
# Grade answers under cache-off and under the final configuration.
# Byte-identity is unreachable with this cache design (it sums two mul_mat_id
# chains, so float addition reorders); this checks the divergence is numerical
# noise rather than damage.
set -u
export CUDA_HOME=$HOME/miniconda3/envs/cudabuild
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
BIN=/home/everett/llama.cpp-build/build/bin/llama-server
MODEL=/home/everett/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
run() {
  local label="$1" slots="$2"; shift 2
  for _ in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:8099/health >/dev/null 2>&1 || break; sleep 1; done
  env "$@" $BIN -m "$MODEL" -ngl 999 -ncmoe 48 -c 4096 --port 8099 --host 127.0.0.1 \
      --no-webui -fa on --no-mmap --moe-expert-cache "$slots" > /tmp/qual-$label.log 2>&1 &
  local pid=$!
  for _ in $(seq 1 900); do curl -s http://127.0.0.1:8099/health 2>/dev/null | grep -q '"status":"ok"' && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
  python3 "$(dirname "$0")/d5_quality.py" "$label"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 4
}
run "cache_off" 0
run "final"    72 LLAMA_MOE_EARLY_ISSUE=1
