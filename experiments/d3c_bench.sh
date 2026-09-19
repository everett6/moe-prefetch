#!/usr/bin/env bash
# D3c: measure early issue and the predictor against the shipped cache.
#
# The harness checks three things before it believes a number, because the first
# version of it did not and produced a 3x difference that was not real:
#
#   port free      a server that cannot bind exits, and curl then measures
#                  WHICHEVER server is still listening -- silently attributing
#                  the previous config's throughput to this one. AI2 hit exactly
#                  this and it cost a day.
#   right PID      the process we started is the process answering.
#   VRAM           libllama's "MoE expert cache enabled" line does not reach the
#                  server's stderr, so the only honest check that --moe-expert-cache
#                  did anything is the ~8.8 GB of slabs it allocates.
#
# Method otherwise as D1: llama-server, server-reported decode timings, -c 4096,
# 175 W cap, one discarded warm-up per instance.
set -u
export CUDA_HOME=$HOME/miniconda3/envs/cudabuild
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
BIN=/home/everett/llama.cpp-build/build/bin/llama-server
MODEL=/home/everett/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
PRED=/home/everett/moe-prefetch/models/predictor-hostfree.bin
PORT=${PORT:-8099}
NPRED=${NPRED:-200}
REPS=${REPS:-3}
PROMPT="Write a detailed explanation of how a B-tree index works in a relational database, including insertion, node splitting, deletion and range scans."
PJSON=$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$PROMPT")

port_free() { ! curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

run() {            # run <label> <cache-slots> [ENV=VAL ...]
  local label="$1" slots="$2"; shift 2
  for _ in $(seq 1 60); do port_free && break; sleep 1; done
  if ! port_free; then
    echo "$label: PORT $PORT STILL HELD -- refusing to measure someone else's server"
    return
  fi
  env "$@" "$BIN" -m "$MODEL" -ngl 999 -ncmoe 48 -c 4096 --port "$PORT" \
      --host 127.0.0.1 --no-webui -fa on --moe-expert-cache "$slots" \
      > "/tmp/d3c-$label.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 600); do
    curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && break
    kill -0 $pid 2>/dev/null || break
    sleep 1
  done
  if ! kill -0 $pid 2>/dev/null; then
    echo "$label: SERVER DIED"; tail -3 "/tmp/d3c-$label.log"; return
  fi
  local vram
  vram=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | awk -F', ' -v p=$pid '$1==p{print $2}')
  if [ -z "$vram" ]; then
    echo "$label: pid $pid is not the process on the GPU -- not measuring"
    kill $pid 2>/dev/null; wait $pid 2>/dev/null; return
  fi
  local out="" r
  for i in $(seq 0 "$REPS"); do
    r=$(curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
      -d "{\"prompt\":$PJSON,\"n_predict\":$NPRED,\"temperature\":0,\"top_k\":1,\"seed\":0,\"cache_prompt\":false}" \
      | python3 -c 'import json,sys
try: print(f"{json.load(sys.stdin)["timings"]["predicted_per_second"]:.1f}")
except Exception: print("ERR")')
    [ "$i" -gt 0 ] && out="$out $r"
  done
  printf '%-24s %-10s %s\n' "$label" "$vram" "$out"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 4
}

printf '%-24s %-10s %s\n' "config" "VRAM" "tok/s per run"
for round in $(seq 1 "${ROUNDS:-2}"); do
  run "no-cache"            0
  run "shipped"            66
  run "early"              66 LLAMA_MOE_EARLY_ISSUE=1
  run "early+pred3"        66 LLAMA_MOE_EARLY_ISSUE=1 LLAMA_MOE_PREDICTOR=$PRED LLAMA_MOE_PREDICT_TOP=3
done
