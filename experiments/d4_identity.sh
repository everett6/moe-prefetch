#!/usr/bin/env bash
# D4 criterion 2: byte-identical greedy output, cache off vs cache on.
#
# Prefetching changes WHERE weights live, never which experts are selected or
# what is computed, so any divergence is a bug rather than a trade-off. That
# makes this a stricter test than n-gram speculation can pass -- which is why
# speculation is off here.
#
# It matters more for the early-issue path than for the shipped one, because
# early issue writes expert data into cache slots while a graph is running. The
# claim is that this is safe: a slot is only written while it is in no table, so
# no operation in the graph can reach it, and the tables themselves still change
# only at the step boundary. If that reasoning is wrong, the output diverges.
set -u
export CUDA_HOME=$HOME/miniconda3/envs/cudabuild
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
BIN=/home/everett/llama.cpp-build/build/bin/llama-server
MODEL=/home/everett/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
PRED=/home/everett/moe-prefetch/models/predictor-hostfree.bin
PORT=8099
OUTDIR=/home/everett/moe-prefetch/state/d4
mkdir -p "$OUTDIR"
PROMPTS=(
  "Explain how a B-tree index works in a relational database."
  "Write a Python function that merges two sorted lists and explain its complexity."
  "Summarise the causes of the 1929 stock market crash."
  "Prove that the square root of two is irrational."
)
capture() {
  local label="$1" slots="$2"; shift 2
  for _ in $(seq 1 60); do curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || break; sleep 1; done
  env "$@" "$BIN" -m "$MODEL" -ngl 999 -ncmoe 48 -c 4096 --port "$PORT" --host 127.0.0.1 \
      --no-webui -fa on --moe-expert-cache "$slots" > "/tmp/d4-$label.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 600); do curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q "^$pid," || {
      echo "$label: not our server"; kill $pid 2>/dev/null; return 1; }
  : > "$OUTDIR/$label.txt"
  local i=0
  for p in "${PROMPTS[@]}"; do
    i=$((i+1))
    local pj
    pj=$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$p")
    curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
      -d "{\"prompt\":$pj,\"n_predict\":180,\"temperature\":0,\"top_k\":1,\"seed\":0,\"cache_prompt\":false}" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["content"])' >> "$OUTDIR/$label.txt"
    echo "----8<---- $i" >> "$OUTDIR/$label.txt"
  done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 4
}
capture "off"        0
capture "off2"       0                              # the control: same config twice
capture "shipped"   72 LLAMA_MOE_EARLY_ISSUE=0
capture "early"     72 LLAMA_MOE_EARLY_ISSUE=1
capture "early_p3"  72 LLAMA_MOE_EARLY_ISSUE=1 LLAMA_MOE_PREDICTOR=$PRED LLAMA_MOE_PREDICT_TOP=3
echo
for f in off2 shipped early early_p3; do
  if cmp -s "$OUTDIR/off.txt" "$OUTDIR/$f.txt"; then
    echo "$f: BYTE-IDENTICAL to cache-off"
  else
    echo "$f: DIVERGES from cache-off  ($(cmp -l "$OUTDIR/off.txt" "$OUTDIR/$f.txt" 2>/dev/null | wc -l) bytes differ)"
  fi
done
wc -c "$OUTDIR"/*.txt
