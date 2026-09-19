#!/usr/bin/env bash
# E4: re-tune the operating point now that the cache actually works.
#
# Every configuration constant in this project was chosen against a cache that
# was collapsing: 66 slots/layer came from a VRAM budget computed in simulation,
# -ncmoe 48 from wanting all experts host-resident, insert rate from the default.
# With residency now at 64/66 and the hit rate at 91%, none of those has been
# checked against a working system.
#
# Two obvious candidates. VRAM at cache 66 measures 10,348 MiB of 12,227, so
# there is room for roughly a dozen more slots per layer -- and slots are hit
# rate. And -ncmoe 48 puts every expert layer on the host; with a working cache
# a different split may be better.
#
# Same guards as d3c_bench.sh: port free, PID verified, VRAM checked.
set -u
export CUDA_HOME=$HOME/miniconda3/envs/cudabuild
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
BIN=/home/everett/llama.cpp-build/build/bin/llama-server
MODEL=/home/everett/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
PORT=${PORT:-8099}
REPS=${REPS:-3}
NPRED=${NPRED:-200}
PROMPT="Write a detailed explanation of how a B-tree index works in a relational database, including insertion, node splitting, deletion and range scans."
PJSON=$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$PROMPT")

run() {   # run <label> <ncmoe> <slots> [extra args / ENV=VAL]
  local label="$1" ncmoe="$2" slots="$3"; shift 3
  local envs=() cli=()
  for a in "$@"; do case "$a" in *=*) envs+=("$a");; *) cli+=("$a");; esac; done
  for _ in $(seq 1 90); do curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || break; sleep 1; done
  env "${envs[@]}" "$BIN" -m "$MODEL" -ngl 999 -ncmoe "$ncmoe" -c 4096 --port "$PORT" \
      --host 127.0.0.1 --no-webui -fa on --moe-expert-cache "$slots" "${cli[@]}" \
      > "/tmp/e4-$label.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 600); do
    curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && break
    kill -0 $pid 2>/dev/null || break; sleep 1
  done
  if ! kill -0 $pid 2>/dev/null; then
    printf '%-26s %-10s OOM/FAILED\n' "$label" "-"; tail -2 "/tmp/e4-$label.log" | sed 's/^/    /'; return
  fi
  local vram
  vram=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | awk -F', ' -v p=$pid '$1==p{print $2}')
  if [ -z "$vram" ]; then printf '%-26s not our server\n' "$label"; kill $pid 2>/dev/null; return; fi
  local out="" r best=0
  for i in $(seq 0 "$REPS"); do
    r=$(curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
      -d "{\"prompt\":$PJSON,\"n_predict\":$NPRED,\"temperature\":0,\"top_k\":1,\"seed\":0,\"cache_prompt\":false}" \
      | python3 -c 'import json,sys
try: print(f"{json.load(sys.stdin)["timings"]["predicted_per_second"]:.1f}")
except Exception: print("0")')
    [ "$i" -gt 0 ] && { out="$out $r"; awk -v a="$r" -v b="$best" 'BEGIN{exit !(a>b)}' && best=$r; }
  done
  local mean sd
  read mean sd < <(python3 -c "
import sys,statistics as st
v=[float(x) for x in '$out'.split()]
print(f'{st.mean(v):.1f}', f'{(st.stdev(v) if len(v)>1 else 0):.1f}')")
  printf '%-26s %-10s mean %6s  sd %4s  best %6s   runs:%s\n' "$label" "$vram" "$mean" "$sd" "$best" "$out"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 4
}

echo "config                     VRAM       throughput (tok/s)"
echo "--- reference points ---"
run "ai2-shipped-ncmoe22"     22  0
run "ncmoe48-nocache"         48  0
echo "--- cache size, early issue on ---"
for s in 56 66 72 78 84; do run "ncmoe48-cache$s-early" 48 "$s" LLAMA_MOE_EARLY_ISSUE=1; done
echo "--- insert rate ---"
run "ncmoe48-cache66-ins1"    48 66 --moe-expert-cache-inserts 1
run "ncmoe48-cache66-ins1-early" 48 66 LLAMA_MOE_EARLY_ISSUE=1 --moe-expert-cache-inserts 1
run "ncmoe48-cache66-ins3-early" 48 66 LLAMA_MOE_EARLY_ISSUE=1 --moe-expert-cache-inserts 3
echo "--- split ---"
for n in 40 44 46; do run "ncmoe$n-cache66-early" "$n" 66 LLAMA_MOE_EARLY_ISSUE=1; done
