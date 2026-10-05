"""Paired benchmark: fused CPU path (arm 0) against the sidecar (arm 1), live cache.

Same protocol as x3_fused_bench.py: 3 rounds x 23 real prompts x 200 tokens,
fresh server per arm, arm order alternating. Writes artifacts/x4_sidecar_*.json.
"""
import x3_fused_bench as bench

bench.ARMS = {0: ({"LLAMA_MOE_FUSED_CPU": "1"}, "fused CPU rows ENABLED"),
              1: ({"LLAMA_MOE_FUSED_CPU": "1", "LLAMA_MOE_SIDECAR": "1"}, "sidecar ENABLED")}
bench.TAG = "x4_sidecar"
bench.LABEL = "SIDECAR"

if __name__ == "__main__":
    bench.main()
