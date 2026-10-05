"""J2: paired benchmark, sidecar (arm 0) against sidecar + persistent team (arm 1).

Same protocol as x3/x4: 3 rounds x 23 real prompts x 200 tokens, fresh server
per arm, alternating order. Gate (docs/STREAMING-IMPLEMENTATION.md): team arm
>= +4 tok/s paired and no sidecar error. Writes artifacts/x6_team_*.json.
"""
import x3_fused_bench as bench

SIDECAR = {"LLAMA_MOE_FUSED_CPU": "1", "LLAMA_MOE_SIDECAR": "1"}
bench.ARMS = {0: (SIDECAR, "sidecar ENABLED"),
              1: (dict(SIDECAR, LLAMA_MOE_SIDECAR_TEAM="1"), "sidecar team ENABLED")}
bench.TAG = "x6_team"
bench.LABEL = "TEAM"

if __name__ == "__main__":
    bench.main()
