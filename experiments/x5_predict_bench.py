"""Does prediction pay on the sidecar? Paired: sidecar + early issue (arm 0)
against the same plus the trained next-layer predictor with in-token
publication (arm 1). Same protocol as x3/x4: 3 rounds x 23 prompts x 200
tokens, fresh server per arm, alternating order. Writes artifacts/x5_predict_*.json
and prints each arm's in-token counters from its server log.
"""
import re
from pathlib import Path

import x3_fused_bench as bench

ROOT = Path(__file__).resolve().parents[1]
BASE = {"LLAMA_MOE_FUSED_CPU": "1", "LLAMA_MOE_SIDECAR": "1", "LLAMA_MOE_EARLY_ISSUE": "1"}
PRED = dict(BASE, LLAMA_MOE_PREDICTOR=str(ROOT / "models/predictor-real.bin"), LLAMA_MOE_PREDICT_TOP="3",
            LLAMA_MOE_SPEC_PROBATION="1", LLAMA_MOE_PREDICT_INTOKEN="1")
bench.ARMS = {0: (BASE, "sidecar ENABLED"), 1: (PRED, "in-token prediction ENABLED")}
bench.TAG = "x5_predict"
bench.LABEL = "PREDICT"

if __name__ == "__main__":
    try:
        bench.main()
    finally:
        for arm in (0, 1):
            log = Path(f"/tmp/r3-PREDICT-{arm}.log")
            if log.exists():
                lines = [l for l in log.read_text(errors="replace").splitlines()
                         if re.search(r"in-token predictions|sidecar \d+ records|hit=", l)]
                print(f"arm {arm} (last server):", *lines[-3:], sep="\n  ", flush=True)
