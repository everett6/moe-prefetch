"""J5: streaming is exact. RECORD runs sidecar + team + streaming on a live
cache; its recorded map gives each token the experts that streamed in before
their layer read its table, in the slots they used. REPLAY runs sidecar + team
without streaming on that map. Gate: zero unequal logits, routing identical,
and RECORD used at least one streamed expert in time (else the test is vacuous).
"""
import json
import sys
from pathlib import Path

import x3_logits_replay as replay
from x7_stream_common import BASE, STREAM, counters

ROOT = Path(__file__).resolve().parents[1]
ARMS = (("record", STREAM, False),
        ("replay", {k: v for k, v in BASE.items() if k != "LLAMA_MOE_EARLY_ISSUE"}, True))

if __name__ == "__main__":
    replay.main(ARMS, "x7_stream_logits")
    root = max((ROOT / "artifacts").glob("x7_stream_logits_*"), key=lambda p: p.stat().st_mtime)
    result = json.loads((root / "result.json").read_text())
    used = [counters(p) for p in sorted(root.glob("*.log-record.txt"))]
    in_time = sum(c["in_time"] for c in used if c)
    verdict = {"replay_identical": bool(result.get("gate_passed")), "record_in_time": in_time,
               "record_counters": used, "gate_passed": bool(result.get("gate_passed")) and in_time >= 1}
    (root / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps({k: verdict[k] for k in ("replay_identical", "record_in_time", "gate_passed")}), flush=True)
    sys.exit(0)
