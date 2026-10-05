"""J6 (and J4): paired benchmark, sidecar + team + early issue (arm 0) against
the same plus just-in-time streaming (arm 1). 3 rounds x 23 prompts x 200
tokens, fresh server per arm, alternating order. Each stream server's log is
kept, and its counters pooled: gate >= +6 tok/s paired, >= 50% of would-be
misses served from VRAM, >= 45% of landed copies used in time, no sidecar
error. J4 (P11) reads the lead times from the same logs.
"""
import json
import shutil
import time
from pathlib import Path

import r3_real_bench as r3
import x3_fused_bench as bench
from x7_stream_common import BASE, STREAM, counters, summarize

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "artifacts" / f"x7_stream_logs_{time.time_ns()}"
bench.ARMS = {0: (BASE, "sidecar team ENABLED"), 1: (STREAM, "streaming ENABLED")}
bench.TAG = "x7_stream"
bench.LABEL = "STREAM"

_stop = r3.stop
_n = [0]


def stop_and_keep_log(p):
    _stop(p)   # the server has exited: its final counters are in the log
    for arm in (0, 1):
        log = Path(f"/tmp/r3-STREAM-{arm}.log")
        if log.exists() and log.stat().st_mtime > LOGS.stat().st_mtime:
            shutil.copy(log, LOGS / f"{_n[0]:02d}-arm{arm}.log")
    _n[0] += 1


if __name__ == "__main__":
    LOGS.mkdir(parents=True)
    r3.stop = stop_and_keep_log
    try:
        bench.main()
    finally:
        rows = [c for c in (counters(p) for p in sorted(LOGS.glob("*-arm1.log"))) if c]
        if rows:
            pooled = summarize(rows)
            pooled["counters_gate"] = pooled["served_from_vram"] >= 0.50 and pooled["copies_used"] >= 0.45 and pooled["failed"] == 0
            (LOGS / "counters.json").write_text(json.dumps({"servers": rows, "pooled": pooled}, indent=2))
            print(json.dumps(pooled), flush=True)
