"""Sidecar soak: decode at least 100,000 tokens through one sidecar server; zero channel errors.

Cycles the real prompts at 512 tokens each (end-of-text ignored) until the
decode-token total passes SOAK_TOKENS (default 100000). Fails on any request
error, any 'sidecar' error line in the server log, or a decode error.
"""
import json
import os
from pathlib import Path
import time

import x3_fused_bench as fused
import r3_real_bench as bench
import requests

TARGET = int(os.environ.get("SOAK_TOKENS", "100000"))
# x7_stream_soak.py reuses this with the streaming environment
ENV = {"LLAMA_MOE_FUSED_CPU": "1", "LLAMA_MOE_SIDECAR": "1"}
LABEL = "SIDECAR-SOAK"
BANNER = "sidecar ENABLED"
TAG = "x4_soak"


def main():
    label = LABEL
    bench.EXPECT[label] = BANNER
    out = {"target_tokens": TARGET, "requests": [], "completed": False}
    path = fused.ROOT / "artifacts" / f"{TAG}_{time.time_ns()}.json"
    p, vram = bench.launch(label, bench.BASE56, dict(ENV))
    try:
        records = [r for r in bench.select(bench.load_prompts(), bench.N_BENCH)
                   if bench.tokenize(r["text"]) <= bench.CTX - 512 - 64]
        total, k, t0 = 0, 0, time.time()
        while total < TARGET:
            rec = records[k % len(records)]
            r = requests.post(f"http://127.0.0.1:{bench.PORT}/completion",
                              json={"prompt": rec["text"], "n_predict": 512, "temperature": 0, "top_k": 1,
                                    "ignore_eos": True, "cache_prompt": False}, timeout=600)
            r.raise_for_status()
            t = r.json()["timings"]
            total += t["predicted_n"]
            out["requests"].append({"id": rec["id"], "n": t["predicted_n"], "tps": t["predicted_per_second"]})
            k += 1
            if k % 20 == 0:
                print(f"{total} tokens, {k} requests, {time.time() - t0:.0f} s", flush=True)
        log = Path(f"/tmp/r3-{label}.log").read_text(errors="replace")
        bad = [l for l in log.splitlines() if ("sidecar" in l and ("error" in l or "fail" in l or "did not drain" in l))
               or "MoE sidecar exchange failed" in l
               or ("moe-cache: stream" in l and ("error" in l or "push failed" in l or "FAILED" in l
                                                 or "failed=1" in l or "did not stop" in l))]
        out.update(total_tokens=total, vram=vram, channel_errors=len(bad), error_lines=bad[:20],
                   mean_tps=sum(x["tps"] for x in out["requests"]) / len(out["requests"]),
                   sidecar_stats=[l for l in log.splitlines() if "moe-cache: sidecar " in l and "records" in l][-1:],
                   completed=True, gate_passed=len(bad) == 0 and total >= TARGET)
        print(json.dumps({k: out[k] for k in ("total_tokens", "channel_errors", "mean_tps", "gate_passed")}), flush=True)
    finally:
        bench.stop(p)
        path.write_text(json.dumps(out, indent=2))
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
