"""Compare old and fused rows end to end with identical, fixed cache mappings.

This is a correctness diagnostic, not a performance benchmark. It preloads a
synthetic hot set through the existing heat-file loader and disables uploads.
"""
import json
from pathlib import Path
import tempfile
import time

import x3_fused_bench as fused
import r3_real_bench as bench
from x3_output_control import IDS


def main():
    if "Qwen3-30B-A3B-Instruct-2507" not in bench.MODEL:
        raise RuntimeError("this synthetic heat file describes only the local Qwen3-30B model")
    prompts = {r["id"]: r for r in bench.load_prompts() if r["id"] in IDS}
    out = {"binary": bench.BIN, "model": bench.MODEL, "runs": [], "completed": False}
    path = fused.ROOT / "artifacts" / f"x3_frozen_control_{time.time_ns()}.json"
    with tempfile.TemporaryDirectory(prefix="moe-frozen-") as tmp:
        heat = Path(tmp) / "heat.txt"
        heat.write_text("moe-heat 1 48 128\n" + "".join(f"{layer} {expert} {256-expert}\n" for layer in range(48) for expert in range(128)))
        try:
            for enabled in (0, 1):
                label = f"FROZEN-{enabled}"
                bench.EXPECT[label] = "cache mapping FROZEN"
                p, vram = bench.launch(label, bench.BASE56, {"LLAMA_MOE_FUSED_CPU": str(enabled),
                                                           "LLAMA_MOE_CACHE_FREEZE": "1", "LLAMA_MOE_HEAT_FILE": str(heat)})
                try:
                    log = Path(f"/tmp/r3-{label}.log").read_text()
                    if "placed before the first token" not in log:
                        raise RuntimeError("cache warm start did not engage")
                    fused.completion(bench.CONTROL, 32)
                    rows = [{"id": pid, **fused.completion(prompts[pid]["text"], 200)} for pid in sorted(prompts)]
                    out["runs"].append({"fused": enabled, "vram": vram, "outputs": rows})
                    print(f"frozen fused={enabled}: " + str([(r["id"], r["output_sha256"][:12]) for r in rows]), flush=True)
                finally:
                    bench.stop(p)
            old = {r["id"]: r["output_sha256"] for r in out["runs"][0]["outputs"]}
            out["output_mismatches"] = [r["id"] for r in out["runs"][1]["outputs"] if r["output_sha256"] != old[r["id"]]]
            out["completed"] = True
            print(json.dumps(out["output_mismatches"]), flush=True)
        finally:
            path.write_text(json.dumps(out, indent=2))
            print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
