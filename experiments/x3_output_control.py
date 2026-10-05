"""Repeat identical baseline arms on the prompts that diverged in the paired run."""
import json
from pathlib import Path
import time

import x3_fused_bench as fused
import r3_real_bench as bench

IDS = {"github_issues:00767", "no_robots:00385", "no_robots:00535", "se_decision:00435"}


def main():
    prompts = {r["id"]: r for r in bench.load_prompts() if r["id"] in IDS}
    out = {"binary": bench.BIN, "model": bench.MODEL, "runs": []}
    path = fused.ROOT / "artifacts" / f"x3_output_control_{time.time_ns()}.json"
    try:
        for repeat in range(3):
            label = f"OUTPUT-CONTROL-{repeat}"
            bench.EXPECT[label] = "fused CPU rows DISABLED"
            p, vram = bench.launch(label, bench.BASE56, {"LLAMA_MOE_FUSED_CPU": "0"})
            try:
                fused.completion(bench.CONTROL, 32)
                rows = []
                for pid in sorted(prompts):
                    rows.append({"id": pid, **fused.completion(prompts[pid]["text"], 200)})
                out["runs"].append({"repeat": repeat, "vram": vram, "outputs": rows})
                print(f"baseline repeat {repeat}: " + str([(r["id"], r["output_sha256"][:12]) for r in rows]), flush=True)
            finally:
                bench.stop(p)
        hashes = {pid: {r["output_sha256"] for run in out["runs"] for r in run["outputs"] if r["id"] == pid} for pid in IDS}
        out["baseline_unstable"] = {pid: len(values) for pid, values in hashes.items() if len(values) > 1}
        print(json.dumps(out["baseline_unstable"], indent=2), flush=True)
    finally:
        path.write_text(json.dumps(out, indent=2))
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
