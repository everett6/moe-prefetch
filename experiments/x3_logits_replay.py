"""Record a live cache, replay its exact placements, and compare every logit.

Runs only when the existing GPU occupancy guard permits testing. Produces
unique logs, routing traces, mappings, token IDs, and a JSON result. Logits
are kept only during each paired comparison (about 122 MB at 200 tokens).
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import x3_fused_bench as fused
import r3_real_bench as bench
from x3_output_control import IDS


def main():
    runner = fused.ROOT / "cpp/build/moe-logits"
    if not runner.is_file():
        raise RuntimeError("build cpp/moe-logits.cpp first")
    root = fused.ROOT / "artifacts" / f"x3_logits_{time.time_ns()}"
    root.mkdir()
    result = {"model": bench.MODEL, "n_predict": bench.N_PREDICT,
              "deterministic_publication": fused.DETERMINISTIC, "pairs": [], "completed": False}
    prompts = {r["id"]: r for r in bench.load_prompts() if r["id"] in IDS}
    try:
        with tempfile.TemporaryDirectory(prefix="moe-logits-") as tmp:
            for index, pid in enumerate(sorted(prompts)):
                prefix = root / str(index)
                prompt = prefix.with_suffix(".prompt.txt")
                prompt.write_text(prompts[pid]["text"])
                mapping = prefix.with_suffix(".map.txt")
                logits = Path(tmp) / "logits.bin"
                tokens = prefix.with_suffix(".tokens.bin")
                row = {"id": pid}
                for enabled in (0, 1):
                    bench.gpu_guard(f"LOGITS-{enabled}")
                    env = dict(bench.ENV)
                    for name in ("MOE_LOGITS_OUT", "MOE_LOGITS_REFERENCE", "MOE_FORCE_TOKENS", "MOE_TOKENS_OUT"):
                        env.pop(name, None)
                    env["LLAMA_MOE_FUSED_CPU"] = str(enabled)
                    trace = prefix.with_suffix(f".routes-{enabled}.txt")
                    env["LLAMA_MOE_ROUTE_TRACE"] = str(trace)
                    if enabled:
                        env.update(LLAMA_MOE_MAP_REPLAY=str(mapping), MOE_LOGITS_REFERENCE=str(logits),
                                   MOE_FORCE_TOKENS=str(tokens))
                    else:
                        env.update(LLAMA_MOE_MAP_RECORD=str(mapping), MOE_LOGITS_OUT=str(logits),
                                   MOE_TOKENS_OUT=str(tokens))
                    args = [str(runner), "-m", bench.MODEL, "-ngl", "999", "-c", str(bench.CTX),
                            "-t", "16", "-fa", "on", "-f", str(prompt), "-n", str(bench.N_PREDICT)] + bench.BASE56
                    with prefix.with_suffix(f".log-{enabled}.txt").open("w") as log:
                        proc = subprocess.run(args, env=env, stdout=subprocess.PIPE, stderr=log,
                                              text=True, timeout=900)
                    if proc.returncode not in (0, 1) or not proc.stdout.strip().endswith("}"):
                        raise RuntimeError(f"logits runner failed: {prefix}, arm {enabled}, exit {proc.returncode}")
                    row[f"fused_{enabled}"] = json.loads(proc.stdout.strip().splitlines()[-1])
                    print(f"{pid} fused={enabled}: {row[f'fused_{enabled}']}", flush=True)
                row["identical_routes_and_slots"] = prefix.with_suffix(".routes-0.txt").read_bytes() == prefix.with_suffix(".routes-1.txt").read_bytes()
                result["pairs"].append(row)
        result["completed"] = True
        result["all_logits_identical"] = all(r["fused_1"]["unequal_logits"] == 0 and r["identical_routes_and_slots"] for r in result["pairs"])
    finally:
        (root / "result.json").write_text(json.dumps(result, indent=2))
        print(f"saved {root / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
