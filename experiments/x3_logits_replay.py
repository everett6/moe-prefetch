"""Record a live cache, replay its exact placements, and compare every logit.

Three arms per prompt: RECORD (original CPU chain, live cache, writes the
placement map, logits and greedy tokens), REPLAY-0 (original chain, replayed
placements, teacher-forced) and REPLAY-1 (fused chain, same). REPLAY-0 is the
control: if it differs from RECORD, replay itself changes the arithmetic and
REPLAY-1's comparison says nothing about the fused op.

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
                for arm, enabled, replay in (("record", 0, False), ("replay_0", 0, True), ("replay_1", 1, True)):
                    bench.gpu_guard(f"LOGITS-{arm}")
                    env = dict(bench.ENV)
                    for name in ("MOE_LOGITS_OUT", "MOE_LOGITS_REFERENCE", "MOE_FORCE_TOKENS", "MOE_TOKENS_OUT"):
                        env.pop(name, None)
                    env["LLAMA_MOE_FUSED_CPU"] = str(enabled)
                    trace = prefix.with_suffix(f".routes-{arm}.txt")
                    env["LLAMA_MOE_ROUTE_TRACE"] = str(trace)
                    if replay:
                        env.update(LLAMA_MOE_MAP_REPLAY=str(mapping), MOE_LOGITS_REFERENCE=str(logits),
                                   MOE_FORCE_TOKENS=str(tokens))
                    else:
                        env.update(LLAMA_MOE_MAP_RECORD=str(mapping), MOE_LOGITS_OUT=str(logits),
                                   MOE_TOKENS_OUT=str(tokens))
                    args = [str(runner), "-m", bench.MODEL, "-ngl", "999", "-c", str(bench.CTX),
                            "-t", "16", "-fa", "on", "-f", str(prompt), "-n", str(bench.N_PREDICT)] + bench.BASE56
                    with prefix.with_suffix(f".log-{arm}.txt").open("w") as log:
                        proc = subprocess.run(args, env=env, stdout=subprocess.PIPE, stderr=log,
                                              text=True, timeout=900)
                    if proc.returncode not in (0, 1) or not proc.stdout.strip().endswith("}"):
                        raise RuntimeError(f"logits runner failed: {prefix}, arm {arm}, exit {proc.returncode}")
                    row[arm] = json.loads(proc.stdout.strip().splitlines()[-1])
                    print(f"{pid} {arm}: {row[arm]}", flush=True)
                routes = {arm: prefix.with_suffix(f".routes-{arm}.txt").read_bytes()
                          for arm in ("record", "replay_0", "replay_1")}
                row["routes_replay_0_match_record"] = routes["replay_0"] == routes["record"]
                row["routes_replay_1_match_record"] = routes["replay_1"] == routes["record"]
                result["pairs"].append(row)
        result["completed"] = True
        result["replay_control_identical"] = all(
            r["replay_0"]["unequal_logits"] == 0 and r["routes_replay_0_match_record"] for r in result["pairs"])
        result["fused_identical"] = all(
            r["replay_1"]["unequal_logits"] == 0 and r["routes_replay_1_match_record"] for r in result["pairs"])
        result["gate_passed"] = result["replay_control_identical"] and result["fused_identical"]
        print(json.dumps({k: result[k] for k in ("replay_control_identical", "fused_identical", "gate_passed")}), flush=True)
    finally:
        (root / "result.json").write_text(json.dumps(result, indent=2))
        print(f"saved {root / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
