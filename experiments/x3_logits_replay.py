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


# (name, environment for the arm, replays the RECORD arm's placement map)
ARMS = (("record", {"LLAMA_MOE_FUSED_CPU": "0"}, False),
        ("replay_0", {"LLAMA_MOE_FUSED_CPU": "0"}, True),
        ("replay_1", {"LLAMA_MOE_FUSED_CPU": "1"}, True))


def main(arms=ARMS, tag="x3_logits"):
    runner = fused.ROOT / "cpp/build/moe-logits"
    if not runner.is_file():
        raise RuntimeError("build cpp/moe-logits.cpp first")
    root = fused.ROOT / "artifacts" / f"{tag}_{time.time_ns()}"
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
                for arm, arm_env, replay in arms:
                    bench.gpu_guard(f"LOGITS-{arm}")
                    env = dict(bench.ENV)
                    for name in ("MOE_LOGITS_OUT", "MOE_LOGITS_REFERENCE", "MOE_FORCE_TOKENS", "MOE_TOKENS_OUT"):
                        env.pop(name, None)
                    env.update(arm_env)
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
                record = prefix.with_suffix(".routes-record.txt").read_bytes()
                for arm, _, replay in arms:
                    if replay:
                        row[f"routes_{arm}_match_record"] = prefix.with_suffix(f".routes-{arm}.txt").read_bytes() == record
                result["pairs"].append(row)
        result["completed"] = True
        for arm, _, replay in arms:
            if replay:
                result[f"{arm}_identical"] = all(
                    r[arm]["unequal_logits"] == 0 and r[f"routes_{arm}_match_record"] for r in result["pairs"])
        result["gate_passed"] = all(result[f"{arm}_identical"] for arm, _, replay in arms if replay)
        print(json.dumps({k: v for k, v in result.items() if k.endswith("_identical") or k == "gate_passed"}), flush=True)
    finally:
        (root / "result.json").write_text(json.dumps(result, indent=2))
        print(f"saved {root / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
