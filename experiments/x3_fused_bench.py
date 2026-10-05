"""Paired fused CPU benchmark; save output hashes and NVIDIA telemetry.

Run with GPU access: python3 experiments/x3_fused_bench.py
BENCH_BIN, MODEL, ROUNDS, N_BENCH, N_PREDICT, SLOTS and PORT are configurable.
The fused CPU switch is the only difference between arms. Other MoE switches
are cleared so an inherited cache-prior setting cannot change the target.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import threading
import time

import requests
import r3_real_bench as bench

ROOT = Path(__file__).resolve().parents[1]
bench.BIN = os.environ.get("BENCH_BIN", str(ROOT / "engine/build/bin/llama-server"))
bench.MAX_FOREIGN_MIB = int(os.environ.get("MAX_FOREIGN_MIB", "1800"))
for name in list(bench.ENV):
    if name.startswith("LLAMA_MOE_"):
        del bench.ENV[name]
DETERMINISTIC = os.environ.get("BENCH_DETERMINISTIC", "0") == "1"
if DETERMINISTIC:
    bench.ENV["LLAMA_MOE_DETERMINISTIC"] = "1"
bench.ENV.update(LLAMA_MOE_ADMIT="window", LLAMA_MOE_BATCH_TABLES="1", LLAMA_MOE_CACHE_PRIOR="0")
bench.ENV["LLAMA_MOE_SLOT_PROFILE"] = (
    "0:88,1:88,3:72,5:64,6:48,7:40,10:72,11:64,12:64,13:64,14:48,15:48,"
    "18:40,19:40,20:40,23:64,24:64,25:64,27:48,30:40,31:40,32:40,34:64,35:64,"
    "36:64,37:64,40:48,43:40,44:48,45:48,46:48")
# Profiles belong to the measured 30B model, never apply them to another model.
if "Qwen3-30B-A3B-Instruct-2507" not in bench.MODEL:
    bench.ENV.pop("LLAMA_MOE_SLOT_PROFILE")


# arm -> (extra environment, banner its server log must print). Arm 1 is the treatment.
ARMS = {0: ({"LLAMA_MOE_FUSED_CPU": "0"}, "fused CPU rows DISABLED"),
        1: ({"LLAMA_MOE_FUSED_CPU": "1"}, "fused CPU rows ENABLED")}
TAG = "x3_fused"
LABEL = "FUSED"


def completion(text, n_predict):
    start = time.monotonic()
    r = requests.post(f"http://127.0.0.1:{bench.PORT}/completion",
                      json={"prompt": text, "n_predict": n_predict, "temperature": 0,
                            "top_k": 1, "seed": 0, "cache_prompt": False,
                            "return_tokens": True}, timeout=600)
    r.raise_for_status()
    result = r.json()
    if "error" in result:
        raise RuntimeError(result["error"])
    timings = result["timings"]
    identity = result.get("tokens")
    if not identity:
        identity = result.get("content")
    if identity is None:
        raise RuntimeError("completion did not return output")
    return {"decode_tps": timings["predicted_per_second"],
            "tokens": result.get("tokens"), "n_pred": timings["predicted_n"], "wall_s": time.monotonic() - start,
            "output_sha256": hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()}


def telemetry(stop, samples):
    while not stop.is_set():
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        host = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemAvailable", "SwapFree", "SwapTotal"):
                host[key + "_kib"] = int(value.split()[0])
        for line in Path("/proc/vmstat").read_text().splitlines():
            key, value = line.split()
            if key in ("pswpin", "pswpout"):
                host[key + "_pages"] = int(value)
        samples.append({"time": time.time(), "gpu": result.stdout.strip(), "status": result.returncode, **host})
        stop.wait(1)


def main():
    if not Path(bench.BIN).is_file():
        raise RuntimeError(f"build llama-server first: {bench.BIN}")
    # Check the power cap too; this machine's established stable cap is 175 W.
    power = subprocess.run(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
                           check=True, capture_output=True, text=True)
    if any(float(line) > 175 for line in power.stdout.splitlines()):
        raise RuntimeError("GPU power cap exceeds the established 175 W test configuration")
    selected = bench.select(bench.load_prompts(), bench.N_BENCH)
    if not selected:
        raise RuntimeError("no benchmark prompts available")
    out = {"binary": bench.BIN, "model": bench.MODEL, "rounds": bench.ROUNDS,
           "n_predict": bench.N_PREDICT, "deterministic_publication": DETERMINISTIC, "measurements": [], "telemetry": [], "completed": False}
    path = ROOT / "artifacts" / f"{TAG}_{time.time_ns()}.json"
    stop = threading.Event()
    monitor = threading.Thread(target=telemetry, args=(stop, out["telemetry"]), daemon=True)
    monitor.start()
    fitted = None
    try:
        for rnd in range(bench.ROUNDS):
            # Alternate arm order to reduce drift; each arm gets a fresh server.
            for fused in ([0, 1] if rnd % 2 == 0 else [1, 0]):
                label = f"{LABEL}-{fused}"
                arm_env, banner = ARMS[fused]
                bench.EXPECT[label] = banner
                p, vram = bench.launch(label, bench.BASE56, dict(arm_env))
                try:
                    if fitted is None:
                        fitted = [r for r in selected if bench.tokenize(r["text"]) <= bench.CTX - bench.N_PREDICT - 64]
                        if not fitted:
                            raise RuntimeError("no prompts fit the configured context")
                    completion(bench.CONTROL, 32)
                    order = list(fitted)
                    random.Random(bench.SEED + rnd).shuffle(order)
                    for record in order:
                        metrics = completion(record["text"], bench.N_PREDICT)
                        out["measurements"].append({"round": rnd, "fused": fused,
                                                    "id": record["id"], "vram": vram, **metrics})
                    recent = [r["decode_tps"] for r in out["measurements"] if r["round"] == rnd and r["fused"] == fused]
                    print(f"round {rnd} fused={fused}: {statistics.fmean(recent):.2f} tok/s ({len(recent)} prompts)", flush=True)
                finally:
                    bench.stop(p)
        pairs = {}
        for row in out["measurements"]:
            pairs.setdefault((row["round"], row["id"]), {})[row["fused"]] = row
        deltas, mismatches = [], []
        for key, arms in pairs.items():
            if (arms[0]["output_sha256"] != arms[1]["output_sha256"] or
                    arms[0]["n_pred"] != arms[1]["n_pred"]):
                a, b = arms[0].get("tokens"), arms[1].get("tokens")
                first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))) if a is not None and b is not None else None
                mismatches.append({"round": key[0], "id": key[1], "first_difference_token": first})
            deltas.append(arms[1]["decode_tps"] - arms[0]["decode_tps"])
        out["summary"] = {"paired_n": len(deltas), "paired_gain_tps": statistics.fmean(deltas),
                          "paired_sem": statistics.stdev(deltas) / len(deltas) ** 0.5 if len(deltas) > 1 else None,
                          "output_mismatches": mismatches,
                          "gate_passed": not mismatches and statistics.fmean(deltas) >= 4}
        out["completed"] = True
        print(json.dumps(out["summary"], indent=2), flush=True)
    finally:
        stop.set()
        monitor.join(timeout=12)
        path.write_text(json.dumps(out, indent=2))
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
