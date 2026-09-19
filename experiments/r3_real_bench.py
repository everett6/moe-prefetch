"""
R3: does the headline number survive real prompts?

Every throughput figure in this repo was measured on ONE prompt, which I wrote:

    "Write a detailed explanation of how a B-tree index works in a relational
     database, including insertion, node splitting, deletion and range scans."

That is 150 characters of fluent English prose asking for prose back. A real
coding request is a 1,600-character GitHub issue with a stack trace and a diff in
it; a real decision request is a Stack Exchange question with three code blocks
and a table. Those differ in prompt length, in the prefill/decode ratio, and --
the part that matters for an expert cache -- in how much of the hidden-state
trajectory is spent in the same region of routing space.

So: same binary, same flags, same guards, same statistics. One variable changed,
the prompt, from mine to a real person's.

Reported per config:
  decode tok/s    the historical headline, server-reported, directly comparable
  prefill tok/s   real prompts are long enough that this stops being free
  wall s/request  what a user experiences, which decode tok/s alone hides

The control prompt is run THREE times in every pass -- first, middle and last.
The expert cache is a server-lifetime LRU, so a prompt that runs after 12 others
meets a warmer cache than one that runs first. The first version of this script
ran the control only at the start, which confounds "real prompts are faster"
with "later prompts are faster"; the three positions separate the two, and if
control_first < control_last the difference is warming, not content.

Guards, all three of which exist because their absence produced a wrong number
earlier in this project: port free before launch, the PID we started is the PID
on the GPU, and the VRAM is actually allocated.
"""
import json
import os
import random
import statistics
import subprocess
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROMPTDIR = os.path.join(ROOT, "data", "prompts")
ART = os.path.join(ROOT, "artifacts")

BIN = "/home/everett/llama.cpp-build/build/bin/llama-server"
MODEL = ("/home/everett/.lmstudio/models/lmstudio-community/"
         "Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf")
PORT = int(os.environ.get("PORT", "8099"))
N_PREDICT = int(os.environ.get("N_PREDICT", "200"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))
N_BENCH = int(os.environ.get("N_BENCH", "24"))
CTX = int(os.environ.get("CTX", "4096"))
SEED = 20260919

CONTROL = ("Write a detailed explanation of how a B-tree index works in a "
           "relational database, including insertion, node splitting, deletion "
           "and range scans.")

# Same three configs the 118.2 number came from, so the comparison is prompt-only.
COMMON = ["-ngl", "999", "-c", str(CTX), "--port", str(PORT),
          "--host", "127.0.0.1", "--no-webui", "-fa", "on"]
CONFIGS = {
    "BASELINE-ncmoe22": ["-ncmoe", "22"],
    "PREV-BEST-mmap":   ["-ncmoe", "48", "--moe-expert-cache", "72"],
    "NEW-BEST-nommap":  ["-ncmoe", "48", "--moe-expert-cache", "72", "--no-mmap"],
}

ENV = dict(os.environ)
ENV["CUDA_HOME"] = os.path.expanduser("~/miniconda3/envs/cudabuild")
ENV["LD_LIBRARY_PATH"] = ENV["CUDA_HOME"] + "/lib:" + ENV.get("LD_LIBRARY_PATH", "")


# -- prompt selection -------------------------------------------------------
def load_prompts():
    recs = []
    for fn in sorted(os.listdir(PROMPTDIR)):
        if fn.endswith(".jsonl"):
            with open(os.path.join(PROMPTDIR, fn)) as f:
                recs += [json.loads(l) for l in f]
    return recs


def select(recs, n):
    """Stratified by domain and by length band.

    A uniform sample would be dominated by whichever source has the most rows,
    and length is the dimension most likely to move throughput, so both are
    fixed by construction rather than left to luck.
    """
    rng = random.Random(SEED)
    bands = [("short", 0, 600), ("medium", 600, 2500), ("long", 2500, 10**9)]
    out = []
    for domain in ("code", "decision"):
        for name, lo, hi in bands:
            pool = [r for r in recs if r["domain"] == domain and lo <= r["chars"] < hi]
            rng.shuffle(pool)
            # one per source inside a band before any source repeats
            by_src, order = {}, []
            for r in pool:
                by_src.setdefault(r["source"], []).append(r)
            while len(order) < len(pool):
                for s in sorted(by_src):
                    if by_src[s]:
                        order.append(by_src[s].pop())
            take = max(1, n // 6)
            for r in order[:take]:
                r = dict(r)
                r["band"] = name
                out.append(r)
    return out[:n]


# -- server plumbing --------------------------------------------------------
def port_free():
    try:
        requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
        return False
    except requests.RequestException:
        return True


def wait_free(timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if port_free():
            return True
        time.sleep(1)
    return False


def launch(label, args):
    if not wait_free():
        raise RuntimeError(f"{label}: port {PORT} still held -- refusing to measure "
                           "someone else's server")
    log = open(f"/tmp/r3-{label}.log", "w")
    p = subprocess.Popen([BIN, "-m", MODEL] + COMMON + args,
                         stdout=log, stderr=subprocess.STDOUT, env=ENV)
    t0 = time.time()
    while time.time() - t0 < 900:
        if p.poll() is not None:
            raise RuntimeError(f"{label}: server died, see /tmp/r3-{label}.log")
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if '"status":"ok"' in r.text:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    else:
        raise RuntimeError(f"{label}: never became ready")
    # the PID answering must be the PID we started, and it must hold VRAM
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    vram = None
    for line in out.strip().splitlines():
        pid, mem = [x.strip() for x in line.split(",")]
        if int(pid) == p.pid:
            vram = mem
    if vram is None:
        p.kill(); p.wait()
        raise RuntimeError(f"{label}: pid {p.pid} holds no VRAM -- not measuring")
    return p, vram


def stop(p):
    p.terminate()
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill(); p.wait()
    wait_free()


def complete(prompt, n_predict=N_PREDICT):
    t0 = time.time()
    r = requests.post(f"http://127.0.0.1:{PORT}/completion",
                      json={"prompt": prompt, "n_predict": n_predict,
                            "temperature": 0, "top_k": 1, "seed": 0,
                            "cache_prompt": False}, timeout=600)
    wall = time.time() - t0
    t = r.json().get("timings", {})
    return {"decode_tps": t.get("predicted_per_second"),
            "prefill_tps": t.get("prompt_per_second"),
            "n_prompt": t.get("prompt_n"), "n_pred": t.get("predicted_n"),
            "wall_s": wall}


def tokenize(prompt):
    r = requests.post(f"http://127.0.0.1:{PORT}/tokenize", json={"content": prompt},
                      timeout=120)
    return len(r.json().get("tokens", []))


def summarise(xs):
    xs = [x for x in xs if x]
    if not xs:
        return None
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 2),
            "sd": round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0,
            "min": round(min(xs), 2), "max": round(max(xs), 2)}


def main():
    recs = load_prompts()
    if not recs:
        sys.exit("no prompts: run real_prompts.py first")
    bench = select(recs, N_BENCH)
    print(f"corpus {len(recs)} prompts -> bench set {len(bench)}")

    results = {c: {"real_decode": [], "real_prefill": [], "real_wall": [],
                   "control_first": [], "control_mid": [], "control_last": [],
                   "per_prompt": []} for c in CONFIGS}
    tok_counts, fitted = {}, None

    for rnd in range(ROUNDS):
        for label, args in CONFIGS.items():
            print(f"\n[round {rnd}] {label}", flush=True)
            p, vram = launch(label, args)
            try:
                if fitted is None:
                    # exact token counts from the model's own tokenizer, once
                    for r in bench:
                        tok_counts[r["id"]] = tokenize(r["text"])
                    room = CTX - N_PREDICT - 64
                    fitted = [r for r in bench if tok_counts[r["id"]] <= room]
                    dropped = len(bench) - len(fitted)
                    print(f"  {len(fitted)} prompts fit in {CTX} "
                          f"({dropped} too long for this context)")
                complete(CONTROL, 32)                       # warm-up, discarded
                results[label]["control_first"].append(complete(CONTROL)["decode_tps"])
                # Shuffle per pass. The selection order is grouped by length
                # band, so unshuffled it would put every long prompt late in the
                # pass -- and "longer" would then be indistinguishable from
                # "later against a warmer cache". A different order each round
                # decorrelates the two.
                order = list(fitted)
                random.Random(SEED + rnd).shuffle(order)
                mid = len(order) // 2
                for n, r in enumerate(order):
                    if n == mid:
                        results[label]["control_mid"].append(
                            complete(CONTROL)["decode_tps"])
                    m = complete(r["text"])
                    results[label]["real_decode"].append(m["decode_tps"])
                    results[label]["real_prefill"].append(m["prefill_tps"])
                    results[label]["real_wall"].append(m["wall_s"])
                    results[label]["per_prompt"].append(
                        {"id": r["id"], "source": r["source"], "band": r["band"],
                         "domain": r["domain"], "n_prompt": m["n_prompt"],
                         "decode_tps": m["decode_tps"], "wall_s": round(m["wall_s"], 2)})
                results[label]["control_last"].append(complete(CONTROL)["decode_tps"])
                print(f"  vram {vram}  control "
                      f"{results[label]['control_first'][-1]:.1f}/"
                      f"{results[label]['control_mid'][-1]:.1f}/"
                      f"{results[label]['control_last'][-1]:.1f} (first/mid/last)  "
                      f"real {statistics.fmean(results[label]['real_decode'][-len(order):]):.1f} tok/s")
            finally:
                stop(p)

    out = {"prompt_set_version": "real-v1-20260919", "ctx": CTX,
           "n_predict": N_PREDICT, "rounds": ROUNDS,
           "bench_prompts": [{"id": r["id"], "source": r["source"], "group": r["group"],
                              "domain": r["domain"], "band": r["band"],
                              "chars": r["chars"], "tokens": tok_counts.get(r["id"])}
                             for r in (fitted or [])],
           "configs": {}}
    for label in CONFIGS:
        R = results[label]
        out["configs"][label] = {
            "real_decode_tps": summarise(R["real_decode"]),
            "real_prefill_tps": summarise(R["real_prefill"]),
            "real_wall_s": summarise(R["real_wall"]),
            "control_first_tps": summarise(R["control_first"]),
            "control_mid_tps": summarise(R["control_mid"]),
            "control_last_tps": summarise(R["control_last"]),
            "per_prompt": R["per_prompt"],
        }
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "r3_real_throughput.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("\n=== decode tok/s ===")
    print(f"{'config':22}{'real prompts':>22}{'ctrl first':>12}{'ctrl mid':>10}"
          f"{'ctrl last':>11}")
    for label in CONFIGS:
        c = out["configs"][label]
        r = c["real_decode_tps"]
        print(f"{label:22} {r['mean']:8.1f} +/- {r['sd']:5.2f} (n={r['n']:3d})"
              f"{c['control_first_tps']['mean']:>12.1f}"
              f"{c['control_mid_tps']['mean']:>10.1f}"
              f"{c['control_last_tps']['mean']:>11.1f}")
    print("\nIf ctrl last >> ctrl first the gap is cache warming, not the prompts.")


if __name__ == "__main__":
    main()
