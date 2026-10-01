"""
Turn the scheduler profiler's cumulative blocks into a steady-state reading.

GGML_SCHED_PROFILE=<n> prints, every n graphs, averages over EVERYTHING since
the server started -- prefill and cache warm-up included. The difference of the
last two blocks is the last n graphs alone, which is what a token costs once
the cache is warm.

    python3 experiments/y1_profile_delta.py <server.log> [completion.json]
"""
import json
import re
import sys


def parse(path):
    blocks, cur = [], None
    for line in open(path, errors="replace"):
        m = re.search(r"sched-profile: (\d+) graphs, ([\d.]+) splits/graph", line)
        if m:
            cur = {"n": int(m.group(1)), "splits": float(m.group(2)), "b": {}}
            blocks.append(cur)
            continue
        m = re.search(r"sched-profile:\s+(\S+)\s+([\d.]+) splits/graph\s+inputs\s+([\d.]+) us/graph"
                      r"\s+compute\s+([\d.]+) us/graph", line)
        if m and cur is not None:
            cur["b"][m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return blocks


def show(tag, n, splits, b):
    tot = sum(v[1] + v[2] for v in b.values())
    print(f"{tag}: {n} graphs, {splits:.1f} splits/graph, {tot:.0f} us/graph inside the scheduler")
    for name, (sp, inp, comp) in b.items():
        per = (inp / sp, comp / sp) if sp else (0.0, 0.0)
        print(f"  {name:10s} {sp:6.1f} splits/graph   inputs {inp:8.1f} us/graph   compute {comp:8.1f} us/graph"
              f"   ({per[0]:.1f} + {per[1]:.1f} us/split)")
    return tot


def main():
    blocks = parse(sys.argv[1])
    if len(sys.argv) > 2:
        t = json.load(open(sys.argv[2])).get("timings", {})
        print(f"# decode: {t.get('predicted_n')} tokens, {t.get('predicted_per_token_ms', 0):.3f} ms/token, "
              f"{t.get('predicted_per_second', 0):.2f} tok/s (server's own timing, whole request)")
    print(f"# profile blocks printed: {len(blocks)}")
    if not blocks:
        sys.exit("NO PROFILE BLOCK was printed -- nothing was measured")
    last = blocks[-1]
    show("CUMULATIVE", last["n"], last["splits"], last["b"])
    if len(blocks) >= 2:
        a, z = blocks[-2], blocks[-1]
        dn = z["n"] - a["n"]
        d = {}
        for name in z["b"]:
            if name in a["b"]:
                d[name] = tuple((z["b"][name][i] * z["n"] - a["b"][name][i] * a["n"]) / dn for i in range(3))
        show("LAST INTERVAL", dn, (z["splits"] * z["n"] - a["splits"] * a["n"]) / dn, d)


if __name__ == "__main__":
    main()
