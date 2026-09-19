"""
D5: if the output is not byte-identical, is it still right?

The strict criterion fails, and it fails for the shipped cache exactly as much as
for anything this project changed: PR #27861 splits one `mul_mat_id` into a
device cache chain and a host chain and sums them, so floating-point addition
happens in a different order and greedy decoding can take a different token.

That makes byte-identity unreachable for this design, which is worth saying
plainly rather than dropping the criterion. But "not identical" and "not correct"
are different claims, so this grades answers that have a checkable answer --
arithmetic and closed-book facts -- under cache-off and under the final
configuration, and compares accuracy rather than bytes.

Small and deliberately verifiable: the point is not to reproduce a benchmark
suite, it is to establish that the divergence is numerical noise rather than
damage.
"""
import json
import os
import re
import subprocess
import sys
import time

import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8099
PROBLEMS = [
    ("What is 17 times 24? Answer with just the number.", "408"),
    ("What is 144 divided by 12? Answer with just the number.", "12"),
    ("What is 2 to the power of 10? Answer with just the number.", "1024"),
    ("What is 365 minus 189? Answer with just the number.", "176"),
    ("A shop sells pens for 3 dollars each. How much for 14 pens? Answer with just the number.", "42"),
    ("What is 15% of 200? Answer with just the number.", "30"),
    ("What is the sum of the first 10 positive integers? Answer with just the number.", "55"),
    ("If a train goes 60 km in 45 minutes, how many km in 1 hour? Answer with just the number.", "80"),
    ("What is 7 factorial? Answer with just the number.", "5040"),
    ("What is the square root of 169? Answer with just the number.", "13"),
    ("How many minutes are in 3 days? Answer with just the number.", "4320"),
    ("What is 1000 minus 37 times 4? Answer with just the number.", "852"),
    ("What is the capital of Japan? Answer with just the city name.", "tokyo"),
    ("What is the chemical symbol for gold? Answer with just the symbol.", "au"),
    ("Who wrote the play Hamlet? Answer with just the surname.", "shakespeare"),
    ("What is the largest planet in the solar system? Answer with just the name.", "jupiter"),
    ("In what year did the Second World War end? Answer with just the year.", "1945"),
    ("What is the chemical formula for water? Answer with just the formula.", "h2o"),
    ("How many sides does a hexagon have? Answer with just the number.", "6"),
    ("What is the freezing point of water in Celsius? Answer with just the number.", "0"),
    ("What is the longest river in Africa? Answer with just the name.", "nile"),
    ("How many continents are there? Answer with just the number.", "7"),
    ("What gas do plants absorb from the air? Answer with just the gas name.", "carbon dioxide"),
    ("What is the speed of light in metres per second, to three significant figures?", "3.00"),
]


def ask(prompt, n=48):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0,
                       "top_k": 1, "seed": 0, "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["content"]


def graded():
    ok, results = 0, []
    for q, want in PROBLEMS:
        a = ask(q)
        norm = re.sub(r"[^a-z0-9. ]", " ", a.lower())
        hit = want.lower() in norm or want.lower().replace(" ", "") in norm.replace(" ", "")
        ok += hit
        results.append({"q": q, "want": want, "got": a.strip()[:70], "correct": bool(hit)})
    return ok, results


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    ok, res = graded()
    print(f"{label}: {ok}/{len(PROBLEMS)} correct")
    out = os.path.join(ROOT, "artifacts", f"quality_{label}.json")
    json.dump({"label": label, "correct": ok, "total": len(PROBLEMS), "results": res},
              open(out, "w"), indent=1)
    for r in res:
        if not r["correct"]:
            print(f"  MISS want={r['want']!r} got={r['got']!r}")
