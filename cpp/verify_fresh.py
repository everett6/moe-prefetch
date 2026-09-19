"""
The export check that matters: same inputs, same numbers, Python and C++.

A predictor file with a permuted block table, a transposed weight matrix, or a
bias read at the wrong offset loads cleanly and simply predicts worse. Nothing
raises. So the test is numerical, on real held-out traces rather than random
input -- random input would not reproduce the overlap between the four expert
sets that occurs in practice, which is exactly where an offset error shows up.
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from d2_export import read_binary, BLOCK_NAME  # noqa: E402
from train_deep import Index  # noqa: E402
import models_deep as M  # noqa: E402

MODEL = os.environ.get("MODEL", os.path.join(ROOT, "models", "predictor-fresh.bin"))
CKPT = os.environ.get("CKPT", os.path.join(ROOT, "artifacts", "ckpt",
                                           "FRESH_SUPERVISED-linearctx-lr1e2.pt"))
BIN = os.path.join(HERE, "build", "test-predictor")
INDEX = os.path.join(ROOT, "data", "index-v3.npz")
ORDER = ["prev", "cur", "below", "self_prev"]
TOL = 2e-4


def main():
    m = read_binary(MODEL)
    ix = Index(INDEX)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = M.build("linearctx", ix.n_layers, context=ck["cfg"].get("context", True))
    net.load_state_dict(ck["state"])
    net.eval()
    print(f"model: {os.path.basename(MODEL)}  blocks "
          f"{[BLOCK_NAME[b] for b in m['blocks']]}  dim {m['feat_dim']}")

    rows = ix.rows(2)[:3000]            # held-out registers, real co-occurrence
    d = ix.d
    cases = []
    for r in rows:
        L = int(d["layer"][r])
        if L not in m["W"]:
            continue
        cases.append((L, {k: [int(e) for e in d[k][r] if e >= 0] for k in ORDER}))
    cases = cases[:2000]
    print(f"{len(cases)} real cases from the locked test registers")

    with tempfile.TemporaryDirectory() as t:
        cp, sp = os.path.join(t, "c.bin"), os.path.join(t, "s.bin")
        with open(cp, "wb") as f:
            f.write(struct.pack("<i", len(cases)))
            for L, ids in cases:
                f.write(struct.pack("<i", L))
                for k in ORDER:
                    f.write(struct.pack("<i", len(ids[k])))
                    f.write(np.asarray(ids[k], dtype="<i4").tobytes())
        subprocess.run([BIN, MODEL, cp, sp], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raw = np.fromfile(sp, dtype="<f4").reshape(len(cases), m["n_expert"] + 8)

    worst, agree = 0.0, 0
    with torch.no_grad():
        for n, (L, ids) in enumerate(cases):
            # An empty list must be padded with -1, not 0. The model masks
            # negative ids; padding with 0 silently adds expert 0's weight row,
            # which is what made this check fail at 0.75 while the top-8 still
            # agreed 99.97% of the time -- a wrong test, not a wrong export.
            batch = {k: torch.tensor([ids[k] if ids[k] else [-1]]) for k in ORDER}
            batch["layer"] = torch.tensor([L])
            ref = net(batch)[0].numpy()[0]
            ref = (ref - ref.mean()) / (ref.std() + 1e-9)   # the C++ z-scores; monotonic
            got = raw[n, :m["n_expert"]]
            worst = max(worst, float(np.max(np.abs(ref - got))))
            py_top = set(np.argpartition(-ref, 8)[:8].tolist())
            agree += len(py_top & set(raw[n, m["n_expert"]:].view("<i4").tolist())) / 8

    print(f"max |C++ - Python| over {len(cases) * m['n_expert']:,} scores: {worst:.3e}")
    print(f"top-8 agreement: {100 * agree / len(cases):.2f}%")
    ok = worst <= TOL and agree / len(cases) >= 0.9999
    print("\nPASS: the C++ loader reproduces the trained model."
          if ok else f"\nFAIL (tolerance {TOL:.0e})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
