"""
D2 verification: the C++ loader and the Python trainer must agree exactly.

This is the step the plan singled out as the one that will silently disagree.
A wrong block order, a transposed weight matrix, or the prior applied before the
z-score instead of after all load cleanly and all produce a predictor that works
-- just worse. None of them raises anything. So the test is numerical: same
(layer, prev, cur) in, same 128 floats out, on real held-out traces rather than
on random input, because random input would not catch an error that only shows
up when `prev` and `cur` overlap the way they do in practice.
"""
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from d2_export import read_binary, BLOCK_PREV, BLOCK_CUR, BLOCK_PCA  # noqa: E402

MODEL = os.environ.get("MODEL", os.path.join(ROOT, "models", "predictor-hostfree.bin"))
DATA = os.environ.get("DATA", os.path.join(ROOT, "experiments", "m2_hidden.npz"))
BIN = os.path.join(HERE, "build", "test-predictor")
N_CASES = int(os.environ.get("N_CASES", "2000"))


def py_score(m, il, prev, cur):
    x = np.zeros(m["feat_dim"], dtype=np.float32)
    for (b, off, size) in m["table"]:
        if b == BLOCK_PREV:
            x[off + np.asarray(prev, dtype=int)] = 1.0
        elif b == BLOCK_CUR:
            x[off + np.asarray(cur, dtype=int)] = 1.0
        elif b == BLOCK_PCA:
            raise SystemExit("this check covers the host-free model only")
    s = x @ m["W"][il]
    s = (s - s.mean()) / (s.std() + 1e-9)
    if m["prior"][il]:
        s[np.asarray(prev, dtype=int)] += m["prior"][il]
    return s


def main():
    m = read_binary(MODEL)
    d = np.load(DATA, allow_pickle=True)
    E = d["experts"].astype(np.int64)
    keys = d["keys"]
    idx = {(p, t, l): i for i, (p, t, l) in enumerate(zip(keys[:, 0], keys[:, 1], keys[:, 2]))}

    # real cases: (layer L, prev = experts of L+1 on the previous token,
    #              cur = experts of L on this token)
    cases = []
    for (p, t, L), i in idx.items():
        if len(cases) >= N_CASES:
            break
        if L not in m["W"]:
            continue
        j = idx.get((p, t - 1, L + 1))
        if j is None:
            continue
        cases.append((int(L), [int(e) for e in E[j]], [int(e) for e in E[i]]))
    print(f"{len(cases)} real (layer, prev, cur) cases from held-out traces")

    cpath = os.path.join(HERE, "build", "cases.bin")
    spath = os.path.join(HERE, "build", "scores.bin")
    with open(cpath, "wb") as f:
        f.write(struct.pack("<i", len(cases)))
        for (L, pv, cu) in cases:
            f.write(struct.pack("<ii", L, len(pv)))
            f.write(np.asarray(pv, dtype="<i4").tobytes())
            f.write(struct.pack("<i", len(cu)))
            f.write(np.asarray(cu, dtype="<i4").tobytes())

    subprocess.run([BIN, MODEL, cpath, spath], check=True)

    n_e = m["n_expert"]
    raw = np.fromfile(spath, dtype="<f4")
    stride = n_e + 8
    assert raw.size == len(cases) * stride, f"{raw.size} floats, expected {len(cases) * stride}"
    got = raw.reshape(len(cases), stride)
    cpp_scores = got[:, :n_e]
    cpp_top = got[:, n_e:].view("<i4")

    worst = 0.0
    agree = 0
    for n, (L, pv, cu) in enumerate(cases):
        ref = py_score(m, L, pv, cu)
        worst = max(worst, float(np.max(np.abs(ref - cpp_scores[n]))))
        py_top = set(np.argpartition(-ref, 8)[:8].tolist())
        agree += len(py_top & set(cpp_top[n].tolist())) / 8

    print(f"max |C++ - Python| over {len(cases) * n_e:,} scores: {worst:.3e}")
    print(f"top-8 agreement: {100 * agree / len(cases):.2f}%")
    tol = 2e-4
    if worst > tol or agree / len(cases) < 0.9999:
        print(f"\nFAIL: the loaders disagree (tolerance {tol:.0e})")
        return 1
    print("\nPASS: the C++ loader reproduces the trainer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
