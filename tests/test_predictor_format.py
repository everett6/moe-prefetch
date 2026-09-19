"""
Tests for the MOEP predictor file: the things that fail silently if unguarded.

A predictor that loads with the wrong feature order, a transposed weight matrix
or a stale schema does not crash. It runs, and it predicts slightly worse, and
the only symptom is a benchmark number that disappoints for no visible reason.
That failure mode has already cost this project twice, so the format carries a
block table and the loader is required to reject anything that disagrees with it.

Run: python3 -m pytest tests/ -q     (or: python3 tests/test_predictor_format.py)
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from d2_export import (read_binary, write_binary, MAGIC, VERSION,  # noqa: E402
                       BLOCK_PCA, BLOCK_PREV, BLOCK_CUR)

HOSTFREE = os.path.join(ROOT, "models", "predictor-hostfree.bin")
FULL = os.path.join(ROOT, "models", "predictor-full.bin")
TEST_BIN = os.path.join(ROOT, "cpp", "build", "test-predictor")
N_EXPERT = 128


def _toy(blocks=(BLOCK_PREV, BLOCK_CUR), n_layers=4, seed=0):
    rng = np.random.RandomState(seed)
    dim = sum(N_EXPERT for b in blocks if b != BLOCK_PCA)
    f = {"blocks": list(blocks), "dim": dim,
         "W": {L: rng.randn(dim, N_EXPERT).astype(np.float32) for L in range(n_layers - 1)},
         "prior": {L: float(L % 3) * 0.5 for L in range(n_layers - 1)}}
    return f, n_layers


def test_round_trip_is_exact():
    f, n_layers = _toy()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.bin")
        write_binary(p, f, None, None, None, n_layers, 0)
        got = read_binary(p)
        assert got["feat_dim"] == f["dim"]
        assert got["blocks"] == f["blocks"]
        assert got["prior"] == {int(k): v for k, v in f["prior"].items()}
        for L, W in f["W"].items():
            assert np.array_equal(got["W"][L], W), f"layer {L} weights changed"


def test_rejects_bad_magic():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.bin")
        with open(p, "wb") as fh:
            fh.write(b"NOPE" + b"\0" * 64)
        with pytest.raises(AssertionError):
            read_binary(p)


def test_rejects_truncated_file():
    f, n_layers = _toy()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.bin")
        write_binary(p, f, None, None, None, n_layers, 0)
        full = open(p, "rb").read()
        open(p, "wb").write(full[: len(full) // 2])
        with pytest.raises(Exception):
            read_binary(p)


@pytest.mark.skipif(not os.path.exists(HOSTFREE), reason="run d2_export.py first")
def test_shipped_models_have_expected_shape():
    m = read_binary(HOSTFREE)
    assert m["feat_dim"] == 256 and m["blocks"] == [BLOCK_PREV, BLOCK_CUR]
    assert m["n_expert"] == N_EXPERT and m["k"] == 8
    assert len(m["W"]) == 47, "one probe per layer that has a successor"
    for L, W in m["W"].items():
        assert W.shape == (256, N_EXPERT)
        assert np.isfinite(W).all(), f"layer {L} has non-finite weights"
    if os.path.exists(FULL):
        mf = read_binary(FULL)
        assert mf["feat_dim"] == 512
        assert mf["blocks"] == [BLOCK_PCA, BLOCK_PREV, BLOCK_CUR]
        assert mf["comp"].shape == (2048, 256)


@pytest.mark.skipif(not (os.path.exists(TEST_BIN) and os.path.exists(HOSTFREE)),
                    reason="build cpp/ and run d2_export.py first")
def test_cpp_matches_python():
    """The round-trip that matters: Python -> file -> C++ -> same scores.

    Tolerance is 2e-4 absolute on z-scored logits. That is not arbitrary: the
    two implementations sum 16 weight rows of 128 floats in a different order
    (numpy's pairwise summation against a plain C loop), so agreement to ~1e-6
    is what float32 allows and anything near 1e-3 would mean a real difference
    in what is being computed. Top-8 agreement is required to be exact, since
    that is what the cache actually consumes.
    """
    m = read_binary(HOSTFREE)
    rng = np.random.RandomState(7)
    layers = sorted(m["W"])
    cases = []
    for _ in range(300):
        L = int(rng.choice(layers))
        prev = sorted(rng.choice(N_EXPERT, 8, replace=False).tolist())
        cur = sorted(rng.choice(N_EXPERT, 8, replace=False).tolist())
        cases.append((L, prev, cur))

    with tempfile.TemporaryDirectory() as d:
        cp, sp = os.path.join(d, "c.bin"), os.path.join(d, "s.bin")
        with open(cp, "wb") as fh:
            fh.write(struct.pack("<i", len(cases)))
            for (L, pv, cu) in cases:
                fh.write(struct.pack("<ii", L, len(pv)))
                fh.write(np.asarray(pv, dtype="<i4").tobytes())
                fh.write(struct.pack("<i", len(cu)))
                fh.write(np.asarray(cu, dtype="<i4").tobytes())
        subprocess.run([TEST_BIN, HOSTFREE, cp, sp], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raw = np.fromfile(sp, dtype="<f4").reshape(len(cases), N_EXPERT + 8)

    worst = 0.0
    for n, (L, pv, cu) in enumerate(cases):
        x = np.zeros(m["feat_dim"], dtype=np.float32)
        for (b, off, size) in m["table"]:
            if b == BLOCK_PREV:
                x[off + np.asarray(pv)] = 1.0
            elif b == BLOCK_CUR:
                x[off + np.asarray(cu)] = 1.0
        s = x @ m["W"][L]
        s = (s - s.mean()) / (s.std() + 1e-9)
        if m["prior"][L]:
            s[np.asarray(pv)] += m["prior"][L]
        worst = max(worst, float(np.max(np.abs(s - raw[n, :N_EXPERT]))))
        py_top = set(np.argpartition(-s, 8)[:8].tolist())
        cpp_top = set(raw[n, N_EXPERT:].view("<i4").tolist())
        assert py_top == cpp_top, f"case {n}: top-8 differs {py_top ^ cpp_top}"
    assert worst < 2e-4, f"max |C++ - Python| = {worst:.2e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
