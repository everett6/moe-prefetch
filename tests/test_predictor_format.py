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
                       BLOCK_PCA, BLOCK_PREV, BLOCK_CUR, BLOCK_BELOW, BLOCK_SELF_PREV)

FRESH = os.path.join(ROOT, "models", "predictor-fresh.bin")
DRAFT = os.path.join(ROOT, "artifacts", "draft_old", "predictor-hostfree.bin")
TEST_BIN = os.path.join(ROOT, "cpp", "build", "test-predictor")
VERIFY = os.path.join(ROOT, "cpp", "verify_fresh.py")
N_EXPERT = 128
BLOCKS4 = (BLOCK_PREV, BLOCK_CUR, BLOCK_BELOW, BLOCK_SELF_PREV)


def _toy(blocks=BLOCKS4, n_layers=4, seed=0):
    rng = np.random.RandomState(seed)
    dim = sum(N_EXPERT for b in blocks if b != BLOCK_PCA)
    f = {"blocks": list(blocks), "dim": dim,
         "W": {L: rng.randn(dim, N_EXPERT).astype(np.float32) for L in range(n_layers - 1)},
         "bias": {L: rng.randn(N_EXPERT).astype(np.float32) for L in range(n_layers - 1)},
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
            assert np.array_equal(got["bias"][L], f["bias"][L]), f"layer {L} bias changed"


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


@pytest.mark.skipif(not os.path.exists(FRESH), reason="run export_fresh.py first")
def test_fresh_model_has_expected_shape():
    m = read_binary(FRESH)
    assert m["feat_dim"] == 512
    assert m["blocks"] == list(BLOCKS4), m["blocks"]
    assert m["n_expert"] == N_EXPERT and m["k"] == 8
    assert len(m["W"]) == 47, "one probe per layer that has a successor"
    for L, W in m["W"].items():
        assert W.shape == (512, N_EXPERT)
        assert np.isfinite(W).all(), f"layer {L} has non-finite weights"
        assert np.isfinite(m["bias"][L]).all(), f"layer {L} has non-finite bias"


@pytest.mark.skipif(not os.path.exists(DRAFT), reason="draft archive missing")
def test_draft_archive_is_a_v2_file_and_is_not_in_models():
    """The draft is kept for the comparison arm only. If it ever reappears in
    models/ something has started loading it as the production predictor."""
    assert not os.path.exists(os.path.join(ROOT, "models", "predictor-hostfree.bin"))
    with open(DRAFT, "rb") as fh:
        assert fh.read(4) == MAGIC
        ver = struct.unpack("<I", fh.read(4))[0]
    assert ver == 2, f"draft archive should stay at v{2}, found v{ver}"


@pytest.mark.skipif(not (os.path.exists(TEST_BIN) and os.path.exists(FRESH)),
                    reason="build cpp/ and run export_fresh.py first")
def test_cpp_matches_python():
    """Python -> file -> C++ -> the same scores, on real held-out traces.

    Delegated to cpp/verify_fresh.py so the check the release actually runs and
    the check CI runs are the same code rather than two implementations that can
    drift apart. Tolerance 2e-4 on z-scored logits: the two sum 32 weight rows of
    128 floats in different orders (numpy's pairwise summation against a plain C
    loop), so ~1e-6 is what float32 allows and 1e-3 would mean a real difference.
    """
    r = subprocess.run([sys.executable, VERIFY], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
