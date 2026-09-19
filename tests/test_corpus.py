"""
Tests for the capture corpus and its split.

The failure this guards against is leakage, and leakage does not look like a
bug -- it looks like a good result. If a register appears in both training and
test, or a prompt is captured twice under different ids, the reported recall is
measuring memorisation and there is nothing in the output to say so.

So: splits are assigned by register from a fixed seed before capture, written to
the manifest, and checked here to be a partition. Nothing downstream is allowed
to reassign them.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from corpus_prompts import build_corpus, SEED_PROMPTS, GENERATORS  # noqa: E402
from capture_corpus import assign_splits  # noqa: E402

MANIFEST = os.path.join(ROOT, "data", "corpus", "manifest.json")


def test_corpus_prompts_are_unique():
    c = build_corpus()
    texts = [t for _, t, _ in c]
    assert len(texts) == len(set(texts)), "duplicate prompt text in the corpus"


def test_corpus_is_deterministic():
    assert build_corpus() == build_corpus()


def test_every_register_is_well_populated():
    """An earlier generator collided and produced 5 prompts for most registers
    instead of 40. It raised no error -- duplicates were simply dropped."""
    c = build_corpus(30)
    by = {}
    for r, _, _ in c:
        by[r] = by.get(r, 0) + 1
    assert len(by) == len(GENERATORS), f"expected {len(GENERATORS)} registers, got {len(by)}"
    for r, n in by.items():
        assert n >= 30, f"register {r} has only {n} prompts"


def test_seed_prompts_are_preserved_verbatim():
    """The original 40 must survive unchanged, or the earlier results stop being
    reproducible on this corpus."""
    c = {t for _, t, src in build_corpus() if src == "seed"}
    for _, t in SEED_PROMPTS:
        assert t in c


def test_split_is_a_partition_and_deterministic():
    regs = set(GENERATORS)
    a = assign_splits(regs, 20260918)
    b = assign_splits(regs, 20260918)
    assert a == b, "split assignment is not deterministic"
    assert set(a) == regs, "some register got no split"
    counts = {s: sum(1 for v in a.values() if v == s) for s in ("train", "val", "test")}
    assert all(counts[s] >= 2 for s in counts), f"degenerate split: {counts}"
    assert sum(counts.values()) == len(regs)


def test_split_seed_changes_the_split():
    regs = set(GENERATORS)
    assert assign_splits(regs, 1) != assign_splits(regs, 2)


@pytest.mark.skipif(not os.path.exists(MANIFEST), reason="no corpus captured yet")
def test_manifest_is_internally_consistent():
    m = json.load(open(MANIFEST))
    assert m["schema_version"] == 2
    ids = [p["id"] for p in m["prompts"]]
    assert ids == sorted(set(ids)), "prompt ids are not unique and ordered"
    for p in m["prompts"]:
        assert p["split"] == m["splits"][p["register"]], \
            f"prompt {p['id']} split disagrees with its register"
    # a prompt's rows must live in exactly one shard, so a split cannot straddle
    seen = {}
    for sh in m["shards"]:
        for pid in sh["prompt_ids"]:
            assert pid not in seen, f"prompt {pid} appears in two shards"
            seen[pid] = sh["file"]


def _no_shards():
    if not os.path.exists(MANIFEST):
        return True
    return not json.load(open(MANIFEST)).get("shards")


@pytest.mark.skipif(_no_shards(), reason="no shards captured yet")
def test_shards_match_their_checksums_and_shapes():
    m = json.load(open(MANIFEST))
    d = os.path.dirname(MANIFEST)
    for sh in m["shards"][:3] + m["shards"][-3:]:      # ends are where truncation shows
        path = os.path.join(d, sh["file"])
        assert os.path.exists(path), sh["file"]
        assert os.path.getsize(path) == sh["bytes"]
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        assert h.hexdigest() == sh["sha256"], f"{sh['file']} checksum mismatch"
        z = np.load(path)
        assert z["hidden"].dtype == np.float16
        assert z["experts"].dtype == np.int16
        assert z["hidden"].shape == (sh["rows"], m["d_model"])
        assert z["experts"].shape == (sh["rows"], m["features"]["top_k"])
        assert z["keys"].shape == (sh["rows"], 3)
        assert int(z["experts"].max()) < m["n_expert"] and int(z["experts"].min()) >= 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
