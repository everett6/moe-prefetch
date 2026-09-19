"""
Regression tests for the real-prompt pipeline.

Each of these guards a property whose failure would be silent. The old corpus
generator shipped a stride bug that produced 5 distinct prompts per register
instead of 40 -- nothing raised, the corpus was just quietly small. These are
the equivalent traps for the real pipeline: a subsample that drops half a layer
stack, a split that puts every decision group in train, a fetch that records a
prompt without its provenance.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import real_prompts as RP  # noqa: E402

PROMPTS = os.path.join(ROOT, "data", "prompts")
have_corpus = os.path.isdir(PROMPTS) and any(
    f.endswith(".jsonl") for f in os.listdir(PROMPTS)) if os.path.isdir(PROMPTS) else False
needs_corpus = pytest.mark.skipif(not have_corpus, reason="fetch data/prompts first")


def test_record_carries_provenance():
    r = RP.record("src", "grp", "code", "  hello  ", "http://x", "MIT")
    assert r["source"] == "src" and r["group"] == "grp" and r["domain"] == "code"
    assert r["text"] == "hello" and r["url"] == "http://x" and r["license"] == "MIT"
    assert len(r["sha256"]) == 64
    assert r["truncated"] is False


def test_record_truncates_and_says_so():
    r = RP.record("s", "g", "code", "x" * (RP.MAX_CHARS + 500), "u", "l")
    assert r["chars"] == RP.MAX_CHARS
    assert r["truncated"] is True


def test_record_drops_empty():
    assert RP.record("s", "g", "code", "   ", "u", "l") is None
    assert RP.record("s", "g", "code", None, "u", "l") is None


def test_strip_html_keeps_code_as_a_block():
    """A flattened code block is a different prompt from a fenced one, and the
    difference is exactly the thing this corpus exists to capture."""
    out = RP.strip_html("<p>Try this:</p><pre><code>def f(x):\n    return x\n</code></pre>")
    assert "```" in out
    assert "def f(x):" in out
    assert "    return x" in out
    assert "<pre>" not in out and "<code>" not in out


def test_strip_html_unescapes_entities():
    assert RP.strip_html("<p>a &lt; b &amp;&amp; c &gt; d</p>") == "a < b && c > d"


@needs_corpus
def test_build_real_corpus_is_deterministic():
    a = RP.build_real_corpus(200)
    b = RP.build_real_corpus(200)
    assert a == b


@needs_corpus
def test_build_real_corpus_prefixes_are_balanced():
    """Capture is capped by a byte budget and may stop early, so any PREFIX of
    the corpus has to cover every group -- not just the whole thing."""
    c = RP.build_real_corpus(1400)
    groups = {g for g, _, _ in c}
    assert len(groups) >= 20
    prefix = {g for g, _, _ in c[:len(groups)]}
    assert prefix == groups, "a prefix of one round is missing groups"
    assert {g.split("/")[0] for g, _, _ in c} == {"code", "decision"}


@needs_corpus
def test_every_prompt_has_a_group_and_domain():
    for fn in os.listdir(PROMPTS):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(PROMPTS, fn)) as f:
            for line in f:
                r = json.loads(line)
                assert r["domain"] in ("code", "decision"), r["id"]
                assert r["group"] and r["source"] and r["text"].strip()
                assert r["chars"] <= RP.MAX_CHARS


def _capture():
    os.environ.setdefault("PROMPT_SET", "real")
    import capture_corpus
    return capture_corpus


def test_subsample_keeps_whole_layer_stacks():
    """A position is kept for all layers or none. A partial stack trains nothing
    and would skew the per-layer row counts without any error."""
    cc = _capture()
    K = np.array([(0, p, l) for p in range(900) for l in range(48)], dtype=np.int32)
    m = cc.subsample_positions(K, 128)
    kept = K[m]
    pos, counts = np.unique(kept[:, 1], return_counts=True)
    assert set(counts.tolist()) == {48}
    assert len(pos) <= 128


def test_subsample_keeps_short_prompts_whole():
    cc = _capture()
    K = np.array([(0, p, l) for p in range(40) for l in range(48)], dtype=np.int32)
    assert cc.subsample_positions(K, 128).all()


def test_subsample_keeps_positions_adjacent():
    """`prev` -- the previous token's experts -- is the strongest feature the
    model has. An evenly spaced sample of single positions leaves t-1 absent for
    almost every t, and prev then silently becomes the -1 fill on nearly every
    row. Blocks keep it available."""
    cc = _capture()
    K = np.array([(0, p, l) for p in range(900) for l in range(48)], dtype=np.int32)
    pos = np.unique(K[cc.subsample_positions(K, 128)][:, 1])
    have_prev = sum(1 for p in pos if p - 1 in set(pos.tolist()))
    assert have_prev / len(pos) > 0.8, f"only {have_prev}/{len(pos)} rows have t-1"


def test_subsample_keeps_the_cold_start():
    """Cold-start rows behave differently from warm ones; pooling them has
    already produced one wrong precision number in this project."""
    cc = _capture()
    K = np.array([(0, p, l) for p in range(900) for l in range(48)], dtype=np.int32)
    pos = np.unique(K[cc.subsample_positions(K, 128)][:, 1])
    assert list(pos[:16]) == list(range(16))
    assert pos[-1] == 899


def test_split_is_stratified_by_domain():
    """An unstratified draw can put every decision group in train, and the test
    number then says nothing about half the workload."""
    cc = _capture()
    groups = {f"code/repo{i}" for i in range(20)} | {f"decision/site{i}" for i in range(8)}
    sp = cc.assign_splits(groups, 20260918)
    for split in ("train", "val", "test"):
        for domain in ("code", "decision"):
            assert any(sp[g] == split and g.startswith(domain) for g in groups), \
                f"{domain} missing from {split}"


def test_split_is_deterministic_and_total():
    cc = _capture()
    groups = {f"code/r{i}" for i in range(20)} | {f"decision/s{i}" for i in range(8)}
    a, b = cc.assign_splits(groups, 7), cc.assign_splits(groups, 7)
    assert a == b
    assert set(a) == groups
    assert cc.assign_splits(groups, 7) != cc.assign_splits(groups, 8)
