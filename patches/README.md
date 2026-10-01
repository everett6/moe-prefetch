# llama.cpp patches

Against PR #27861 at `bccbacd`.

**To reproduce the engine**, from a clean checkout of `bccbacd`:

```bash
cp <this repo>/cpp/moe-predictor.{h,cpp} src/
git apply <this repo>/patches/0006-*.patch
git apply <this repo>/patches/0007-*.patch
```

That sequence is verified to apply cleanly and to produce a tree byte-identical
to the one the experiments ran on (0006) plus the 2026-09-30 work (0007).

| patch | what | status |
|---|---|---|
| `0006` | early issue, predictor and evictor hooks, online learning, batched uploads, per-layer slot counts, and the upstream fixes of 0001/0003/0004 | measured; see docs/ |
| `0007` | admission gate, batched table writes, heat-file warm start, cache-prior routing, probation for predicted uploads, scheduler split profiler | **compiles; never run** |

`0001` – `0005` are the same changes as they were found, one bug or feature
each. They are kept because `docs/UPSTREAM-BUGS.md`, `docs/D1-RESULT.md` and
`docs/EVICTION.md` cite them by number. They do **not** apply in sequence any
more (0002 fails on a clean checkout) and are superseded by 0006 for building.
