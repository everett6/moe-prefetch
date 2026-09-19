"""
Sharded, resumable capture of (hidden state, routed experts) for the predictor.

m2_capture.py accumulated everything in Python lists and wrote one .npz at the
end. That was fine for 667 MB and cannot work at 15 GB: it needs the whole
corpus resident, it loses everything if the run dies, and it records nothing
about how the data was made.

What this adds, and why each matters here:

  sharding      flush at a prompt boundary once the buffer is large enough, so
                peak RAM is one shard rather than the corpus, and no sequence is
                ever split across two shards
  resumability  the manifest is rewritten atomically after every shard, listing
                the prompt ids already captured; re-running skips them
  validation    every shard carries its own row count, dtypes, shapes and a
                sha256, and `--validate` re-checks all of them. A truncated
                shard from a killed run is then a loud failure rather than a
                quietly smaller training set
  provenance    model identity, quantisation, git commit, feature schema
                version, capture config and timestamps go in the manifest, so a
                number measured on this corpus can be traced to what produced it
  split safety  train/val/test are assigned BY REGISTER before capture starts,
                from a fixed seed, and recorded. Nothing downstream may reassign
                them, which is what stops a hyperparameter search from quietly
                seeing the test set

Capture is CPU-only on purpose. The eval callback reads tensor data directly, so
the tensors have to be in host memory; with layers on the GPU `t.data` is a
device pointer and reading it is a segfault, not a wrong number.

**Schema v3** adds the labels the next model actually needs. Recall@8 against the
true routing tells you how often a predictor is right; it does not tell you which
rows were *hard*, and a prefetcher is only paid for the hard ones. So every row
also carries, computed exactly in a post-pass over the shard where rows can be
ordered by (prompt, position, layer):

  lru_miss_66   how many of this row's 8 experts would MISS a per-layer LRU cache
                of 66 slots at this point in the sequence -- the operating point
                the engine actually runs. 0 means the cache already had them and
                no prediction could have helped
  lru_miss_32   the same at 32 slots: the same sequence under real cache pressure
  repeat_hit    how many of the 8 match the previous token's experts at this layer.
                This is the naive-repeat baseline per row, so `repeat_hit < 8` is
                precisely the set of rows where a predictor has to do work

Capture also varies sequence length, context size and batch size across sessions,
because the draft corpus held all three fixed at 150 tokens, n_ctx 512, n_batch
512 -- and a predictor fitted only to short greedy continuations has never seen
the long-context routing it will meet in use.
"""
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
# Which corpus. "real" is the default from v4 on: the synthetic corpus is 616
# prompts of a median 14 tokens containing no code at all (artifacts/
# r2_characterisation.json), which is not what this model is asked to route.
PROMPT_SET = os.environ.get("PROMPT_SET", "real")
if PROMPT_SET == "real":
    from real_prompts import build_real_corpus as build_corpus  # noqa: E402
else:
    from corpus_prompts import build_corpus  # noqa: E402

MODEL = os.environ.get(
    "MODEL",
    "/home/everett/.lmstudio/models/lmstudio-community/"
    "Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
)
OUTDIR = os.environ.get("OUTDIR", os.path.join(ROOT, "data", "corpus-v4"))
MANIFEST = os.path.join(OUTDIR, "manifest.json")
N_PREDICT = int(os.environ.get("N_PREDICT", "0"))   # 0 = use the session schedule
FLUSH_ROWS = int(os.environ.get("FLUSH_ROWS", "120000"))
LIMIT = int(os.environ.get("LIMIT", "0"))          # 0 = whole corpus
PER_REGISTER = int(os.environ.get("PER_REGISTER", "1400"))
# Real prompts are ~30x longer than the synthetic ones, so a fixed byte budget
# buys ~30x fewer of them. Positions within a prompt are highly correlated and
# prompts are not, so the budget is spent on prompt diversity: at most this many
# positions are STORED per prompt, chosen after labelling.
KEEP_POSITIONS = int(os.environ.get("KEEP_POSITIONS", "128"))
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", "20260918"))
SCHEMA_VERSION = 3
DATASET_VERSION = os.environ.get("DATASET_VERSION", "v4-20260919-real")
TARGET_BYTES = float(os.environ.get("TARGET_GB", "0")) * 1e9   # 0 = no cap

GGML_MAX_DIMS, GGML_MAX_SRC, GGML_MAX_OP_PARAMS_I32, GGML_MAX_NAME = 4, 10, 16, 64


class GgmlTensor(ctypes.Structure):
    pass


GgmlTensor._fields_ = [
    ("type", ctypes.c_int), ("buffer", ctypes.c_void_p),
    ("ne", ctypes.c_int64 * GGML_MAX_DIMS), ("nb", ctypes.c_size_t * GGML_MAX_DIMS),
    ("op", ctypes.c_int), ("op_params", ctypes.c_int32 * GGML_MAX_OP_PARAMS_I32),
    ("flags", ctypes.c_int32), ("src", ctypes.POINTER(GgmlTensor) * GGML_MAX_SRC),
    ("view_src", ctypes.POINTER(GgmlTensor)), ("view_offs", ctypes.c_size_t),
    ("data", ctypes.c_void_p), ("name", ctypes.c_char * GGML_MAX_NAME),
    ("extra", ctypes.c_void_p), ("padding", ctypes.c_char * 8),
]


# (n_predict, n_ctx, n_batch). Cycled deterministically over the corpus so every
# register sees every regime -- otherwise "long context" would be confounded with
# whatever subject matter happened to land in the long bucket.
SESSION_PLANS = [
    {"n_predict": 64,  "n_ctx": 512,  "n_batch": 512, "label": "short"},
    {"n_predict": 150, "n_ctx": 1024, "n_batch": 256, "label": "medium"},
    {"n_predict": 320, "n_ctx": 2048, "n_batch": 512, "label": "long"},
    {"n_predict": 150, "n_ctx": 1024, "n_batch": 64,  "label": "medium-smallbatch"},
    {"n_predict": 640, "n_ctx": 4096, "n_batch": 512, "label": "verylong"},
    {"n_predict": 320, "n_ctx": 2048, "n_batch": 128, "label": "long-smallbatch"},
]


# Real prompts span 14 to 15,133 tokens, so n_ctx cannot be a free variable the
# way it was for a corpus whose longest prompt was 24 tokens -- it has to be big
# enough to hold the prompt. n_predict and n_batch stay free and cycle on the
# prompt id, which keeps batch size and continuation length orthogonal to length
# rather than confounded with it.
CTX_LADDER = [1024, 2048, 4096, 8192]
PREDICT_CYCLE = [64, 128, 192]
BATCH_CYCLE = [512, 256, 128, 64]
CHARS_PER_TOKEN = 3.0          # conservative: code tokenises denser than prose


def plan_for(pid, text=None):
    """Session parameters for one prompt.

    With no text this is the v3 behaviour, so the old corpus stays reproducible.
    With text, n_ctx is the smallest rung that holds the prompt plus its
    continuation plus a margin; a prompt too long for the top rung is truncated
    and the truncation is recorded on the prompt.
    """
    if text is None:
        return SESSION_PLANS[pid % len(SESSION_PLANS)]
    n_predict = PREDICT_CYCLE[pid % len(PREDICT_CYCLE)]
    n_batch = BATCH_CYCLE[(pid // len(PREDICT_CYCLE)) % len(BATCH_CYCLE)]
    est = len(text) / CHARS_PER_TOKEN
    need = est + n_predict + 64
    n_ctx = next((c for c in CTX_LADDER if c >= need), CTX_LADDER[-1])
    room = int((n_ctx - n_predict - 64) * CHARS_PER_TOKEN)
    return {"n_predict": n_predict, "n_ctx": n_ctx, "n_batch": n_batch,
            "label": f"ctx{n_ctx}-p{n_predict}-b{n_batch}",
            "max_chars": room, "truncated": len(text) > room}


def label_shard(H, E, Kk, n_expert, caps=(66, 32)):
    """Exact per-row cache and difficulty labels.

    Done here rather than in the ggml callback because it needs rows in
    (prompt, position, layer) order, and the callback sees a whole prefill batch
    for one layer at a time. The LRU is reset per prompt: each prompt is an
    independent session, and carrying a warm cache across them would label the
    first tokens of every prompt as unrealistically easy.
    """
    order = np.lexsort((Kk[:, 2], Kk[:, 1], Kk[:, 0]))
    out = {c: np.zeros(len(Kk), dtype=np.uint8) for c in caps}
    rep = np.zeros(len(Kk), dtype=np.uint8)
    caches = {c: {} for c in caps}          # (layer) -> dict expert -> tick, per capacity
    ticks = {c: 0 for c in caps}
    prev_by_layer = {}
    cur_prompt = None
    for i in order:
        p, t, L = int(Kk[i, 0]), int(Kk[i, 1]), int(Kk[i, 2])
        if p != cur_prompt:
            cur_prompt = p
            caches = {c: {} for c in caps}
            ticks = {c: 0 for c in caps}
            prev_by_layer = {}
        ids = [int(x) for x in E[i]]
        for c in caps:
            lay = caches[c].setdefault(L, {})
            miss = 0
            for e in ids:
                if e in lay:
                    ticks[c] += 1
                    lay[e] = ticks[c]
                else:
                    miss += 1
            for e in ids:                    # a miss is computed then admitted
                if e not in lay and len(lay) >= c:
                    lay.pop(min(lay, key=lay.get))
                ticks[c] += 1
                lay[e] = ticks[c]
            out[c][i] = miss
        pv = prev_by_layer.get(L)
        rep[i] = len(set(ids) & set(pv)) if pv else 0
        prev_by_layer[L] = ids
    return out, rep


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        read = 0
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            read += len(b)
            if limit and read >= limit:
                break
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def assign_splits(registers, seed):
    """By GROUP, so a held-out group is a whole repo, site or category.

    Real group keys are "<domain>/<group>" and are stratified within domain, so
    LOCKED TEST always contains both coding and decision-making groups -- an
    unstratified draw can put all the decision groups in train and then the test
    number says nothing about half the workload. Synthetic register names have no
    "/" and fall through to the original single-pool draw, so the v3 split stays
    reproducible.

    Deterministic from the seed and written to the manifest, which is what stops
    a later hyperparameter search from quietly reassigning it.
    """
    regs = sorted(registers)
    strata = {}
    for r in regs:
        strata.setdefault(r.split("/", 1)[0] if "/" in r else "_", []).append(r)
    out = {}
    for si, (name, pool) in enumerate(sorted(strata.items())):
        rng = np.random.RandomState(seed + si)
        order = list(rng.permutation(len(pool)))
        n_test = max(1 if "/" in pool[0] else 3, len(pool) // 5)
        n_val = max(1 if "/" in pool[0] else 2, len(pool) // 6)
        for rank, i in enumerate(order):
            if rank < n_test:
                out[pool[i]] = "test"
            elif rank < n_test + n_val:
                out[pool[i]] = "val"
            else:
                out[pool[i]] = "train"
    return out


def subsample_positions(K, keep, block=8):
    """Keep at most `keep` positions per prompt, as CONTIGUOUS BLOCKS.

    Three properties this has to have, and the reason it runs where it does:

      labels first   lru_miss_66 and repeat_hit are computed on the FULL
                     trajectory before anything is dropped, so a miss count
                     still describes the real sequence rather than the sample.
                     Subsampling first would make every row look like a cache
                     miss, which is the exact failure the labels exist to catch.
      whole stacks   a position is kept or dropped for all 48 layers at once. A
                     partial layer stack cannot train a per-layer model and would
                     silently skew the per-layer row counts.
      adjacency      the strongest feature the model has is `prev`, the previous
                     TOKEN's experts at the target layer. An evenly spaced sample
                     of single positions has no t-1 for almost every t, so prev
                     would be the missing-value fill on nearly every row and the
                     repeat signal would vanish -- quietly, since -1 is a legal
                     value. Blocks of `block` consecutive positions keep prev
                     available for all but the first row of each block.

    The head is kept in full because cold-start rows behave differently and
    pooling them with warm rows has already produced one wrong number here.
    """
    mask = np.zeros(len(K), dtype=bool)
    for pid in np.unique(K[:, 0]):
        sel = K[:, 0] == pid
        pos = np.unique(K[sel, 1])
        if len(pos) <= keep:
            chosen = pos
        else:
            head, tail, mid = pos[:16], pos[-16:], pos[16:-16]
            n_blocks = max(1, (keep - 32) // block)
            if len(mid) <= block:
                picked = [mid]
            else:
                starts = np.unique(np.linspace(0, len(mid) - block,
                                               n_blocks).astype(int))
                picked = [mid[s:s + block] for s in starts]
            chosen = np.unique(np.concatenate([head] + picked + [tail]))
        mask |= sel & np.isin(K[:, 1], chosen)
    return mask


def summarise_coverage(m):
    """What the corpus actually contains, in the terms the model cares about."""
    tot = sum(s["rows"] for s in m["shards"]) or 1
    hist = np.zeros(N_EXPERT_MODEL, dtype=np.int64)
    for s in m["shards"]:
        hist += np.asarray(s.get("expert_hist", [0] * N_EXPERT_MODEL), dtype=np.int64)
    w = lambda k: sum(s.get(k, 0) * s["rows"] for s in m["shards"]) / tot
    by_session, by_reg = {}, {}
    split_of = {p["id"]: p["split"] for p in m["prompts"]}
    sess_of = {p["id"]: p.get("session") for p in m["prompts"]}
    reg_of = {p["id"]: p["register"] for p in m["prompts"]}
    by_split = {}
    for s in m["shards"]:
        for pid in s["prompt_ids"]:
            by_session[sess_of.get(pid)] = by_session.get(sess_of.get(pid), 0) + 1
            by_reg[reg_of.get(pid)] = by_reg.get(reg_of.get(pid), 0) + 1
            by_split[split_of.get(pid)] = by_split.get(split_of.get(pid), 0) + 1
    return {
        "mean_lru_miss_66": w("mean_lru_miss_66"),
        "mean_lru_miss_32": w("mean_lru_miss_32"),
        "mean_repeat_hit": w("mean_repeat_hit"),
        "hard_row_pct": 100.0 * w("hard_row_fraction"),
        "expert_miss_rate_66_pct": 100.0 * w("mean_lru_miss_66") / 8.0,
        "expert_miss_rate_32_pct": 100.0 * w("mean_lru_miss_32") / 8.0,
        "min_expert_count": int(hist.min()), "max_expert_count": int(hist.max()),
        "expert_imbalance": float(hist.max() / max(hist.min(), 1)),
        "experts_never_seen": int((hist == 0).sum()),
        "prompts_by_session": by_session, "prompts_by_register": by_reg,
        "prompts_by_split": by_split,
    }


def _prompt_record(i, group, src, txt, splits):
    """One manifest entry. The text stored is the text actually fed to the model,
    truncation included -- a manifest that records the untruncated prompt would
    misdescribe every row captured from it."""
    plan = plan_for(i, txt) if PROMPT_SET == "real" else plan_for(i)
    if plan.get("truncated"):
        txt = txt[:plan["max_chars"]]
    return {"id": i, "register": group, "source": src, "split": splits[group],
            "session": plan["label"], "n_ctx": plan["n_ctx"],
            "n_predict": plan["n_predict"], "n_batch": plan["n_batch"],
            "truncated": bool(plan.get("truncated", False)), "text": txt}


def load_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            return json.load(f)
    return None


def save_manifest(m):
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, MANIFEST)          # atomic: a killed run never half-writes it


def validate(verbose=True):
    m = load_manifest()
    if not m:
        print("no manifest", file=sys.stderr)
        return 1
    bad, rows, bytes_ = 0, 0, 0
    for sh in m["shards"]:
        path = os.path.join(OUTDIR, sh["file"])
        if not os.path.exists(path):
            print(f"MISSING {sh['file']}")
            bad += 1
            continue
        size = os.path.getsize(path)
        if size != sh["bytes"]:
            print(f"SIZE MISMATCH {sh['file']}: {size} != {sh['bytes']}")
            bad += 1
            continue
        digest = sha256_file(path)
        if digest != sh["sha256"]:
            print(f"CHECKSUM MISMATCH {sh['file']}")
            bad += 1
            continue
        d = np.load(path)
        if d["hidden"].shape[0] != sh["rows"] or d["hidden"].shape[1] != m["d_model"]:
            print(f"SHAPE MISMATCH {sh['file']}: {d['hidden'].shape}")
            bad += 1
            continue
        if d["hidden"].dtype != np.float16 or d["experts"].dtype != np.int16:
            print(f"DTYPE MISMATCH {sh['file']}")
            bad += 1
            continue
        rows += sh["rows"]
        bytes_ += size
    if verbose:
        print(f"{len(m['shards'])} shards, {rows:,} rows, {bytes_ / 1e9:.2f} GB, "
              f"{bad} bad")
    return 1 if bad else 0


# ---- capture state, shared with the ggml callback -------------------------
N_EXPERT_MODEL = 128
BUF = {"hidden": [], "experts": []}
CUR = {"pid": 0, "pos": 0}


def make_cb():
    def cb(t_addr, ask, user_data):
        if ask or not t_addr:
            return True
        t = ctypes.cast(t_addr, ctypes.POINTER(GgmlTensor)).contents
        if not t.data:
            return True
        name = t.name.split(b"\x00", 1)[0].decode("utf-8", "ignore")
        if name.startswith("ffn_inp-"):
            L, dim, n = int(name.split("-")[-1]), int(t.ne[0]), int(t.ne[1])
            a = np.ctypeslib.as_array(
                ctypes.cast(t.data, ctypes.POINTER(ctypes.c_float)), shape=(n, dim))
            for i in range(n):
                BUF["hidden"].append((CUR["pid"], CUR["pos"] + i, L,
                                      a[i].astype(np.float16).copy()))
        elif name.startswith("ffn_moe_topk-"):
            L, k, n = int(name.split("-")[-1]), int(t.ne[0]), int(t.ne[1])
            a = np.ctypeslib.as_array(
                ctypes.cast(t.data, ctypes.POINTER(ctypes.c_int32)), shape=(n, k))
            for i in range(n):
                BUF["experts"].append((CUR["pid"], CUR["pos"] + i, L,
                                       a[i].astype(np.int16).copy()))
        return True

    return __import__("llama_cpp.llama_cpp", fromlist=["x"]).ggml_backend_sched_eval_callback(cb)


def flush_shard(m, idx):
    """Align hidden and experts on (prompt, pos, layer) and write one shard."""
    hk = {(p, t, l): v for p, t, l, v in BUF["hidden"]}
    ek = {(p, t, l): v for p, t, l, v in BUF["experts"]}
    keys = sorted(set(hk) & set(ek))
    BUF["hidden"].clear()
    BUF["experts"].clear()
    if not keys:
        return None
    H = np.stack([hk[k] for k in keys])
    E = np.stack([ek[k] for k in keys])
    K = np.array(keys, dtype=np.int32)
    miss, rep = label_shard(H, E, K, N_EXPERT_MODEL)
    n_full = int(H.shape[0])
    keep = subsample_positions(K, KEEP_POSITIONS)
    H, E, K = H[keep], E[keep], K[keep]
    miss = {c: v[keep] for c, v in miss.items()}
    rep = rep[keep]
    if not len(K):
        return None
    hist = np.bincount(E.reshape(-1).astype(np.int64), minlength=N_EXPERT_MODEL)
    name = f"shard-{idx:04d}.npz"
    path = os.path.join(OUTDIR, name)
    tmp = path + ".tmp"
    # np.savez appends ".npz" to a path that lacks it, which silently writes
    # shard-NNNN.npz.tmp.npz and leaves the rename pointing at nothing. Handing
    # it an open file object writes exactly where asked.
    with open(tmp, "wb") as fh:
        np.savez(fh, hidden=H, experts=E, keys=K,
                 lru_miss_66=miss[66], lru_miss_32=miss[32], repeat_hit=rep)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    rec = {"file": name, "rows": int(H.shape[0]), "rows_before_subsample": n_full,
           "prompt_ids": sorted({int(v) for v in np.unique(K[:, 0])}),
           "bytes": os.path.getsize(path), "sha256": sha256_file(path),
           "expert_hist": hist.tolist(),
           "mean_lru_miss_66": float(miss[66].mean()),
           "mean_lru_miss_32": float(miss[32].mean()),
           "mean_repeat_hit": float(rep.mean()),
           "hard_row_fraction": float((rep < 8).mean())}
    m["shards"].append(rec)
    return rec


def main():
    if "--validate" in sys.argv:
        sys.exit(validate())
    import llama_cpp.llama_cpp as lc

    os.makedirs(OUTDIR, exist_ok=True)
    corpus = build_corpus(PER_REGISTER)
    if LIMIT:
        corpus = corpus[:LIMIT]
    registers = {r for r, _, _ in corpus}
    splits = assign_splits(registers, SPLIT_SEED)

    m = load_manifest()
    if m and m.get("schema_version") != SCHEMA_VERSION:
        sys.exit(f"manifest schema {m.get('schema_version')} != {SCHEMA_VERSION}; "
                 "move it aside or delete the corpus")
    if m is None:
        m = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git_commit(),
            "model": {"path": MODEL, "bytes": os.path.getsize(MODEL),
                      "quant": "Q4_K_M", "arch": "qwen3moe",
                      "header_sha256": sha256_file(MODEL, limit=1 << 20)},
            "dataset_version": DATASET_VERSION,
            "capture": {"n_predict_override": N_PREDICT or None,
                        "session_plans": SESSION_PLANS,
                        "per_register": PER_REGISTER,
                        "prompt_set": PROMPT_SET,
                        "keep_positions": KEEP_POSITIONS,
                        "flush_rows": FLUSH_ROWS, "device": "cpu",
                        "sampling": "greedy"},
            "features": {"hidden_tensor": "ffn_inp-<L>", "hidden_dtype": "float16",
                         "expert_tensor": "ffn_moe_topk-<L>", "expert_dtype": "int16",
                         "keys": ["prompt_id", "position", "layer"], "top_k": 8,
                         "labels": {
                             "lru_miss_66": "uint8, experts of this row missing a 66-slot per-layer LRU",
                             "lru_miss_32": "uint8, the same at 32 slots (higher pressure)",
                             "repeat_hit": "uint8, overlap with the previous token's experts at this layer"}},
            "d_model": 2048, "n_expert": 128,
            "split_seed": SPLIT_SEED, "splits": splits,
            "prompts": [_prompt_record(i, r, src, txt, splits) 
                        for i, (r, txt, src) in enumerate(corpus)],
            "shards": [], "done_prompt_ids": [],
        }
        save_manifest(m)
        print(f"new corpus: {len(corpus)} prompts, {len(registers)} registers",
              file=sys.stderr)
        for s in ("train", "val", "test"):
            rs = sorted(r for r, v in splits.items() if v == s)
            print(f"  {s:<5} {rs}", file=sys.stderr)

    done = set(m["done_prompt_ids"])
    todo = [(i, r, txt) for i, (r, txt, _) in enumerate(corpus) if i not in done]
    if not todo:
        print("corpus already complete", file=sys.stderr)
        sys.exit(validate())
    print(f"{len(done)} prompts already captured, {len(todo)} to go",
          file=sys.stderr, flush=True)

    cb = make_cb()
    lc.llama_backend_init()
    mp = lc.llama_model_default_params()
    mp.n_gpu_layers = 0
    model = lc.llama_model_load_from_file(MODEL.encode(), mp)
    if not model:
        sys.exit("failed to load model")
    vocab = lc.llama_model_get_vocab(model)

    shard_idx = len(m["shards"])
    total_bytes = sum(s["bytes"] for s in m["shards"])
    t0 = time.time()
    pending = []
    for n_done, (pid, reg, prompt) in enumerate(todo):
        plan = plan_for(pid, prompt) if PROMPT_SET == "real" else plan_for(pid)
        if plan.get("truncated"):
            prompt = prompt[:plan["max_chars"]]
        n_pred = N_PREDICT or plan["n_predict"]
        cp = lc.llama_context_default_params()
        cp.n_ctx, cp.n_batch, cp.cb_eval = plan["n_ctx"], plan["n_batch"], cb
        # n_ubatch must not exceed n_batch; its default is 512 and the plans go
        # down to 64.
        cp.n_ubatch = min(plan["n_batch"], 512)
        ctx = lc.llama_init_from_model(model, cp)
        CUR["pid"], CUR["pos"] = pid, 0
        raw = prompt.encode()
        # The token buffer has to hold the whole prompt. llama_tokenize returns a
        # NEGATIVE count when it does not fit, and the v3 code passed a fixed
        # 256-token buffer -- fine for a corpus whose longest prompt was 24
        # tokens, and silently catastrophic for a 15,000-token GitHub issue:
        # llama_batch_get_one would then be handed a negative length.
        cap = max(64, len(raw) + 16)
        toks = (lc.llama_token * cap)()
        n = lc.llama_tokenize(vocab, raw, len(raw), toks, cap, True, True)
        if n < 0:
            print(f"  prompt {pid}: needs {-n} tokens, buffer {cap} -- skipped",
                  file=sys.stderr)
            lc.llama_free(ctx)
            continue
        # Second guard: the plan sizes n_ctx from a chars-per-token ESTIMATE, and
        # an estimate that is wrong in the unlucky direction would overflow the
        # context. Trim at token level so the prompt still fits.
        room = plan["n_ctx"] - n_pred - 8
        if n > room:
            n = room
        # Prefill in n_batch-sized chunks. llama_decode asserts
        # n_tokens_all <= cparams.n_batch, so a 2,000-token prompt submitted as
        # one batch aborts the process -- which a 14-token corpus never
        # discovered. CUR["pos"] is advanced per chunk because the observe
        # callback labels row i of the batch as position CUR["pos"] + i.
        n_batch = plan["n_batch"]
        pos = 0
        while pos < n:
            chunk = min(n_batch, n - pos)
            sub = (lc.llama_token * chunk)(*toks[pos:pos + chunk])
            CUR["pos"] = pos
            lc.llama_decode(ctx, lc.llama_batch_get_one(sub, chunk))
            pos += chunk
        CUR["pos"] = n
        sampler = lc.llama_sampler_chain_init(lc.llama_sampler_chain_default_params())
        lc.llama_sampler_chain_add(sampler, lc.llama_sampler_init_greedy())
        for _ in range(n_pred):
            tk = lc.llama_sampler_sample(sampler, ctx, -1)
            if lc.llama_vocab_is_eog(vocab, tk):
                break
            lc.llama_decode(ctx, lc.llama_batch_get_one((lc.llama_token * 1)(tk), 1))
            CUR["pos"] += 1
        lc.llama_sampler_free(sampler)
        lc.llama_free(ctx)
        pending.append(pid)

        if len(BUF["hidden"]) >= FLUSH_ROWS or n_done == len(todo) - 1:
            rec = flush_shard(m, shard_idx)
            if rec:
                shard_idx += 1
                total_bytes += rec["bytes"]
            m["done_prompt_ids"] = sorted(done | set(pending))
            done |= set(pending)
            pending = []
            save_manifest(m)
            el = time.time() - t0
            print(f"  shard {shard_idx:3d}  {len(done)}/{len(corpus)} prompts  "
                  f"{total_bytes / 1e9:.2f} GB  {el:.0f}s", file=sys.stderr, flush=True)
            if TARGET_BYTES and total_bytes >= TARGET_BYTES:
                print(f"reached target {TARGET_BYTES / 1e9:.1f} GB", file=sys.stderr)
                break

    lc.llama_model_free(model)
    m["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    m["total_rows"] = sum(s["rows"] for s in m["shards"])
    m["total_bytes"] = sum(s["bytes"] for s in m["shards"])
    m["coverage"] = summarise_coverage(m)
    save_manifest(m)
    cv = m["coverage"]
    print(f"coverage: {cv['expert_miss_rate_66_pct']:.1f}% of expert lookups miss a "
          f"66-slot cache ({cv['expert_miss_rate_32_pct']:.1f}% at 32), "
          f"{cv['hard_row_pct']:.1f}% of rows are not pure repeats,\n"
          f"          rarest expert seen {cv['min_expert_count']:,} times, "
          f"commonest {cv['max_expert_count']:,} ({cv['expert_imbalance']:.1f}x)",
          file=sys.stderr)
    print(f"\n{len(m['shards'])} shards, {m['total_rows']:,} rows, "
          f"{m['total_bytes'] / 1e9:.2f} GB", file=sys.stderr)
    sys.exit(validate())


if __name__ == "__main__":
    main()
