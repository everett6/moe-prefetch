"""
D2: train the A2 predictor properly, measure what each feature is worth to a
*deployable* model, and write it out as a flat binary C++ can load.

Two things this does that A2 did not.

**It saves the weights.** `a2_richer_inputs.py` fitted per-layer ridge inside a
loop and threw every matrix away, keeping only the recall number. There is no
trained artifact in this repo -- the 68.9% is real and reproducible, but nothing
was ever serialised. D2 has to fit it again anyway, so it fits it once and keeps
it.

**It asks what the hidden state costs.** The A2 feature vector is

    [ PCA(h_norm)  256 | prev 128 | cur 128 ]

and the last two blocks are free in llama.cpp: `moe_obs_cb` already sees every
expert id on the host, so `prev` (the previous token's experts at L+1) and `cur`
(this token's experts at L) are bookkeeping. `PCA(h_norm)` is not free. The
residual stream lives on the GPU, so using it means a 8 KiB device-to-host copy
and a synchronisation *per layer* -- 47 of them inside the token it is trying to
speed up.

So the ablation below is not curiosity, it decides the integration: if
`prev+cur` alone lands close to the full set, the predictor runs entirely from
data the cache already has and D3 is a hundred lines of bookkeeping. If it does
not, D3 has to pay for the readback and that cost has to come off the +20.

Same register-level split as A2 (5 train / 1 val / 2 test). A2 reported 68.9%
with a repeat prior applied *after* scoring -- `z(scores) + w * prev` -- which
was never part of any artifact, so the number was not reproducible by anything
that loaded a saved model. Rather than drop it, it is exported: a per-layer
scalar chosen on the validation register, applied to the z-scored logits. Ten
lines in C, and it makes the published figure and the shipped artifact the same
thing.
"""
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get("DATA", os.path.join(HERE, "m2_hidden.npz"))
OUT = os.path.join(HERE, "d2_export_result.json")
MODEL_DIR = os.path.join(ROOT, "models")
N_COMP = int(os.environ.get("N_COMP", "256"))
LAMBDAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
N_EXPERT, K = 128, 8

MAGIC = b"MOEP"
VERSION = 2
BLOCK_PCA, BLOCK_PREV, BLOCK_CUR = 0, 1, 2
BLOCK_NAME = {BLOCK_PCA: "pca_h_norm", BLOCK_PREV: "prev_experts_L+1", BLOCK_CUR: "cur_experts_L"}


PRIOR_WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def recall_at_k(scores, truth, k=K):
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    return np.mean([len(set(t) & set(p)) for t, p in zip(truth, top)]) / k


def zs(a):
    """Z-score each row. The prior is added to scores whose scale varies by
    layer, so the weight only means the same thing across layers once the
    logits are standardised."""
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)


def main():
    d = np.load(DATA, allow_pickle=True)
    H = d["hidden"].astype(np.float32)
    E = d["experts"].astype(np.int64)
    keys = d["keys"]
    reg = d["register"].astype(str)
    idx = {(p, t, l): i for i, (p, t, l) in enumerate(zip(keys[:, 0], keys[:, 1], keys[:, 2]))}
    n_layers = int(keys[:, 2].max()) + 1
    d_model = H.shape[1]
    regs = sorted(set(reg))
    train_r, val_r, test_r = regs[:5], regs[5:6], regs[6:]
    print(f"{len(H):,} rows, {n_layers} layers, d={d_model}; "
          f"train {train_r} val {val_r} test {test_r}", file=sys.stderr)

    # ---- pairs: h at layer L predicts experts at layer L+1, same token
    src, dst, lay = [], [], []
    for (p, t, l), i in idx.items():
        j = idx.get((p, t, l + 1))
        if j is not None:
            src.append(i); dst.append(j); lay.append(l)
    src = np.array(src); dst = np.array(dst); lay = np.array(lay)
    r = reg[src]
    m_tr, m_va, m_te = np.isin(r, train_r), np.isin(r, val_r), np.isin(r, test_r)
    print(f"{len(src):,} (layer, token) pairs", file=sys.stderr)

    # ---- PCA on the L2-normalised hidden state, fitted on train rows only
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-6)
    rows = np.unique(src[m_tr])
    mean = Hn[rows].mean(0)
    Xc = Hn[rows] - mean
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w, V = np.linalg.eigh(C.astype(np.float64))
    comp = V[:, ::-1][:, :N_COMP].astype(np.float32)
    proj_tr = (Hn[rows] - mean) @ comp
    scale = (proj_tr.std(0) + 1e-6).astype(np.float32)
    P = ((Hn - mean) @ comp) / scale
    del Hn, Xc, C, V
    print(f"PCA: {d_model} -> {N_COMP}, {100 * w[::-1][:N_COMP].sum() / w.sum():.1f}% "
          "of variance on train rows", file=sys.stderr)

    # ---- the two host-free blocks
    prev = np.zeros((len(src), N_EXPERT), dtype=np.float32)
    cur = np.zeros((len(src), N_EXPERT), dtype=np.float32)
    for m, (i, L) in enumerate(zip(src, lay)):
        p, t = keys[i][:2]
        j = idx.get((p, t - 1, L + 1))
        if j is not None:
            prev[m, E[j]] = 1.0
        cur[m, E[i]] = 1.0

    Y = np.zeros((len(dst), N_EXPERT), dtype=np.float32)
    Y[np.arange(len(dst))[:, None], E[dst]] = 1.0

    # ---- fit one feature set across all layers, return per-layer W + test recall
    SETS = {
        "h_norm+prev+cur": [BLOCK_PCA, BLOCK_PREV, BLOCK_CUR],
        "prev+cur":        [BLOCK_PREV, BLOCK_CUR],
        "cur":             [BLOCK_CUR],
        "prev":            [BLOCK_PREV],
        "h_norm":          [BLOCK_PCA],
    }
    block_data = {BLOCK_PCA: P, BLOCK_PREV: prev, BLOCK_CUR: cur}

    def features(blocks, sel):
        cols = []
        for b in blocks:
            cols.append(block_data[b][src[sel]] if b == BLOCK_PCA else block_data[b][sel])
        return np.concatenate(cols, axis=1)

    fitted = {}
    for name, blocks in SETS.items():
        W_by_layer, prior_by_layer, va_rec, te_rec, lam_pick = {}, {}, [], [], []
        for L in range(n_layers - 1):
            sel_tr = m_tr & (lay == L)
            sel_va = m_va & (lay == L)
            sel_te = m_te & (lay == L)
            if min(sel_tr.sum(), sel_va.sum(), sel_te.sum()) < 40:
                continue
            Xtr, Ytr = features(blocks, sel_tr), Y[sel_tr]
            Xva, Xte = features(blocks, sel_va), features(blocks, sel_te)
            Eva, Ete = E[dst][sel_va], E[dst][sel_te]
            G = Xtr.T @ Xtr
            XtY = Xtr.T @ Ytr
            best = (-1.0, None, None)
            for lam in LAMBDAS:
                Wl = np.linalg.solve(G + lam * np.eye(Xtr.shape[1], dtype=np.float32), XtY)
                v = recall_at_k(Xva @ Wl, Eva)
                if v > best[0]:
                    best = (v, Wl, lam)
            v, Wl, lam = best
            # repeat prior: weight chosen on validation only, then applied to test
            pv, pt = prev[sel_va], prev[sel_te]
            sva, ste = zs(Xva @ Wl), zs(Xte @ Wl)
            bw = max(PRIOR_WEIGHTS, key=lambda w: recall_at_k(sva + w * pv, Eva))
            W_by_layer[L] = Wl.astype(np.float32)
            prior_by_layer[L] = float(bw)
            va_rec.append(recall_at_k(sva + bw * pv, Eva))
            te_rec.append(recall_at_k(ste + bw * pt, Ete))
            lam_pick.append(lam)
        fitted[name] = {"W": W_by_layer, "prior": prior_by_layer,
                        "val": float(np.mean(va_rec)),
                        "test": float(np.mean(te_rec)), "per_layer_test": te_rec,
                        "blocks": blocks, "dim": sum(N_COMP if b == BLOCK_PCA else N_EXPERT
                                                     for b in blocks),
                        "lambdas": lam_pick}
        print(f"  {name:<18} val {100 * fitted[name]['val']:.1f}%  "
              f"test {100 * fitted[name]['test']:.1f}%  "
              f"({len(W_by_layer)} layers, d={fitted[name]['dim']})",
              file=sys.stderr, flush=True)

    # naive-repeat floor, for scale
    base = recall_at_k(prev[m_te] + 1e-6 * np.random.RandomState(0).randn(
        int(m_te.sum()), N_EXPERT).astype(np.float32), E[dst][m_te])

    print(f"\n=== D2: WHAT THE HIDDEN STATE IS WORTH (test {test_r}, no repeat prior) ===\n")
    print("%-20s %6s %10s %10s %12s" % ("features", "dims", "recall@8", "vs full", "host-free?"))
    full = fitted["h_norm+prev+cur"]["test"]
    for name in SETS:
        f = fitted[name]
        free = "yes" if BLOCK_PCA not in f["blocks"] else "no - needs D2H"
        print("%-20s %6d %9.1f%% %9.1f %14s"
              % (name, f["dim"], 100 * f["test"], 100 * (f["test"] - full), free))
    print("%-20s %6s %9.1f%% %9.1f %14s" % ("naive-repeat", "-", 100 * base,
                                            100 * (base - full), "yes"))

    gap = full - fitted["prev+cur"]["test"]
    print(f"\nThe hidden state is worth {100 * gap:+.1f} points over what the cache "
          "already knows.")

    # ---- export both: the full model, and the host-free one
    os.makedirs(MODEL_DIR, exist_ok=True)
    exported = {}
    for name in ("h_norm+prev+cur", "prev+cur"):
        f = fitted[name]
        path = os.path.join(MODEL_DIR, ("predictor-full.bin" if BLOCK_PCA in f["blocks"]
                                        else "predictor-hostfree.bin"))
        write_binary(path, f, mean, comp, scale, n_layers, d_model)
        sz = os.path.getsize(path)
        n_par = sum(W.size for W in f["W"].values()) + (
            comp.size + mean.size + scale.size if BLOCK_PCA in f["blocks"] else 0)
        exported[name] = {"path": os.path.relpath(path, ROOT), "bytes": sz,
                          "params": int(n_par), "test": f["test"], "dim": f["dim"]}
        print(f"\nwrote {path}  ({sz / 1e6:.1f} MB, {n_par / 1e6:.2f} M params)")

        # round-trip: reload and confirm the scores are bit-identical
        chk = read_binary(path)
        L0 = sorted(f["W"])[len(f["W"]) // 2]
        sel = m_te & (lay == L0)
        Xte = features(f["blocks"], sel)
        s_ref = Xte @ f["W"][L0]
        s_rt = Xte @ chk["W"][L0]
        assert np.array_equal(s_ref, s_rt), "round-trip changed the scores"
        assert chk["feat_dim"] == f["dim"] and chk["blocks"] == f["blocks"]
        assert chk["prior"] == {int(k): float(v) for k, v in f["prior"].items()}
        print(f"  round-trip OK: layer {L0} scores bit-identical, "
              f"{len(chk['W'])} layers, blocks {[BLOCK_NAME[b] for b in chk['blocks']]}")

    json.dump({"sets": {k: {"val": v["val"], "test": v["test"], "dim": v["dim"],
                            "n_layers": len(v["W"])} for k, v in fitted.items()},
               "naive_repeat": float(base), "hidden_state_worth": float(gap),
               "exported": exported, "n_comp": N_COMP, "test_registers": test_r,
               "prior_weights": {k: v["prior"] for k, v in fitted.items()}},
              open(OUT, "w"), indent=1, default=float)
    print(f"\nsaved {OUT}")


# ---------------------------------------------------------------------------
# The file format. Deliberately dull: one magic, fixed-width little-endian
# integers, then raw float32 in C order. The only interesting field is the block
# table, which spells out the feature layout so the loader cannot silently
# disagree with the trainer about what column means what -- getting that wrong
# does not crash, it just quietly predicts worse.
# ---------------------------------------------------------------------------

def write_binary(path, f, mean, comp, scale, n_layers, d_model):
    blocks = f["blocks"]
    has_pca = BLOCK_PCA in blocks
    layers = sorted(f["W"])
    off, table = 0, []
    for b in blocks:
        size = N_COMP if b == BLOCK_PCA else N_EXPERT
        table.append((b, off, size))
        off += size
    assert off == f["dim"]

    with open(path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<11I", VERSION, d_model if has_pca else 0,
                             N_COMP if has_pca else 0, N_EXPERT, n_layers, K,
                             f["dim"], len(layers), len(table),
                             1 if has_pca else 0, 0))
        for b, o, s in table:
            fh.write(struct.pack("<3I", b, o, s))
        if has_pca:
            fh.write(np.ascontiguousarray(mean, dtype="<f4").tobytes())
            fh.write(np.ascontiguousarray(comp, dtype="<f4").tobytes())
            fh.write(np.ascontiguousarray(scale, dtype="<f4").tobytes())
        fh.write(np.asarray(layers, dtype="<i4").tobytes())
        fh.write(np.asarray([f["prior"][L] for L in layers], dtype="<f4").tobytes())
        for L in layers:
            fh.write(np.ascontiguousarray(f["W"][L], dtype="<f4").tobytes())


def read_binary(path):
    with open(path, "rb") as fh:
        assert fh.read(4) == MAGIC, "not a MOEP file"
        (ver, d_model, n_comp, n_expert, n_layers, k, feat_dim,
         n_present, n_blocks, has_pca, _) = struct.unpack("<11I", fh.read(44))
        assert ver == VERSION
        table = [struct.unpack("<3I", fh.read(12)) for _ in range(n_blocks)]
        out = {"feat_dim": feat_dim, "blocks": [b for b, _, _ in table],
               "table": table, "n_expert": n_expert, "k": k}
        if has_pca:
            out["mean"] = np.frombuffer(fh.read(4 * d_model), dtype="<f4")
            out["comp"] = np.frombuffer(fh.read(4 * d_model * n_comp),
                                        dtype="<f4").reshape(d_model, n_comp)
            out["scale"] = np.frombuffer(fh.read(4 * n_comp), dtype="<f4")
        layers = np.frombuffer(fh.read(4 * n_present), dtype="<i4")
        priors = np.frombuffer(fh.read(4 * n_present), dtype="<f4")
        out["prior"] = {int(L): float(w) for L, w in zip(layers, priors)}
        out["W"] = {int(L): np.frombuffer(fh.read(4 * feat_dim * n_expert),
                                          dtype="<f4").reshape(feat_dim, n_expert)
                    for L in layers}
    return out


if __name__ == "__main__":
    main()
