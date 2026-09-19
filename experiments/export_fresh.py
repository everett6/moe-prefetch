"""
Export the fresh supervised predictor to the runtime's MOEP format.

The trained model is `LinearCtx`: four per-layer embedding tables, one per
feature block, each Embedding(n_layers * 128, 128). For layer L, block b, the
rows [L*128 : (L+1)*128] are exactly a 128x128 weight matrix -- so the model IS
already the flat per-layer form the runtime wants, and exporting is a reshape
rather than a conversion.

The block ORDER in the file is the contract. It is written into the header block
table and the loader refuses a file whose blocks do not tile the feature vector
exactly, because a silently permuted feature order does not crash, it just
predicts worse -- which is the failure this project has twice nearly reported as
a finding.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import models_deep as M  # noqa: E402
from d2_export import (write_binary, read_binary, BLOCK_PREV, BLOCK_CUR,  # noqa: E402
                       BLOCK_BELOW, BLOCK_SELF_PREV, BLOCK_NAME, N_EXPERT)

BLOCK_OF = {"prev": BLOCK_PREV, "cur": BLOCK_CUR,
            "below": BLOCK_BELOW, "self_prev": BLOCK_SELF_PREV}


def export(ckpt_path, out_path, n_layers=48):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    assert cfg["arch"] == "linearctx", f"exporter handles linearctx, not {cfg['arch']}"
    model = M.build("linearctx", n_layers, context=cfg.get("context", True))
    model.load_state_dict(ck["state"])
    names = model.names                       # the order the model itself uses
    blocks = [BLOCK_OF[n] for n in names]
    dim = len(names) * N_EXPERT

    W, bias, prior = {}, {}, {}
    present = sorted({L for L in range(n_layers - 1)})
    for L in present:
        parts = []
        for n in names:
            w = model.tables[n].weight.detach().numpy()      # (n_layers*128, 128)
            parts.append(w[L * N_EXPERT:(L + 1) * N_EXPERT])
        W[L] = np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)
        bias[L] = model.bias.detach().numpy()[L].astype(np.float32)
        prior[L] = 0.0                       # the bias subsumes it; kept for format compat
    f = {"W": W, "bias": bias, "prior": prior, "blocks": blocks, "dim": dim}
    write_binary(out_path, f, None, None, None, n_layers, 0)
    return f, names, ck


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "artifacts", "ckpt", "FRESH_SUPERVISED-linearctx-lr1e2.pt")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        ROOT, "models", "predictor-fresh.bin")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    f, names, ck = export(ckpt, out)
    print(f"exported {os.path.basename(ckpt)} -> {out}")
    print(f"  blocks: {[BLOCK_NAME[b] for b in f['blocks']]}")
    print(f"  feature dim {f['dim']}, {len(f['W'])} layers, "
          f"{os.path.getsize(out) / 1e6:.1f} MB")

    # round-trip: reload and confirm the reshape preserved everything
    chk = read_binary(out)
    assert chk["blocks"] == f["blocks"], (chk["blocks"], f["blocks"])
    assert chk["feat_dim"] == f["dim"]
    for L in f["W"]:
        assert np.array_equal(chk["W"][L], f["W"][L]), f"layer {L} weights changed"
        assert np.array_equal(chk["bias"][L], f["bias"][L]), f"layer {L} bias changed"
    print("  round-trip OK: weights and biases bit-identical")

    # and that the flat form reproduces the torch model's scores exactly
    n_layers = 48
    model = M.build("linearctx", n_layers, context=ck["cfg"].get("context", True))
    model.load_state_dict(ck["state"])
    model.eval()
    rng = np.random.RandomState(0)
    worst = 0.0
    with torch.no_grad():
        for _ in range(200):
            L = int(rng.randint(0, n_layers - 1))
            ids = {n: sorted(rng.choice(N_EXPERT, 8, replace=False).tolist()) for n in names}
            batch = {n: torch.tensor([ids[n]]) for n in names}
            batch["layer"] = torch.tensor([L])
            ref = model(batch)[0].numpy()[0]
            x = np.zeros(f["dim"], dtype=np.float32)
            for (b, off, size) in chk["table"]:
                name = [n for n in names if BLOCK_OF[n] == b][0]
                x[off + np.asarray(ids[name])] = 1.0
            got = x @ chk["W"][L] + chk["bias"][L]
            worst = max(worst, float(np.max(np.abs(ref - got))))
    print(f"  flat form vs torch: max |diff| = {worst:.2e}")
    assert worst < 1e-4, "the exported matrices do not reproduce the model"
    print("  PASS")


if __name__ == "__main__":
    main()
