"""
Milestone 1b, step 0: what tensors can we actually read from the MoE block?

experiments/m1_prediction_signals.py showed that expert IDs at layer L carry
almost nothing about layer L+1 (6.5% vs a 6.2% floor). The one remaining route
to a trained predictor is the hidden state: Pre-gated MoE predicts E_{L+1} from
h_L, which carries far more than 8 integers.

Before capturing hidden states, find out what is on offer. AI2's expert_trace.py
matched exactly one name, `ffn_moe_topk-<il>`; llama.cpp names many more tensors
in that graph and the useful ones are whatever feeds the router. This logs every
distinct tensor name in one short forward pass, with shapes and dtypes, so the
capture script can pick real names instead of guessed ones.

CPU-only for the same reason as AI2's tracer: llama-cpp-python does not bind
ggml_backend_tensor_get, so a GPU-resident tensor cannot be copied back from
Python. CPU-only makes every ->data pointer host memory, readable via ctypes.
"""
import ctypes
import os
import sys
from collections import OrderedDict

import llama_cpp.llama_cpp as lc

MODEL = os.environ.get(
    "MODEL",
    "/home/everett/.lmstudio/models/lmstudio-community/"
    "Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
)
GGML_MAX_DIMS, GGML_MAX_SRC, GGML_MAX_OP_PARAMS_I32, GGML_MAX_NAME = 4, 10, 16, 64
# ggml_type enum -> name, only the ones we might meet here.
TYPES = {0: "f32", 1: "f16", 26: "i32", 30: "bf16"}  # ggml_type enum: I32 is 26


class GgmlTensor(ctypes.Structure):
    pass


GgmlTensor._fields_ = [
    ("type", ctypes.c_int),
    ("buffer", ctypes.c_void_p),
    ("ne", ctypes.c_int64 * GGML_MAX_DIMS),
    ("nb", ctypes.c_size_t * GGML_MAX_DIMS),
    ("op", ctypes.c_int),
    ("op_params", ctypes.c_int32 * GGML_MAX_OP_PARAMS_I32),
    ("flags", ctypes.c_int32),
    ("src", ctypes.POINTER(GgmlTensor) * GGML_MAX_SRC),
    ("view_src", ctypes.POINTER(GgmlTensor)),
    ("view_offs", ctypes.c_size_t),
    ("data", ctypes.c_void_p),
    ("name", ctypes.c_char * GGML_MAX_NAME),
    ("extra", ctypes.c_void_p),
    ("padding", ctypes.c_char * 8),
]

seen = OrderedDict()


def make_cb():
    def cb(t_addr, ask, user_data):
        if ask or not t_addr:
            return True
        t = ctypes.cast(t_addr, ctypes.POINTER(GgmlTensor)).contents
        name = t.name.split(b"\x00", 1)[0].decode("utf-8", "ignore")
        # Collapse the per-layer suffix so 48 layers do not produce 48 entries.
        key = name.rsplit("-", 1)[0] if name.rsplit("-", 1)[-1].isdigit() else name
        if key not in seen:
            ne = [t.ne[i] for i in range(GGML_MAX_DIMS)]
            seen[key] = {
                "example": name,
                "shape": [n for n in ne if n > 1] or [1],
                "type": TYPES.get(t.type, f"type{t.type}"),
                "readable": bool(t.data),
            }
        return True

    return lc.ggml_backend_sched_eval_callback(cb)


def main():
    if not os.path.exists(MODEL):
        print(f"model not found: {MODEL}", file=sys.stderr)
        sys.exit(2)
    cb = make_cb()
    p = lc.llama_model_default_params()
    p.n_gpu_layers = 0
    print("loading (CPU-only, this takes a minute)...", file=sys.stderr, flush=True)
    lc.llama_backend_init()
    model = lc.llama_model_load_from_file(MODEL.encode(), p)
    if not model:
        print("failed to load model", file=sys.stderr)
        sys.exit(1)
    cp = lc.llama_context_default_params()
    cp.n_ctx, cp.n_batch, cp.cb_eval = 256, 256, cb
    ctx = lc.llama_init_from_model(model, cp)

    vocab = lc.llama_model_get_vocab(model)
    text = b"The capital of France is"
    toks = (lc.llama_token * 64)()
    n = lc.llama_tokenize(vocab, text, len(text), toks, 64, True, True)
    batch = lc.llama_batch_get_one(toks, n)
    print(f"one forward pass over {n} tokens...", file=sys.stderr, flush=True)
    lc.llama_decode(ctx, batch)

    print(f"\n=== TENSORS VISIBLE IN THE EVAL CALLBACK ({len(seen)} distinct names) ===\n")
    print("%-34s %-16s %-7s %s" % ("name (layer suffix stripped)", "shape", "dtype", "data?"))
    for k, v in seen.items():
        print("%-34s %-16s %-7s %s" % (k, "x".join(map(str, v["shape"])), v["type"],
                                       "yes" if v["readable"] else "NO"))
    print("\nlooking for the router's input and output:")
    for k, v in seen.items():
        if "moe" in k or "ffn_norm" in k or "ffn_inp" in k:
            print(f"   {k:<32} {'x'.join(map(str, v['shape'])):<14} {v['type']}")

    lc.llama_free(ctx)
    lc.llama_model_free(model)


if __name__ == "__main__":
    main()
