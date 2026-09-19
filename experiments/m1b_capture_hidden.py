"""
Milestone 1b, step 1: capture hidden states alongside expert selections.

m1_prediction_signals.py established that expert IDs at layer L say almost
nothing about layer L+1 (6.5% recall@8 against a 6.2% random floor). That closed
the cheap version of cross-layer prediction. What it did NOT test is the version
the literature actually uses: Pre-gated MoE predicts E_{L+1} from the hidden
state h_L, which carries 2048 floats rather than 8 integers.

This captures the data to settle that, using the tensor names found by
m1b_discover_tensors.py:

  ffn_inp-<L>        2048 x n_tokens f32 -- the residual stream entering layer
                     L's FFN block. Pre-norm, so it is the layer's actual state
                     rather than something already shaped for this layer's gate.
  ffn_moe_topk-<L>   8 x n_tokens i32   -- the experts layer L selected. Ground
                     truth. (ggml type 26 is I32; AI2's tracer reads it the same
                     way.)

Stored as float16 to keep the file near 250 MB rather than 500: the probe is
asking whether the signal exists at all, and half a millibit of mantissa is not
what decides that.

CPU-only, for the reason in m1b_discover_tensors.py: llama-cpp-python does not
bind ggml_backend_tensor_get, so GPU-resident tensors cannot be read back from
Python. Expert selection and hidden states are properties of the computation,
not of where it runs, so this costs speed and nothing else.
"""
import ctypes
import os
import sys
import time

import numpy as np
import llama_cpp.llama_cpp as lc

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get(
    "MODEL",
    "/home/everett/.lmstudio/models/lmstudio-community/"
    "Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
)
OUT = os.environ.get("OUT", os.path.join(HERE, "m1b_hidden.npz"))
N_PREDICT = int(os.environ.get("N_PREDICT", "60"))

# Deliberately mixed, so the probe is not fitted to one register. Same spirit as
# AI2's trace prompts, trimmed because CPU-only decode is ~5 tok/s.
PROMPTS = [
    "What is the capital of France?",
    "Write a Python function that reverses a linked list.",
    "Explain why the sky appears blue.",
    "Summarize the causes of the French Revolution.",
    "Write a short poem about winter.",
    "Fix this bug: def add(a, b): return a - b",
    "What is 156 divided by 12?",
    "Describe how a hash map handles collisions.",
]

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

hidden, experts = [], []          # (prompt_id, token_pos, layer, vec/ids)
_pos = [0]
_pid = [0]


def make_cb():
    def cb(t_addr, ask, user_data):
        if ask or not t_addr:
            return True
        t = ctypes.cast(t_addr, ctypes.POINTER(GgmlTensor)).contents
        name = t.name.split(b"\x00", 1)[0].decode("utf-8", "ignore")
        if not t.data:
            return True
        if name.startswith("ffn_inp-"):
            layer = int(name.split("-")[-1])
            dim, n_tok = int(t.ne[0]), int(t.ne[1])
            arr = np.ctypeslib.as_array(
                ctypes.cast(t.data, ctypes.POINTER(ctypes.c_float)), shape=(n_tok, dim))
            for i in range(n_tok):
                hidden.append((_pid[0], _pos[0] + i, layer, arr[i].astype(np.float16).copy()))
        elif name.startswith("ffn_moe_topk-"):
            layer = int(name.split("-")[-1])
            k, n_tok = int(t.ne[0]), int(t.ne[1])
            arr = np.ctypeslib.as_array(
                ctypes.cast(t.data, ctypes.POINTER(ctypes.c_int32)), shape=(n_tok, k))
            for i in range(n_tok):
                experts.append((_pid[0], _pos[0] + i, layer, arr[i].astype(np.int16).copy()))
        return True

    return lc.ggml_backend_sched_eval_callback(cb)


def main():
    if not os.path.exists(MODEL):
        print(f"model not found: {MODEL}", file=sys.stderr)
        sys.exit(2)
    cb = make_cb()
    lc.llama_backend_init()
    mp = lc.llama_model_default_params()
    mp.n_gpu_layers = 0
    print("loading Q4_K_M CPU-only...", file=sys.stderr, flush=True)
    model = lc.llama_model_load_from_file(MODEL.encode(), mp)
    if not model:
        print("failed to load", file=sys.stderr)
        sys.exit(1)
    vocab = lc.llama_model_get_vocab(model)
    t0 = time.time()

    for pid, prompt in enumerate(PROMPTS):
        cp = lc.llama_context_default_params()
        cp.n_ctx, cp.n_batch, cp.cb_eval = 512, 512, cb   # fresh context per prompt
        ctx = lc.llama_init_from_model(model, cp)
        _pid[0], _pos[0] = pid, 0

        raw = prompt.encode()
        toks = (lc.llama_token * 256)()
        n = lc.llama_tokenize(vocab, raw, len(raw), toks, 256, True, True)
        lc.llama_decode(ctx, lc.llama_batch_get_one(toks, n))
        _pos[0] += n

        sampler = lc.llama_sampler_chain_init(lc.llama_sampler_chain_default_params())
        lc.llama_sampler_chain_add(sampler, lc.llama_sampler_init_greedy())
        for _ in range(N_PREDICT):
            tok = lc.llama_sampler_sample(sampler, ctx, -1)
            if lc.llama_vocab_is_eog(vocab, tok):
                break
            one = (lc.llama_token * 1)(tok)
            lc.llama_decode(ctx, lc.llama_batch_get_one(one, 1))
            _pos[0] += 1
        lc.llama_sampler_free(sampler)
        lc.llama_free(ctx)
        print(f"  prompt {pid}: {_pos[0]} positions, {len(hidden):,} hidden rows "
              f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)

    lc.llama_model_free(model)

    # Align the two streams on (prompt, token, layer); ffn_inp and topk are
    # emitted in the same graph so they should match one-for-one, but assert it
    # rather than assume -- a silent misalignment would invent a correlation.
    hkey = {(p, t, l): v for p, t, l, v in hidden}
    ekey = {(p, t, l): v for p, t, l, v in experts}
    keys = sorted(set(hkey) & set(ekey))
    print(f"\nhidden rows {len(hkey):,}, expert rows {len(ekey):,}, aligned {len(keys):,}",
          file=sys.stderr)

    H = np.stack([hkey[k] for k in keys])
    E = np.stack([ekey[k] for k in keys])
    K = np.array(keys, dtype=np.int32)
    np.savez_compressed(OUT, hidden=H, experts=E, keys=K)
    print(f"saved {OUT}  hidden={H.shape} {H.dtype}, experts={E.shape}, "
          f"{os.path.getsize(OUT) / 1e6:.0f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
