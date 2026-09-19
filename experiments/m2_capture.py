"""
Milestone 2, step 1: capture enough data to trust the milestone 1 number.

Milestone 1 reached 51.3% recall@8 on 8 prompts and 557 token positions, with
the ridge strength and ensemble weight both chosen on the held-out set. That
makes it a ceiling, not an estimate. Two things have to change before it means
anything: more data, and a split that does not let hyperparameters see the test.

This handles the first. Capture turned out to cost ~25 s for 557 positions, so
there is no reason to be stingy -- 40 prompts across eight registers, 80 tokens
each, is ~3,200 positions and ~150K (token, layer) rows for about two minutes of
CPU-only decode.

Register coverage is deliberate. Expert routing is a function of the hidden
state, and the hidden state is a function of what is being written about; a
probe fitted only to Q&A prose would be flattered by a test set of the same. The
eight groups below are held out whole, by group, in m2_probe.py.

Same tensors and same mechanism as m1b_capture_hidden.py:
  ffn_inp-<L>       2048 x n_tok f32, the residual stream entering layer L's FFN
  ffn_moe_topk-<L>  8 x n_tok i32, the experts layer L chose (ggml type 26 = I32)
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
OUT = os.environ.get("OUT", os.path.join(HERE, "m2_hidden.npz"))
N_PREDICT = int(os.environ.get("N_PREDICT", "80"))

# (register, prompt). Registers are the unit of the train/val/test split.
PROMPTS = [
    ("code", "Write a Python function that reverses a singly linked list."),
    ("code", "Implement binary search over a sorted list, with docstring."),
    ("code", "Write a class for a least-recently-used cache with get and put."),
    ("code", "Fix this bug: def add(a, b): return a - b"),
    ("code", "Write a SQL query joining orders and customers by id."),
    ("math", "What is 156 divided by 12?"),
    ("math", "A train travels 240 km in 3 hours. What is its average speed?"),
    ("math", "Compute the derivative of x^3 + 2x^2 - 5x + 1."),
    ("math", "If 5 machines take 5 minutes to make 5 widgets, how long for 100?"),
    ("math", "Explain why the square root of 2 is irrational."),
    ("science", "Explain why the sky appears blue."),
    ("science", "How do vaccines train the immune system?"),
    ("science", "Describe the carbon cycle in a few paragraphs."),
    ("science", "What causes the seasons on Earth?"),
    ("science", "Explain how a refrigerator moves heat against a gradient."),
    ("history", "Summarize the causes of the French Revolution."),
    ("history", "What led to the fall of the Western Roman Empire?"),
    ("history", "Describe the significance of the printing press."),
    ("history", "What were the main outcomes of the Congress of Vienna?"),
    ("history", "Explain the origins of the Silk Road."),
    ("creative", "Write a short poem about winter."),
    ("creative", "Write a story about a robot discovering music."),
    ("creative", "Write a fable about a fox and a river."),
    ("creative", "Describe a lighthouse at dawn, in vivid prose."),
    ("creative", "Write a dialogue between two old friends reuniting."),
    ("factual", "What is the capital of France?"),
    ("factual", "Who wrote Pride and Prejudice?"),
    ("factual", "What year did the Berlin Wall fall?"),
    ("factual", "How many continents are there?"),
    ("factual", "What is the largest ocean on Earth?"),
    ("technical", "Describe how a hash map handles collisions."),
    ("technical", "Explain the CAP theorem in two sentences."),
    ("technical", "What is the difference between TCP and UDP?"),
    ("technical", "Explain how DNS resolves a domain name."),
    ("technical", "What does a load balancer do, and why?"),
    ("reasoning", "Should a small team prefer a monolith or microservices? Why?"),
    ("reasoning", "Compare electric and hydrogen cars for a cold climate."),
    ("reasoning", "Argue both sides of remote versus office work."),
    ("reasoning", "How would you decide whether to rewrite or refactor a system?"),
    ("reasoning", "What are the trade-offs of renting versus buying a home?"),
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

hidden, experts = [], []
_pos, _pid = [0], [0]


def make_cb():
    def cb(t_addr, ask, user_data):
        if ask or not t_addr:
            return True
        t = ctypes.cast(t_addr, ctypes.POINTER(GgmlTensor)).contents
        name = t.name.split(b"\x00", 1)[0].decode("utf-8", "ignore")
        if not t.data:
            return True
        if name.startswith("ffn_inp-"):
            L, dim, n = int(name.split("-")[-1]), int(t.ne[0]), int(t.ne[1])
            a = np.ctypeslib.as_array(
                ctypes.cast(t.data, ctypes.POINTER(ctypes.c_float)), shape=(n, dim))
            for i in range(n):
                hidden.append((_pid[0], _pos[0] + i, L, a[i].astype(np.float16).copy()))
        elif name.startswith("ffn_moe_topk-"):
            L, k, n = int(name.split("-")[-1]), int(t.ne[0]), int(t.ne[1])
            a = np.ctypeslib.as_array(
                ctypes.cast(t.data, ctypes.POINTER(ctypes.c_int32)), shape=(n, k))
            for i in range(n):
                experts.append((_pid[0], _pos[0] + i, L, a[i].astype(np.int16).copy()))
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
    print(f"loading Q4_K_M CPU-only, {len(PROMPTS)} prompts x {N_PREDICT} tokens...",
          file=sys.stderr, flush=True)
    model = lc.llama_model_load_from_file(MODEL.encode(), mp)
    if not model:
        sys.exit("failed to load")
    vocab = lc.llama_model_get_vocab(model)
    t0 = time.time()

    for pid, (register, prompt) in enumerate(PROMPTS):
        cp = lc.llama_context_default_params()
        cp.n_ctx, cp.n_batch, cp.cb_eval = 512, 512, cb
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
            tk = lc.llama_sampler_sample(sampler, ctx, -1)
            if lc.llama_vocab_is_eog(vocab, tk):
                break
            lc.llama_decode(ctx, lc.llama_batch_get_one((lc.llama_token * 1)(tk), 1))
            _pos[0] += 1
        lc.llama_sampler_free(sampler)
        lc.llama_free(ctx)
        if pid % 5 == 4 or pid == len(PROMPTS) - 1:
            print(f"  {pid + 1}/{len(PROMPTS)} prompts, {len(hidden):,} rows "
                  f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    lc.llama_model_free(model)

    hk = {(p, t, l): v for p, t, l, v in hidden}
    ek = {(p, t, l): v for p, t, l, v in experts}
    keys = sorted(set(hk) & set(ek))
    print(f"\nhidden {len(hk):,}, experts {len(ek):,}, aligned {len(keys):,}", file=sys.stderr)
    H = np.stack([hk[k] for k in keys])
    E = np.stack([ek[k] for k in keys])
    K = np.array(keys, dtype=np.int32)
    reg = np.array([PROMPTS[k[0]][0] for k in keys])
    np.savez_compressed(OUT, hidden=H, experts=E, keys=K, register=reg)
    print(f"saved {OUT}  hidden={H.shape}, {os.path.getsize(OUT) / 1e6:.0f} MB",
          file=sys.stderr)


if __name__ == "__main__":
    main()
