// Probe P1 recorder: during greedy decode, write for every decode step and every
// MoE layer the residual stream entering the FFN (ffn_inp), the router's input
// (ffn_norm), and the 8 routed experts (ffn_moe_topk), plus the layer-0
// attention input (attn_norm-0, the normed token embedding). Offline,
// experiments/p10_lookahead_probe.py applies later layers' routers to these to
// measure how predictable future routing is. Diagnostic; runs on CPU or GPU.
//
// Output (MOE_DUMP_OUT): header "MRD1", int32 n_layer, n_embd, n_used, then per
// decode step: float16 attn_norm0[n_embd], and per layer float16 ffn_inp[n_embd],
// float16 ffn_norm[n_embd], int32 ids[n_used].
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct dump_state {
    bool decoding = false;
    int n_layer = 0, n_embd = 0, n_used = 0;
    std::vector<ggml_fp16_t> attn0;
    std::vector<std::vector<ggml_fp16_t>> inp, norm;
    std::vector<std::vector<int32_t>> ids;
    std::vector<int> seen;   // per layer: bit 1 inp, 2 norm, 4 ids
    bool attn0_seen = false;
    std::vector<float> tmp;
};

static bool parse(const char * name, const char * prefix, int & il) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || name[n] != '-') return false;
    il = atoi(name + n + 1);
    return true;
}

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = (dump_state *) ud;
    int il = -1;
    const char * name = t->name;
    const bool want = parse(name, "ffn_inp", il) || parse(name, "ffn_norm", il) ||
                      parse(name, "ffn_moe_topk", il) || strcmp(name, "attn_norm-0") == 0;
    if (ask) return want && st->decoding;
    if (!want || !st->decoding || t->ne[1] != 1) return true;
    if (strcmp(name, "attn_norm-0") == 0) {
        st->tmp.resize(st->n_embd);
        ggml_backend_tensor_get(t, st->tmp.data(), 0, st->n_embd * sizeof(float));
        ggml_fp32_to_fp16_row(st->tmp.data(), st->attn0.data(), st->n_embd);
        st->attn0_seen = true;
        return true;
    }
    if (il < 0 || il >= st->n_layer) return true;
    if (parse(name, "ffn_moe_topk", il)) {
        ggml_backend_tensor_get(t, st->ids[il].data(), 0, st->n_used * sizeof(int32_t));
        st->seen[il] |= 4;
        return true;
    }
    st->tmp.resize(st->n_embd);
    ggml_backend_tensor_get(t, st->tmp.data(), 0, st->n_embd * sizeof(float));
    const bool is_inp = strncmp(name, "ffn_inp", 7) == 0;
    ggml_fp32_to_fp16_row(st->tmp.data(), (is_inp ? st->inp : st->norm)[il].data(), st->n_embd);
    st->seen[il] |= is_inp ? 1 : 2;
    return true;
}

static void usage(int, char **) {}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON, usage)) return 2;
    const char * out_path = getenv("MOE_DUMP_OUT");
    if (!out_path) { fprintf(stderr, "set MOE_DUMP_OUT\n"); return 2; }
    dump_state st;
    params.cb_eval = cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;
    common_init();
    llama_backend_init();
    auto init = common_init_from_params(params);
    auto * ctx = init->context();
    auto * model = init->model();
    if (!ctx || !model) return 2;
    auto * vocab = llama_model_get_vocab(model);
    st.n_layer = llama_model_n_layer(model);
    st.n_embd  = llama_model_n_embd(model);
    st.n_used  = 8;
    st.attn0.resize(st.n_embd);
    st.inp.assign(st.n_layer, std::vector<ggml_fp16_t>(st.n_embd));
    st.norm.assign(st.n_layer, std::vector<ggml_fp16_t>(st.n_embd));
    st.ids.assign(st.n_layer, std::vector<int32_t>(st.n_used));
    st.seen.assign(st.n_layer, 0);

    FILE * out = fopen(out_path, "wb");
    if (!out) { perror(out_path); return 2; }
    fwrite("MRD1", 1, 4, out);
    fwrite(&st.n_layer, 4, 1, out); fwrite(&st.n_embd, 4, 1, out); fwrite(&st.n_used, 4, 1, out);

    auto tokens = common_tokenize(vocab, params.prompt, true);
    if (tokens.empty() || tokens.size() + params.n_predict > llama_n_ctx(ctx)) return 2;
    llama_batch batch = llama_batch_init(llama_n_batch(ctx), 0, 1);
    for (size_t off = 0; off < tokens.size();) {
        common_batch_clear(batch);
        const size_t end = std::min(tokens.size(), off + llama_n_batch(ctx));
        for (; off < end; ++off) common_batch_add(batch, tokens[off], off, {0}, off + 1 == tokens.size());
        if (llama_decode(ctx, batch)) return 2;
    }
    const uint32_t nv = llama_vocab_n_tokens(vocab);
    int written = 0;
    for (int step = 0; step < params.n_predict; ++step) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        llama_token next = 0;
        for (uint32_t i = 1; i < nv; ++i) if (logits[i] > logits[next]) next = (llama_token) i;
        st.decoding = true;
        std::fill(st.seen.begin(), st.seen.end(), 0);
        st.attn0_seen = false;
        common_batch_clear(batch);
        common_batch_add(batch, next, tokens.size() + step, {0}, true);
        if (llama_decode(ctx, batch)) return 2;
        st.decoding = false;
        bool complete = st.attn0_seen;
        for (int l = 0; l < st.n_layer; ++l) complete &= st.seen[l] == 7;
        if (!complete) { fprintf(stderr, "step %d: incomplete capture\n", step); return 3; }
        fwrite(st.attn0.data(), sizeof(ggml_fp16_t), st.n_embd, out);
        for (int l = 0; l < st.n_layer; ++l) {
            fwrite(st.inp[l].data(),  sizeof(ggml_fp16_t), st.n_embd, out);
            fwrite(st.norm[l].data(), sizeof(ggml_fp16_t), st.n_embd, out);
            fwrite(st.ids[l].data(),  sizeof(int32_t), st.n_used, out);
        }
        ++written;
    }
    fclose(out);
    llama_batch_free(batch);
    printf("{\"steps\":%d,\"n_layer\":%d,\"n_embd\":%d}\n", written, st.n_layer, st.n_embd);
    return 0;
}
