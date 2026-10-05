// Diagnostic runner: compare all vocabulary logits under an identical prefix.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static FILE * env_file(const char * name, const char * mode) {
    const char * path = getenv(name);
    if (!path) return nullptr;
    FILE * f = fopen(path, mode);
    if (!f) { perror(path); exit(2); }
    return f;
}
static void usage(int, char **) {}
int main(int argc, char ** argv) {
    common_init();
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_BATCHED, usage)) return 2;
    params.warmup = false;
    params.n_parallel = 1;
    llama_backend_init();
    auto init = common_init_from_params(params);
    auto * ctx = init->context();
    auto * model = init->model();
    if (!ctx || !model) return 2;
    auto * vocab = llama_model_get_vocab(model);
    const uint32_t nv = llama_vocab_n_tokens(vocab);
    auto tokens = common_tokenize(vocab, params.prompt, true);
    if (tokens.empty() || tokens.size() + params.n_predict > llama_n_ctx(ctx)) return 2;
    FILE * out = env_file("MOE_LOGITS_OUT", "wb");
    FILE * reference = env_file("MOE_LOGITS_REFERENCE", "rb");
    FILE * token_out = env_file("MOE_TOKENS_OUT", "wb");
    FILE * force = env_file("MOE_FORCE_TOKENS", "rb");
    if (!out && !reference) { fprintf(stderr, "Set MOE_LOGITS_OUT or MOE_LOGITS_REFERENCE\n"); return 2; }
    if (out && fwrite(&nv, sizeof(nv), 1, out) != 1) return 2;
    if (reference) {
        uint32_t ref_nv = 0;
        if (fread(&ref_nv, sizeof(ref_nv), 1, reference) != 1 || ref_nv != nv) return 2;
    }
    llama_batch batch = llama_batch_init(llama_n_batch(ctx), 0, 1);
    for (size_t offset = 0; offset < tokens.size();) {
        common_batch_clear(batch);
        size_t end = std::min(tokens.size(), offset + llama_n_batch(ctx));
        for (; offset < end; ++offset) common_batch_add(batch, tokens[offset], offset, {0}, offset + 1 == tokens.size());
        if (llama_decode(ctx, batch)) return 2;
    }
    std::vector<float> ref(nv);
    uint64_t unequal_logits = 0;
    int first_step = -1;
    double max_abs = 0;
    for (int step = 0; step < params.n_predict; ++step) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) return 2;
        llama_token greedy = std::max_element(logits, logits + nv) - logits;
        if (out && fwrite(logits, sizeof(float), nv, out) != nv) return 2;
        if (reference) {
            if (fread(ref.data(), sizeof(float), nv, reference) != nv) return 2;
            for (uint32_t i = 0; i < nv; ++i) {
                if (memcmp(logits + i, ref.data() + i, sizeof(float))) {
                    ++unequal_logits;
                    if (first_step < 0) first_step = step;
                    max_abs = std::max(max_abs, std::abs(double(logits[i]) - double(ref[i])));
                }
            }
        }
        if (token_out && fwrite(&greedy, sizeof(greedy), 1, token_out) != 1) return 2;
        llama_token next = greedy;
        if (force && fread(&next, sizeof(next), 1, force) != 1) return 2;
        if (next < 0 || next >= (llama_token) nv) return 2;
        if (step + 1 == params.n_predict) break;
        common_batch_clear(batch);
        common_batch_add(batch, next, tokens.size() + step, {0}, true);
        if (llama_decode(ctx, batch)) return 2;
    }
    if (reference && fgetc(reference) != EOF) return 2;
    for (FILE * f : {out, reference, token_out, force}) if (f && fclose(f)) return 2;
    llama_batch_free(batch);
    printf("{\"steps\":%d,\"vocab\":%u,\"unequal_logits\":%llu,\"first_difference_step\":%d,\"max_abs_difference\":%.9g}\n",
        params.n_predict, nv, (unsigned long long) unequal_logits, first_step, max_abs);
    return unequal_logits ? 1 : 0;
}
