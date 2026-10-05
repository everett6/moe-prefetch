#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static void fill(ggml_tensor * tensor, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-0.15f, 0.15f);
    std::vector<float> values(ggml_nelements(tensor));
    for (float & value : values) value = dist(rng);
    ggml_quantize_chunk(tensor->type, values.data(), tensor->data, 0,
                        ggml_nrows(tensor), tensor->ne[0], nullptr);
}

static int observations;
static void observe(const char *, const ggml_tensor *, void *) { ++observations; }

int main() {
    ggml_cpu_init();
    ggml_set_moe_obs_callback(observe, nullptr);
    int cases = 0;
    for (int threads : {1, 2, 8}) {
        auto pool_params = ggml_threadpool_params_default(threads);
        auto * pool = ggml_threadpool_new(&pool_params);
        for (int tokens : {1, 2, 8}) {
            for (auto weight_type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) {
                auto * ctx = ggml_init({64u << 20, nullptr, false});
                auto * up = ggml_new_tensor_3d(ctx, weight_type, 256, 512, 16);
                const auto gate_type = weight_type == GGML_TYPE_Q8_0 ? GGML_TYPE_Q5_K : weight_type;
                auto * gate = ggml_new_tensor_3d(ctx, gate_type, 256, 512, 16);
                auto * down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q6_K, 512, 256, 16);
                auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, tokens);
                auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, tokens);
                auto * table = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 16);
                ggml_set_name(gate, "blk.0.ffn_gate_exps.weight");
                std::mt19937 rng(2026 + tokens);
                fill(up, rng); fill(gate, rng); fill(down, rng);
                auto * ref_up = ggml_mul_mat_id(ctx, up, input, ids);
                auto * ref_gate = ggml_mul_mat_id(ctx, gate, input, ids);
                auto * ref_act = ggml_swiglu_split(ctx, ref_gate, ref_up);
                auto * reference = ggml_mul_mat_id(ctx, down, ref_act, ids);
                for (auto * t : {ref_up, ref_gate, reference}) {
                    t->src[3] = table;
                    t->op_params[0] = 8;
                }
                auto * fused = ggml_moe_rows(ctx, up, input, ids, table, gate, down, 8);
                auto * ref_graph = ggml_new_graph(ctx);
                auto * fused_graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(ref_graph, reference);
                ggml_build_forward_expand(fused_graph, fused);
                auto ref_plan = ggml_graph_plan(ref_graph, threads, pool);
                auto fused_plan = ggml_graph_plan(fused_graph, threads, pool);
                std::vector<uint8_t> ref_work(ref_plan.work_size), fused_work(fused_plan.work_size);
                ref_plan.work_data = ref_work.data(); fused_plan.work_data = fused_work.data();
                for (int mode = 0; mode < 5; ++mode) {
                    // All hits, all misses, mixed, repeated ids, and no cache table.
                    auto * mapping = static_cast<int32_t *>(table->data);
                    for (int e = 0; e < 16; ++e) mapping[e] = mode == 0 ? e % 8 : mode == 1 ? 8 : e % 2 ? 8 : e % 8;
                    for (auto * t : {ref_up, ref_gate, reference, fused}) t->src[3] = mode == 4 ? nullptr : table;
                    for (int replay = 0; replay < 3; ++replay) {
                        fill(input, rng);
                        auto * selected = static_cast<int32_t *>(ids->data);
                        for (int i = 0; i < 8 * tokens; ++i) selected[i] = mode == 3 ? i % 2 : rng() % 16;
                        observations = 0;
                        if (ggml_graph_compute(ref_graph, &ref_plan) != GGML_STATUS_SUCCESS ||
                            ggml_graph_compute(fused_graph, &fused_plan) != GGML_STATUS_SUCCESS) return 2;
                        if (observations != 2 || std::memcmp(reference->data, fused->data, ggml_nbytes(reference))) {
                            std::fprintf(stderr, "FAIL threads=%d tokens=%d type=%s mode=%d replay=%d observations=%d\n",
                                         threads, tokens, ggml_type_name(weight_type), mode, replay, observations);
                            return 1;
                        }
                        ++cases;
                    }
                }
                ggml_free(ctx);
            }
        }
        ggml_threadpool_free(pool);
    }
    ggml_set_moe_obs_callback(nullptr, nullptr);
    std::printf("PASS: %d bit-identical expert-row comparisons, including graph replay and routing observation\n", cases);
}
