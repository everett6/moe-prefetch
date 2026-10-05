// The persistent sidecar team (ggml_cpu_team_run) must produce exactly what
// ggml_graph_compute produces for the same MOE_ROWS node: byte-identical rows
// and the same number of routing observations, over many jobs inside ONE team
// region, with all-hit, all-miss and mixed tables. CPU only.
#include "ggml.h"
#include "ggml-cpu.h"

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

struct job {
    std::vector<float>   x;
    std::vector<int32_t> ids, table;
    std::vector<uint8_t> ref;
    int                  obs = 0;
};

struct run_state {
    std::vector<job> * jobs;
    ggml_tensor *input, *ids, *table, *node;
    size_t next = 0, checked = 0, bad = 0, done_calls = 0;
    int obs_before = 0;
};

static void load(const job & j, ggml_tensor * input, ggml_tensor * ids, ggml_tensor * table) {
    std::memcpy(input->data, j.x.data(), j.x.size() * sizeof(float));
    std::memcpy(ids->data, j.ids.data(), j.ids.size() * sizeof(int32_t));
    std::memcpy(table->data, j.table.data(), j.table.size() * sizeof(int32_t));
}

static ggml_tensor * team_next(void * ud) {
    auto * st = static_cast<run_state *>(ud);
    if (st->next == st->jobs->size()) return nullptr;
    const job & j = (*st->jobs)[st->next++];
    load(j, st->input, st->ids, st->table);
    std::memset(st->node->data, 0xA5, ggml_nbytes(st->node));   // stale output must not survive
    st->obs_before = observations;
    return st->node;
}

static void team_done(void * ud, ggml_tensor * t) {
    auto * st = static_cast<run_state *>(ud);
    const job & j = (*st->jobs)[st->next - 1];
    st->done_calls++;
    if (t != st->node || std::memcmp(t->data, j.ref.data(), j.ref.size()) != 0 ||
        observations - st->obs_before != j.obs) {
        st->bad++;
    }
    st->checked++;
}

int main() {
    ggml_cpu_init();
    ggml_set_moe_obs_callback(observe, nullptr);
    const int n_jobs = 500;
    size_t total = 0;
    for (int threads : {1, 2, 8}) {
        for (auto weight_type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) {
            auto * ctx = ggml_init({64u << 20, nullptr, false});
            auto * up = ggml_new_tensor_3d(ctx, weight_type, 256, 512, 16);
            const auto gate_type = weight_type == GGML_TYPE_Q8_0 ? GGML_TYPE_Q5_K : weight_type;
            auto * gate = ggml_new_tensor_3d(ctx, gate_type, 256, 512, 16);
            auto * down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q6_K, 512, 256, 16);
            auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 1);
            auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 1);
            auto * table = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 16);
            ggml_set_name(gate, "blk.0.ffn_gate_exps.weight");
            std::mt19937 rng(7 + threads * 31 + (int) weight_type);
            fill(up, rng); fill(gate, rng); fill(down, rng);
            auto * node = ggml_moe_rows(ctx, up, input, ids, table, gate, down, 8);
            auto * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, node);
            auto plan = ggml_graph_plan(graph, threads, nullptr);
            std::vector<uint8_t> work(plan.work_size);
            plan.work_data = work.data();

            // references first, with the ordinary graph path
            std::vector<job> jobs(n_jobs);
            std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
            for (int k = 0; k < n_jobs; ++k) {
                job & j = jobs[k];
                j.x.resize(256);
                for (float & v : j.x) v = xd(rng);
                j.ids.resize(8);
                for (auto & id : j.ids) id = (int32_t) (rng() % 16);
                const int mode = k % 3;   // all hit, all miss, mixed
                j.table.resize(16);
                for (int e = 0; e < 16; ++e) j.table[e] = mode == 0 ? e % 8 : mode == 1 ? 8 : (rng() % 2 ? 8 : e % 8);
                load(j, input, ids, table);
                observations = 0;
                if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) return 2;
                j.obs = observations;
                j.ref.assign((const uint8_t *) node->data, (const uint8_t *) node->data + ggml_nbytes(node));
            }

            auto * team = ggml_cpu_team_new(threads, plan.work_size, -1);
            if (!team) {
                std::fprintf(stderr, "FAIL: ggml_cpu_team_new returned NULL (built without OpenMP?)\n");
                return 1;
            }
            run_state st{&jobs, input, ids, table, node};
            // two regions back to back: a region must end cleanly and the next start cleanly
            const size_t half = n_jobs / 2;
            std::vector<job> first(jobs.begin(), jobs.begin() + half), second(jobs.begin() + half, jobs.end());
            st.jobs = &first;
            ggml_cpu_team_run(team, team_next, team_done, &st);
            const size_t checked_first = st.checked, bad_first = st.bad;
            st.jobs = &second; st.next = 0;
            ggml_cpu_team_run(team, team_next, team_done, &st);
            ggml_cpu_team_free(team);
            if (st.bad || st.checked != (size_t) n_jobs || checked_first != half || bad_first) {
                std::fprintf(stderr, "FAIL threads=%d type=%s: %zu of %zu jobs differ (done calls %zu)\n",
                             threads, ggml_type_name(weight_type), st.bad, st.checked, st.done_calls);
                return 1;
            }
            total += st.checked;
            ggml_free(ctx);
        }
    }
    ggml_set_moe_obs_callback(nullptr, nullptr);
    std::printf("PASS: %zu team jobs byte-identical to ggml_graph_compute, observation counts equal\n", total);
    return 0;
}
