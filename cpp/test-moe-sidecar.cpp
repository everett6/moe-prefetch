// MOE_POST / MOE_JOIN on the CUDA backend against a host echo thread.
//
// Graph: post = MOE_POST(x) ; chain = cached + repeat(post) ; out = MOE_JOIN(chain)
// For every run the output must equal, bit for bit, f(seq, e, i) + chain for a
// routed expert whose slot is the dummy slot and chain for every other one.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-moe-sidecar.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

static const int N_EMBD = 2048, N_USED = 8, DUMMY = 56, LAYER = 5;

static float row_value(uint32_t seq, int e, int i) {
    return (float) ((seq * 131u + (uint32_t) e * 17u + (uint32_t) i) % 1000u) * 0.001f - 0.5f;
}

struct echo_state {
    void * host = nullptr;
    std::atomic<bool> stop{false};
    std::atomic<int>  answer{1};          // 0: never answer a miss
    std::atomic<int>  delay_us{0};
    std::atomic<uint32_t> consumed{0};
    std::atomic<int>  bad_ids{0};
    std::vector<int32_t> slots;
    std::vector<int32_t> ids;
};

static void echo_loop(echo_state * st) {
    auto * ctl = (ggml_moe_sc_ctl *) st->host;
    while (!st->stop.load()) {
        const uint32_t head = __atomic_load_n(&ctl->head, __ATOMIC_ACQUIRE);
        uint32_t done = st->consumed.load();
        if (done == head) {
            continue;
        }
        const ggml_moe_sc_record rec = ctl->ring[done % GGML_MOE_SC_RING];
        for (int e = 0; e < N_USED; ++e) {
            if (ctl->ids[rec.layer][e] != st->ids[e]) {
                st->bad_ids++;
            }
        }
        if (rec.miss) {
            if (!st->answer.load()) {
                st->consumed = done + 1;
                continue;
            }
            if (st->delay_us.load()) {
                std::this_thread::sleep_for(std::chrono::microseconds(st->delay_us.load()));
            }
            float * rows = ggml_moe_sc_rows(st->host, N_EMBD, (int) rec.layer);
            for (int e = 0; e < N_USED; ++e) {
                if (st->slots[e] == DUMMY) {
                    for (int i = 0; i < N_EMBD; ++i) {
                        rows[(size_t) e * N_EMBD + i] = row_value(rec.seq, e, i);
                    }
                }
            }
            __atomic_store_n(&ctl->resp[rec.layer], rec.seq + 1, __ATOMIC_RELEASE);
        }
        st->consumed = done + 1;
    }
}

int main() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { printf("FAIL: no GPU backend\n"); return 1; }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto sc_alloc = (bool (*)(int, int, void **, void **, void **)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_sc_alloc");
    auto sc_free  = (void (*)(void *, void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_sc_free");
    auto sc_test  = (int (*)(int, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_sc_selftest");
    if (!sc_alloc || !sc_free || !sc_test) { printf("FAIL: sidecar entry points missing\n"); return 1; }

    void * host = nullptr, * host_dev = nullptr, * dev_state = nullptr;
    if (!sc_alloc(0, N_EMBD, &host, &host_dev, &dev_state)) { printf("FAIL: alloc\n"); return 1; }

    ggml_init_params ip = { 16u << 20, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, 1, 1);
    ggml_tensor * ids    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, 1);
    ggml_tensor * slots  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, 1);
    ggml_tensor * cached = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, N_USED, 1);
    ggml_tensor * post   = ggml_moe_post(ctx, x, ids, slots, LAYER, DUMMY, host_dev, dev_state);
    ggml_tensor * chain  = ggml_add(ctx, cached, ggml_repeat(ctx, post, cached));
    ggml_tensor * out    = ggml_moe_join(ctx, chain, slots, LAYER, DUMMY, host_dev, dev_state);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { printf("FAIL: tensor alloc\n"); return 1; }

    std::vector<float> hx(N_EMBD), hc((size_t) N_EMBD * N_USED), ho((size_t) N_EMBD * N_USED);
    for (size_t i = 0; i < hc.size(); ++i) hc[i] = (float) (i % 97) * 0.01f - 0.3f;
    ggml_backend_tensor_set(cached, hc.data(), 0, hc.size() * sizeof(float));

    echo_state st;
    st.host = host;
    st.ids = { 3, 17, 40, 41, 77, 90, 101, 127 };
    ggml_backend_tensor_set(ids, st.ids.data(), 0, N_USED * sizeof(int32_t));
    std::thread echo(echo_loop, &st);

    uint32_t seq = 0;
    auto run = [&](const char * name, std::vector<int32_t> slot_values, int runs, int expect_err) -> bool {
        st.slots = slot_values;
        ggml_backend_tensor_set(slots, slot_values.data(), 0, N_USED * sizeof(int32_t));
        int bad = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < runs; ++k, ++seq) {
            for (int i = 0; i < N_EMBD; ++i) hx[i] = (float) ((i + k) % 13) * 0.25f;
            ggml_backend_tensor_set(x, hx.data(), 0, N_EMBD * sizeof(float));
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { printf("FAIL %s: compute\n", name); return false; }
            ggml_backend_tensor_get(out, ho.data(), 0, ho.size() * sizeof(float));
            const bool answered = expect_err == 0;
            for (int e = 0; e < N_USED; ++e) {
                for (int i = 0; i < N_EMBD; ++i) {
                    const float c = hc[(size_t) e * N_EMBD + i] + hx[i];
                    const float h = (slot_values[e] == DUMMY && answered) ? row_value(seq, e, i) : 0.0f;
                    const float want = h + c;
                    if (memcmp(&want, &ho[(size_t) e * N_EMBD + i], sizeof(float)) != 0) ++bad;
                }
            }
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const uint32_t err = __atomic_load_n(&((ggml_moe_sc_ctl *) host)->err, __ATOMIC_ACQUIRE);
        const bool ok = bad == 0 && (int) (err != 0) == expect_err && st.bad_ids == 0;
        printf("%s %s: %d runs, %d bad values, err=%u, bad ids=%d, %.3f s\n", ok ? "ok  " : "FAIL", name, runs, bad, err, st.bad_ids.load(), secs);
        return ok;
    };

    bool ok = true;
    ok &= run("all_hit",   { 0, 1, 2, 3, 4, 5, 6, 7 }, 100, 0);
    ok &= run("mixed",     { DUMMY, 1, 2, DUMMY, 4, 5, 6, DUMMY }, 1000, 0);
    ok &= run("all_miss",  std::vector<int32_t>(N_USED, DUMMY), 100, 0);
    st.delay_us = 2000;
    ok &= run("delayed",   { DUMMY, 1, 2, 3, 4, 5, 6, 7 }, 20, 0);
    st.delay_us = 0;
    st.answer = 0;
    ok &= run("dead_host", { DUMMY, 1, 2, 3, 4, 5, 6, 7 }, 1, 1);

    st.stop = true;
    echo.join();
    const int selftest = sc_test(0, 10000);
    printf("%s selftest: %d failed of 10000\n", selftest == 0 ? "ok  " : "FAIL", selftest);
    ok &= selftest == 0;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    sc_free(host, dev_state);
    ggml_backend_free(backend);
    printf(ok ? "PASS: sidecar channel\n" : "FAIL: sidecar channel\n");
    return ok ? 0 : 1;
}
