// MOE_POST / MOE_JOIN on the CUDA backend against a host echo thread.
//
// Graph: post = MOE_POST(x) ; chain = cached + repeat(post) ; out = MOE_JOIN(chain)
// For every run the output must equal, bit for bit, f(seq, e, i) + chain for a
// routed expert whose slot is the dummy slot and chain for every other one.
//
// Streaming (0010): MOE_AHEAD's posted candidates must equal a CPU reference
// (integer-valued inputs, so every score is exact in any summation order), and
// the copy queue must land data and the table word written after it.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-moe-sidecar.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <random>
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
            if (ctl->ids[rec.layer][e] != st->ids[e] || ctl->slots[rec.layer][e] != st->slots[e]) {
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

    // ---- MOE_AHEAD: two targets, top 8, filtered by each target's table ----
    {
        const int NE = 128, K = 8, D0 = 56, D1 = 40, T0 = LAYER + 1, T1 = LAYER + 2;
        ggml_context * actx = ggml_init({ 16u << 20, nullptr, true });
        ggml_tensor * ax = ggml_new_tensor_1d(actx, GGML_TYPE_F32, N_EMBD);
        ggml_tensor * w[2], * r[2], * tbl[2];
        for (int t = 0; t < 2; ++t) {
            w[t]   = ggml_new_tensor_2d(actx, GGML_TYPE_F32, N_EMBD, NE);
            r[t]   = ggml_new_tensor_1d(actx, GGML_TYPE_F32, N_EMBD);
            tbl[t] = ggml_new_tensor_1d(actx, GGML_TYPE_I32, NE);
        }
        const int32_t targets[2] = { T0, T1 }, dummies[2] = { D0, D1 };
        ggml_tensor * ahead = ggml_moe_ahead(actx, ax, w, r, tbl, 2, LAYER, targets, dummies, K, host_dev, dev_state);
        ggml_cgraph * ag = ggml_new_graph(actx);
        ggml_build_forward_expand(ag, ahead);
        ggml_backend_buffer_t abuf = ggml_backend_alloc_ctx_tensors(actx, backend);
        std::mt19937 rng(11);
        auto * ctl = (ggml_moe_sc_ctl *) host;
        int bad = 0;
        const int runs = 200;
        for (int k = 0; k < runs; ++k) {
            std::vector<float> hxv(N_EMBD), hw[2], hr[2];
            std::vector<int32_t> ht[2];
            for (auto & v : hxv) v = (float) ((int) (rng() % 5) - 2);
            for (int t = 0; t < 2; ++t) {
                hw[t].resize((size_t) N_EMBD * NE);
                for (auto & v : hw[t]) v = (float) ((int) (rng() % 7) - 3);
                hr[t].resize(N_EMBD);
                for (auto & v : hr[t]) v = (rng() % 3 == 0) ? 0.5f : (rng() % 2 ? 1.0f : 2.0f);
                ht[t].resize(NE);
                for (int e = 0; e < NE; ++e) ht[t][e] = rng() % 2 ? dummies[t] : e % 50;
                ggml_backend_tensor_set(w[t], hw[t].data(), 0, hw[t].size() * sizeof(float));
                ggml_backend_tensor_set(r[t], hr[t].data(), 0, hr[t].size() * sizeof(float));
                ggml_backend_tensor_set(tbl[t], ht[t].data(), 0, NE * sizeof(int32_t));
            }
            ggml_backend_tensor_set(ax, hxv.data(), 0, N_EMBD * sizeof(float));
            const uint32_t before = __atomic_load_n(&ctl->ahead_head, __ATOMIC_ACQUIRE);
            if (ggml_backend_graph_compute(backend, ag) != GGML_STATUS_SUCCESS) { printf("FAIL ahead: compute\n"); return 1; }
            const uint32_t after = __atomic_load_n(&ctl->ahead_head, __ATOMIC_ACQUIRE);
            const ggml_moe_sc_ahead rec = ctl->ahead[before % GGML_MOE_SC_RING];
            bool ok_run = after == before + 1 && rec.seq == before && rec.layer == (uint32_t) LAYER &&
                          rec.target[0] == T0 && rec.target[1] == T1 && rec.t_ns != 0;
            for (int t = 0; t < 2 && ok_run; ++t) {
                std::vector<std::pair<double, int>> sc(NE);
                for (int e = 0; e < NE; ++e) {
                    double s = 0;
                    for (int i = 0; i < N_EMBD; ++i) s += (double) hw[t][(size_t) e * N_EMBD + i] * hxv[i] * hr[t][i];
                    sc[e] = { -s, e };     // best first, lower id on a tie
                }
                std::sort(sc.begin(), sc.end());
                std::vector<int32_t> want;
                for (int j = 0; j < K; ++j) {
                    if (ht[t][sc[j].second] == dummies[t]) want.push_back(sc[j].second);
                }
                ok_run = rec.n[t] == want.size() && std::equal(want.begin(), want.end(), rec.ids[t]);
            }
            bad += !ok_run;
        }
        printf("%s ahead: %d runs, %d wrong\n", bad ? "FAIL" : "ok  ", runs, bad);
        ok &= bad == 0;
        ggml_backend_buffer_free(abuf);
        ggml_free(actx);
    }

    // ---- copy queue: chunks then the table word, landed when the ticket says so ----
    {
        auto mnew  = (void * (*)(int, int, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_mover_new");
        auto mfree = (void (*)(void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_mover_free");
        auto mword = (const int32_t * (*)(void *, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_mover_word");
        auto mpush = (int64_t (*)(void *, void **, const void **, const size_t *, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_mover_push");
        auto mdone = (int64_t (*)(void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_mover_done");
        auto msync = (int64_t (*)(void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_mover_sync");
        auto pinned = (bool (*)(const void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_is_pinned");
        if (!mnew || !mfree || !mword || !mpush || !mdone || !msync || !pinned) { printf("FAIL: copy queue entry points missing\n"); return 1; }
        const size_t EXP = 2831155, CH = 512 * 1024;
        const int SLOTS = 4, NE = 128;
        ggml_context * mctx = ggml_init({ 1u << 20, nullptr, true });
        ggml_tensor * dst = ggml_new_tensor_1d(mctx, GGML_TYPE_I8, (int64_t) (EXP * SLOTS));
        ggml_tensor * tab = ggml_new_tensor_1d(mctx, GGML_TYPE_I32, NE);
        ggml_backend_buffer_t mbuf = ggml_backend_alloc_ctx_tensors(mctx, backend);
        ggml_backend_buffer_t hbuf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(dev), EXP * 8);
        uint8_t * src = (uint8_t *) ggml_backend_buffer_get_base(hbuf);
        std::vector<int32_t> dummy_tab(NE, SLOTS);
        ggml_backend_tensor_set(tab, dummy_tab.data(), 0, NE * sizeof(int32_t));
        void * m = mnew(0, 16, SLOTS + 1);
        std::vector<uint8_t> plain(16);
        bool mok = m && pinned(src) && !pinned(plain.data());
        int wrong = 0;
        for (int k = 0; k < 40 && mok; ++k) {
            const int slot = k % SLOTS, expert = (k * 37) % 8, e_id = (k * 13) % NE;
            for (size_t i = 0; i < EXP; ++i) src[(size_t) expert * EXP + i] = (uint8_t) (i * 7 + k);
            int64_t last = 0;
            for (size_t off = 0; off < EXP; off += CH) {
                const size_t len = std::min(CH, EXP - off);
                void * d[2] = { (char *) dst->data + (size_t) slot * EXP + off, nullptr };
                const void * s2[2] = { src + (size_t) expert * EXP + off, nullptr };
                size_t n[2] = { len, 0 };
                int cnt = 1;
                if (off + len == EXP) {
                    d[1] = (char *) tab->data + (size_t) e_id * sizeof(int32_t);
                    s2[1] = mword(m, slot);
                    n[1] = sizeof(int32_t);
                    cnt = 2;
                }
                last = mpush(m, d, s2, n, cnt);
                if (last <= 0) { mok = false; break; }
            }
            while (mok && mdone(m) < last) {}
            std::vector<uint8_t> got(EXP);
            ggml_backend_tensor_get(dst, got.data(), (size_t) slot * EXP, EXP);
            int32_t word = -1;
            ggml_backend_tensor_get(tab, &word, (size_t) e_id * sizeof(int32_t), sizeof(int32_t));
            wrong += memcmp(got.data(), src + (size_t) expert * EXP, EXP) != 0 || word != slot;
            ggml_backend_tensor_set(tab, &SLOTS, (size_t) e_id * sizeof(int32_t), sizeof(int32_t));
        }
        mok = mok && msync(m) >= 0 && wrong == 0;
        printf("%s copy queue: 40 experts, %d wrong\n", mok ? "ok  " : "FAIL", wrong);
        ok &= mok;
        mfree(m);
        ggml_backend_buffer_free(hbuf);
        ggml_backend_buffer_free(mbuf);
        ggml_free(mctx);
    }
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
