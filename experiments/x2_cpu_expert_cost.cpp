// x2: what does ONE missed expert cost the CPU, measured on its own?
//
// The cost model has always carried a per-miss term (47 us, later refitted) but
// it was only ever inferred from end-to-end throughput, where it is tangled up
// with the scheduler's round trip and with uploads. This computes the thing
// directly: the three matrix-vector products of one routed expert, with the
// same ggml kernels the engine's CPU chain uses (quantise the activation to the
// weight type's dot type, then one vec_dot per row), on weights read from the
// real GGUF, visiting experts in an order that keeps them out of the CPU cache.
//
// It answers three things the design needs:
//   - microseconds per missed expert, against thread count;
//   - whether it is memory-bandwidth-bound (so more threads stop helping);
//   - how a layer with m misses scales, i.e. how much the GPU has to hide.
//
// build:
//   E=/home/everett/llama.cpp-build
//   g++ -O3 -march=native -std=c++17 -fopenmp x2_cpu_expert_cost.cpp -I$E/ggml/include \
//       -L$E/build/bin -lggml-cpu -lggml-base -Wl,-rpath,$E/build/bin -o x2_cpu_expert_cost
// run (keep the OpenMP team spinning, as the engine's does between layers):
//   OMP_WAIT_POLICY=active ./x2_cpu_expert_cost <model.gguf> [json_out]

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

static inline double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

struct mat {
    uint8_t *                         data = nullptr;
    ggml_type                         type = GGML_TYPE_F32;
    int64_t                           n_in = 0, n_out = 0, n_expert = 0;
    size_t                            row_bytes = 0, expert_bytes = 0;
    const struct ggml_type_traits_cpu * tr = nullptr;
    ggml_type                         dot_type = GGML_TYPE_F32;
    ggml_from_float_t                 quant = nullptr;
    size_t                            q_bytes = 0;   // bytes of a quantised input vector
};

static bool load(mat & m, gguf_context * g, ggml_context * meta, int fd, const std::string & name) {
    const int64_t id = gguf_find_tensor(g, name.c_str());
    if (id < 0) {
        fprintf(stderr, "tensor %s not found\n", name.c_str());
        return false;
    }
    const ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
    m.type       = gguf_get_tensor_type(g, id);
    m.n_in       = t->ne[0];
    m.n_out      = t->ne[1];
    m.n_expert   = t->ne[2];
    m.row_bytes  = ggml_row_size(m.type, m.n_in);
    m.expert_bytes = m.row_bytes * m.n_out;
    m.tr         = ggml_get_type_traits_cpu(m.type);
    m.dot_type   = m.tr->vec_dot_type;
    m.quant      = ggml_get_type_traits_cpu(m.dot_type)->from_float;
    m.q_bytes    = ggml_row_size(m.dot_type, m.n_in);
    const size_t bytes = gguf_get_tensor_size(g, id);
    if (bytes != m.expert_bytes * (size_t) m.n_expert) {
        fprintf(stderr, "%s: size %zu is not n_expert * rows * row_bytes (%zu)\n", name.c_str(), bytes,
                m.expert_bytes * (size_t) m.n_expert);
        return false;
    }
    if (posix_memalign((void **) &m.data, 4096, bytes)) {
        return false;
    }
    // the engine's host copy is driver-pinned 4 KiB pages, not huge pages; match it
    madvise(m.data, bytes, MADV_NOHUGEPAGE);
    const off_t off = (off_t) (gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id));
    size_t done = 0;
    while (done < bytes) {
        const ssize_t r = pread(fd, m.data + done, std::min<size_t>(bytes - done, 64u << 20), off + (off_t) done);
        if (r <= 0) {
            fprintf(stderr, "read failed on %s\n", name.c_str());
            return false;
        }
        done += (size_t) r;
    }
    return true;
}

struct layer {
    mat up, gate, down;
};

struct job {
    const layer * L;
    int           e;
};

// Compute `n` experts' outputs for one activation, the way the engine's CPU
// chain does, with the calling OpenMP team. Returns nothing useful; `out` is
// [n][n_embd].
static void compute(const job * jobs, int n, const float * x, std::vector<float> & out, std::vector<uint8_t> & qx,
                    std::vector<uint8_t> & qact, std::vector<float> & up, std::vector<float> & gate,
                    std::vector<float> & act) {
    const layer & L0   = *jobs[0].L;
    const int64_t n_ff = L0.up.n_out, n_embd = L0.up.n_in;
    #pragma omp parallel
    {
        #pragma omp single
        {
            // one quantisation of x serves up and gate when they share a dot type
            L0.up.quant(x, qx.data(), n_embd);
        }
        #pragma omp for schedule(static) collapse(2)
        for (int j = 0; j < n; j++) {
            for (int64_t r = 0; r < 2 * n_ff; r++) {
                const layer & L = *jobs[j].L;
                const mat &   m = r < n_ff ? L.up : L.gate;
                const int64_t rr = r < n_ff ? r : r - n_ff;
                float *       dst = (r < n_ff ? up.data() : gate.data()) + j * n_ff + rr;
                m.tr->vec_dot((int) n_embd, dst, 0, m.data + (size_t) jobs[j].e * m.expert_bytes + rr * m.row_bytes, 0,
                              qx.data(), 0, 1);
            }
        }
        #pragma omp for schedule(static)
        for (int j = 0; j < n; j++) {
            for (int64_t r = 0; r < n_ff; r++) {
                const float g = gate[j * n_ff + r];
                act[j * n_ff + r] = g / (1.0f + expf(-g)) * up[j * n_ff + r];
            }
            jobs[j].L->down.quant(act.data() + j * n_ff, qact.data() + j * jobs[j].L->down.q_bytes, n_ff);
        }
        #pragma omp for schedule(static) collapse(2)
        for (int j = 0; j < n; j++) {
            for (int64_t r = 0; r < n_embd; r++) {
                const mat & m = jobs[j].L->down;
                m.tr->vec_dot((int) n_ff, out.data() + j * n_embd + r, 0,
                              m.data + (size_t) jobs[j].e * m.expert_bytes + r * m.row_bytes, 0,
                              qact.data() + j * m.q_bytes, 0, 1);
            }
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [json_out]\n", argv[0]);
        return 1;
    }
    const char * path     = argv[1];
    const char * json_out = argc > 2 ? argv[2] : nullptr;
    ggml_cpu_init();

    ggml_context *   meta = nullptr;
    gguf_init_params ip;
    ip.no_alloc = true;
    ip.ctx      = &meta;
    gguf_context * g = gguf_init_from_file(path, ip);
    if (!g) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    const int fd = open(path, O_RDONLY);

    // Layers spread over the stack, including both down-projection quant types.
    std::vector<int> want = {3, 14, 25, 36};
    if (getenv("X2_LAYERS")) {
        want.clear();
        for (char * tok = strtok(getenv("X2_LAYERS"), ","); tok; tok = strtok(nullptr, ",")) {
            want.push_back(atoi(tok));
        }
    }
    std::vector<layer> layers(want.size());
    size_t             total = 0;
    for (size_t i = 0; i < want.size(); i++) {
        const std::string p = "blk." + std::to_string(want[i]) + ".ffn_";
        if (!load(layers[i].up, g, meta, fd, p + "up_exps.weight") ||
            !load(layers[i].gate, g, meta, fd, p + "gate_exps.weight") ||
            !load(layers[i].down, g, meta, fd, p + "down_exps.weight")) {
            return 1;
        }
        const layer & L = layers[i];
        const size_t  eb = L.up.expert_bytes + L.gate.expert_bytes + L.down.expert_bytes;
        total += eb * (size_t) L.up.n_expert;
        printf("layer %2d: up %s  gate %s  down %s   %lld -> %lld -> %lld   %.2f MiB per expert, %lld experts\n", want[i],
               ggml_type_name(L.up.type), ggml_type_name(L.gate.type), ggml_type_name(L.down.type),
               (long long) L.up.n_in, (long long) L.up.n_out, (long long) L.down.n_out, eb / 1048576.0,
               (long long) L.up.n_expert);
    }
    printf("loaded %.2f GiB of expert weights (far more than the CPU cache, so every visit is cold)\n\n",
           total / 1073741824.0);

    const int64_t n_embd = layers[0].up.n_in, n_ff = layers[0].up.n_out;
    const int     max_m  = 8;
    std::vector<float>   x(n_embd), out(max_m * n_embd), up(max_m * n_ff), gate(max_m * n_ff), act(max_m * n_ff);
    std::vector<uint8_t> qx(layers[0].up.q_bytes + 64), qact(max_m * (layers[0].down.q_bytes + 64));
    std::mt19937         rng(12345);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & v : x) {
        v = nd(rng);
    }

    // every (layer, expert) pair, shuffled once: a fixed, cache-hostile tour
    std::vector<job> tour;
    for (auto & L : layers) {
        for (int e = 0; e < (int) L.up.n_expert; e++) {
            tour.push_back({&L, e});
        }
    }
    std::shuffle(tour.begin(), tour.end(), rng);

    struct row {
        int    threads, m;
        double p50, p90, mean, mib;
    };
    std::vector<row> rows;
    auto run = [&](int threads, int m) {
        omp_set_num_threads(threads);
        std::vector<double> t;
        double              bytes = 0;
        for (int pass = 0; pass < 2; pass++) {
            for (size_t i = 0; i + m <= tour.size(); i += m) {
                // one "layer" = m experts of the SAME layer in the engine; here any m keep the
                // arithmetic identical and the memory traffic the same
                job js[max_m];
                for (int j = 0; j < m; j++) {
                    js[j] = tour[i + j];
                    js[j].L = tour[i].L;   // same shapes within a call
                }
                const double t0 = now_us();
                compute(js, m, x.data(), out, qx, qact, up, gate, act);
                const double t1 = now_us();
                if (pass == 1) {
                    t.push_back(t1 - t0);
                    const layer & L = *js[0].L;
                    bytes += (double) m * (L.up.expert_bytes + L.gate.expert_bytes + L.down.expert_bytes);
                }
            }
        }
        std::sort(t.begin(), t.end());
        double s = 0;
        for (double v : t) {
            s += v;
        }
        row r{threads, m, t[t.size() / 2], t[(size_t) (t.size() * 0.9)], s / t.size(), bytes / t.size() / 1048576.0};
        rows.push_back(r);
        printf("  threads %2d   %d expert%s per call   %7.1f us p50   %7.1f p90   %7.1f us per expert   %5.1f GB/s\n",
               threads, m, m > 1 ? "s" : " ", r.p50, r.p90, r.p50 / m, r.mib * 1048576.0 / r.p50 / 1e3);
    };

    printf("[1] one missed expert, by thread count\n");
    for (int th : {1, 2, 4, 6, 8, 12, 16, 24, 32}) {
        if (th <= omp_get_num_procs()) {
            run(th, 1);
        }
    }
    printf("\n[2] a layer with m misses, 16 threads (what the GPU would have to hide)\n");
    for (int m : {1, 2, 3, 4, 8}) {
        run(16, m);
    }
    printf("\n[3] the same with 8 threads\n");
    for (int m : {1, 2, 4, 8}) {
        run(8, m);
    }

    // [4] The same expert computed twice in a row: the second time its weights
    // are in the CPU cache. This is what a miss would cost if the host had been
    // told a layer ahead which expert to pull in -- a prefetch into the CPU
    // cache instead of into VRAM, which takes no slot and evicts nothing.
    auto run_warm = [&](int threads) {
        omp_set_num_threads(threads);
        std::vector<double> cold, warm;
        for (size_t i = 0; i < tour.size(); i++) {
            job js[1] = {tour[i]};
            double t0 = now_us();
            compute(js, 1, x.data(), out, qx, qact, up, gate, act);
            double t1 = now_us();
            compute(js, 1, x.data(), out, qx, qact, up, gate, act);
            double t2 = now_us();
            cold.push_back(t1 - t0);
            warm.push_back(t2 - t1);
        }
        std::sort(cold.begin(), cold.end());
        std::sort(warm.begin(), warm.end());
        const layer & L = *tour[0].L;
        const double mib = (L.up.expert_bytes + L.gate.expert_bytes + L.down.expert_bytes) / 1048576.0;
        rows.push_back({threads, -1, warm[warm.size() / 2], warm[(size_t) (warm.size() * 0.9)], 0, mib});
        printf("  threads %2d   cold %7.1f us   warm (in CPU cache) %7.1f us   ratio %.2fx\n", threads,
               cold[cold.size() / 2], warm[warm.size() / 2], cold[cold.size() / 2] / warm[warm.size() / 2]);
    };
    printf("\n[4] the same expert again, now in the CPU cache\n");
    for (int th : {4, 8, 16}) {
        run_warm(th);
    }

    // [5] An upload reads the expert out of the same DIMMs the CPU chain is
    // streaming from. Emulate that traffic with threads that only read memory,
    // and see what it does to one cold expert.
    printf("\n[5] one cold expert while other threads stream memory (what an upload in flight does to a miss)\n");
    {
        const layer & bgL   = layers.back();
        const size_t  bgn   = bgL.up.expert_bytes * (size_t) bgL.up.n_expert;
        for (int n_bg : {0, 1, 2, 4}) {
            std::atomic<bool>        stop{false};
            std::atomic<uint64_t>    bg_bytes{0};
            std::vector<std::thread> bg;
            for (int b = 0; b < n_bg; b++) {
                bg.emplace_back([&, b]() {
                    uint64_t sink = 0;
                    size_t   off  = (size_t) b * (bgn / 4);
                    while (!stop.load(std::memory_order_relaxed)) {
                        const uint64_t * q = (const uint64_t *) (bgL.up.data + (off & ~(size_t) 63));
                        for (size_t k = 0; k < (2u << 20) / 8; k += 8) {
                            sink += q[k];
                        }
                        off = (off + (2u << 20)) % (bgn - (4u << 20));
                        bg_bytes.fetch_add(2u << 20, std::memory_order_relaxed);
                    }
                    if (sink == 42) {
                        printf(" ");
                    }
                });
            }
            omp_set_num_threads(8);
            std::vector<double> t;
            const double w0 = now_us();
            for (size_t i = 0; i < tour.size(); i++) {
                if (tour[i].L == &bgL) {
                    continue;   // do not compute the layer the readers are walking
                }
                job js[1] = {tour[i]};
                const double t0 = now_us();
                compute(js, 1, x.data(), out, qx, qact, up, gate, act);
                t.push_back(now_us() - t0);
            }
            const double w1 = now_us();
            stop = true;
            for (auto & th : bg) {
                th.join();
            }
            std::sort(t.begin(), t.end());
            rows.push_back({8, -10 - n_bg, t[t.size() / 2], t[(size_t) (t.size() * 0.9)], 0, 0});
            printf("  8 compute threads, %d reader%s (%5.1f GB/s of other traffic)   %7.1f us per expert p50\n", n_bg,
                   n_bg == 1 ? " " : "s", bg_bytes.load() / (w1 - w0) / 1e3, t[t.size() / 2]);
        }
    }

    if (json_out) {
        FILE * f = fopen(json_out, "w");
        if (f) {
            fprintf(f, "{\n  \"model\": \"%s\",\n  \"loaded_gib\": %.3f,\n  \"rows\": [\n", path, total / 1073741824.0);
            for (size_t i = 0; i < rows.size(); i++) {
                fprintf(f, "    {\"threads\": %d, \"experts_per_call\": %d, \"us_p50\": %.2f, \"us_p90\": %.2f, "
                           "\"us_mean\": %.2f, \"mib_per_call\": %.3f}%s\n",
                        rows[i].threads, rows[i].m, rows[i].p50, rows[i].p90, rows[i].mean, rows[i].mib,
                        i + 1 < rows.size() ? "," : "");
            }
            fprintf(f, "  ]\n}\n");
            fclose(f);
        }
    }
    close(fd);
    return 0;
}
