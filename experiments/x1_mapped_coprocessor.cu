// x1: can a running GPU graph and a host thread exchange a miss through pinned
// mapped memory, with no CUDA synchronisation call on the path?
//
// Today every cached layer leaves the GPU: the scheduler synchronises the
// stream, copies the layer input and the routed ids to the host, runs the CPU
// expert chain, copies the result back, and relaunches. This program measures
// that round trip against a different protocol, on this card and this slot:
//
//   GPU graph:  ... router -> post request -> cached-expert chain -> wait -> join ...
//   host:       spin on a word in pinned memory -> compute -> write the answer
//
// The GPU posts a request by STORING to host memory from a kernel, runs its own
// work, then a one-thread kernel spins on a host word until the answer is
// there. Nothing in that path is a CUDA API call, so the whole token can stay
// one captured CUDA graph. If a layer has no miss it posts nothing and waits
// for nothing.
//
// What this answers:
//   1. Does it work at all (integrity over many exchanges, no torn payloads)?
//   2. What one exchange costs, against the synchronising round trip.
//   3. Whether host time hides behind GPU work (the overlap claim).
//   4. Whether a DMA upload on a second stream proceeds while a kernel spins.
//   5. How fast a kernel can read host memory in bulk (the cost of loading a
//      missed expert from the device side, which is what SeqMoE does).
//
// build:
//   $CUDA_HOME/bin/nvcc -O3 -std=c++17 -arch=sm_120 x1_mapped_coprocessor.cu \
//       -o x1_mapped_coprocessor -L$CUDA_HOME/lib -Xlinker -rpath -Xlinker $CUDA_HOME/lib -lpthread
// run:
//   ./x1_mapped_coprocessor [json_out]

#include <cuda_runtime.h>

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, \
                    __LINE__);                                                             \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

static const int N_EMBD  = 2048;
static const int N_LAYER = 48;
static const int MAX_ROWS = 8;

struct ctl_t {
    volatile uint64_t head;  // GPU -> host: (seq << 8) | layer
    char              pad0[56];
    volatile uint64_t resp;  // host -> GPU: seq whose answer is ready
    char              pad1[56];
    volatile uint64_t err;   // GPU -> host: seq of a wait that timed out
    char              pad2[56];
};

// Exactly representable in a float, so equality is a real integrity check.
__host__ __device__ inline float pat(uint64_t seq, int i) {
    return (float) ((seq * 2654435761ull + (uint64_t) i * 40503ull) & 0xFFFFull);
}

__host__ __device__ inline uint64_t seq_of(uint64_t tok, int layer) {
    return tok * 64ull + (uint64_t) layer + 1ull;
}

template <class T> static void dmalloc(T *& p, size_t bytes) {
    CK(cudaMalloc((void **) &p, bytes));
}

template <class T> static void hmalloc(T *& p, size_t bytes, unsigned flags) {
    CK(cudaHostAlloc((void **) &p, bytes, flags));
}

static inline double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- kernels --

__global__ void k_token(uint64_t * dev_tok) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*dev_tok)++;
    }
}

// stands in for attention + router: the layer's activation appears on the device
__global__ void k_make_x(float * dev_x, const uint64_t * dev_tok, int layer) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_EMBD) {
        dev_x[i] = pat(seq_of(*dev_tok, layer), i);
    }
}

// stands in for the cached-expert chain: burns a calibrated amount of GPU time
__global__ void k_work(float * out, int iters) {
    float a = (float) threadIdx.x * 1e-3f + 1.0f;
    for (int i = 0; i < iters; i++) {
        a = a * 1.0000001f + 1e-7f;
    }
    if (threadIdx.x == 0) {
        out[blockIdx.x] = a;
    }
}

// request payload by kernel stores into host memory
__global__ void k_export(float * host_x, const float * dev_x, const unsigned char * dev_miss, int layer) {
    if (!dev_miss[layer]) {
        return;
    }
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_EMBD) {
        host_x[i] = dev_x[i];
    }
}

// one store: the request becomes visible to the host
__global__ void k_post(ctl_t * ctl, const uint64_t * dev_tok, const unsigned char * dev_miss, int layer) {
    if (!dev_miss[layer]) {
        return;
    }
    ctl->head = (seq_of(*dev_tok, layer) << 8) | (uint64_t) layer;
}

// one thread spins on a host word
__global__ void k_wait(ctl_t * ctl, const uint64_t * dev_tok, const unsigned char * dev_miss, int layer,
                       uint64_t * dev_spins, long long max_iter) {
    if (!dev_miss[layer]) {
        return;
    }
    const uint64_t seq = seq_of(*dev_tok, layer);
    long long      it  = 0;
    while (ctl->resp != seq) {
        if (++it >= max_iter) {
            ctl->err = seq;
            break;
        }
    }
    *dev_spins += (uint64_t) it;
}

// join: take the host's rows (read straight from host memory, or from a device
// staging buffer a memcpy filled) and check them
__global__ void k_join(const float * resp, float * dev_out, const uint64_t * dev_tok,
                       const unsigned char * dev_miss, int layer, int n_rows, unsigned int * dev_bad) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows * N_EMBD) {
        return;
    }
    float v = 0.0f;
    if (dev_miss[layer]) {
        v = resp[i];
        if (v != pat(seq_of(*dev_tok, layer) ^ 0x5bd1e995ull, i)) {
            atomicAdd(dev_bad, 1u);
        }
    }
    dev_out[i] = v;
}

__global__ void k_setword(uint64_t * w, uint64_t v) {
    *w = v;
}

__global__ void k_copy16(const uint4 * src, uint4 * dst, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}

// ------------------------------------------------------------------- host --

struct server_cfg {
    ctl_t *            ctl;
    float *            host_x;
    float *            host_resp;
    int                n_rows     = 1;
    double             delay_us   = 0;
    bool               check_x    = true;
    // optional: a DMA upload on a second stream while the GPU waits
    cudaStream_t       up_stream  = nullptr;
    void *             up_dst     = nullptr;
    void *             up_src     = nullptr;
    size_t             up_bytes   = 0;
    std::atomic<bool>  stop{false};
    std::atomic<bool>  ready{false};
    uint64_t           last0      = 0;   // head as it stood before the graph was launched
    uint64_t           served     = 0;
    uint64_t           bad        = 0;
    double             up_us_sum  = 0;
    uint64_t           up_n       = 0;
};

static void serve(server_cfg * c) {
    // `last` is taken by the launching thread, not here: a thread takes tens of
    // microseconds to start, and a server that reads the head word after the
    // GPU has already posted to it treats that request as old and never
    // answers it. (The first version of this probe did exactly that and
    // reported two protocol failures that were its own.)
    uint64_t last = c->last0;
    c->ready.store(true, std::memory_order_release);
    while (!c->stop.load(std::memory_order_relaxed)) {
        const uint64_t h = c->ctl->head;
        if (h == last) {
            _mm_pause();
            continue;
        }
        const double t = now_us();
        last = h;
        const uint64_t seq = h >> 8;
        if (c->check_x) {
            for (int i = 0; i < N_EMBD; i++) {
                if (c->host_x[i] != pat(seq, i)) {
                    c->bad++;
                }
            }
        }
        const uint64_t rs = seq ^ 0x5bd1e995ull;
        for (int i = 0; i < c->n_rows * N_EMBD; i++) {
            c->host_resp[i] = pat(rs, i);
        }
        if (c->up_bytes) {
            const double u0 = now_us();
            CK(cudaMemcpyAsync(c->up_dst, c->up_src, c->up_bytes, cudaMemcpyHostToDevice, c->up_stream));
            CK(cudaStreamSynchronize(c->up_stream));
            c->up_us_sum += now_us() - u0;
            c->up_n++;
        }
        while (now_us() - t < c->delay_us) {
            _mm_pause();
        }
        c->ctl->resp = seq;
        c->served++;
    }
}

struct pct {
    double p50, p90, p99, mean, mn;
};

static pct summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double s = 0;
    for (double x : v) {
        s += x;
    }
    auto at = [&](double q) { return v[std::min(v.size() - 1, (size_t) (q * v.size()))]; };
    return {at(0.50), at(0.90), at(0.99), s / v.size(), v.front()};
}

struct bufs {
    cudaStream_t    s;
    uint64_t *      dev_tok;
    uint64_t *      dev_spins;
    unsigned int *  dev_bad;
    unsigned char * dev_miss;
    float *         dev_x;
    float *         dev_out;
    float *         dev_work;
    float *         dev_resp;   // staging for the memcpy variant
    ctl_t *         ctl;        // host
    ctl_t *         ctl_d;      // same block, device pointer
    float *         host_x;
    float *         host_x_d;
    float *         host_resp;
    float *         host_resp_d;
    unsigned char * host_miss;  // pinned
};

enum { EXPORT_KERNEL = 0, EXPORT_MEMCPY = 1 };
enum { RESP_KERNEL = 0, RESP_MEMCPY = 1 };

// Enqueue one token's worth of layers. `exchange` false leaves the host out
// entirely: that is the all-on-GPU reference the others are compared against.
static void enqueue_token(const bufs & b, bool exchange, int exp_mode, int resp_mode, int n_rows, int work_iters) {
    const long long max_iter = 1000000;
    k_token<<<1, 1, 0, b.s>>>(b.dev_tok);
    for (int l = 0; l < N_LAYER; l++) {
        k_make_x<<<N_EMBD / 256, 256, 0, b.s>>>(b.dev_x, b.dev_tok, l);
        if (exchange) {
            if (exp_mode == EXPORT_KERNEL) {
                k_export<<<N_EMBD / 256, 256, 0, b.s>>>(b.host_x_d, b.dev_x, b.dev_miss, l);
            } else {
                CK(cudaMemcpyAsync(b.host_x, b.dev_x, N_EMBD * sizeof(float), cudaMemcpyDeviceToHost, b.s));
            }
            k_post<<<1, 1, 0, b.s>>>(b.ctl_d, b.dev_tok, b.dev_miss, l);
        }
        if (work_iters > 0) {
            k_work<<<8, 256, 0, b.s>>>(b.dev_work, work_iters);
        }
        if (exchange) {
            k_wait<<<1, 1, 0, b.s>>>(b.ctl_d, b.dev_tok, b.dev_miss, l, b.dev_spins, max_iter);
            const int blocks = (n_rows * N_EMBD + 255) / 256;
            if (resp_mode == RESP_KERNEL) {
                k_join<<<blocks, 256, 0, b.s>>>(b.host_resp_d, b.dev_out, b.dev_tok, b.dev_miss, l, n_rows, b.dev_bad);
            } else {
                CK(cudaMemcpyAsync(b.dev_resp, b.host_resp, (size_t) n_rows * N_EMBD * sizeof(float),
                                   cudaMemcpyHostToDevice, b.s));
                k_join<<<blocks, 256, 0, b.s>>>(b.dev_resp, b.dev_out, b.dev_tok, b.dev_miss, l, n_rows, b.dev_bad);
            }
        }
    }
}

struct result {
    std::string name;
    double      us_per_layer_p50;
    double      us_per_layer_p90;
    double      us_per_layer_mean;
    uint64_t    served;
    uint64_t    bad_host;
    unsigned    bad_dev;
    uint64_t    timeouts;
    double      spins_per_wait;
    double      up_us;
};

// Run `n_tok` tokens of a captured graph and report the wall time per layer.
static result run_graph(const char * name, bufs & b, bool exchange, int exp_mode, int resp_mode, int n_rows,
                        int work_iters, double delay_us, const std::vector<unsigned char> & miss, int n_tok,
                        bool upload = false, void * up_dst = nullptr, void * up_src = nullptr, size_t up_bytes = 0,
                        cudaStream_t up_stream = nullptr) {
    memcpy(b.host_miss, miss.data(), N_LAYER);
    CK(cudaMemcpy(b.dev_miss, b.host_miss, N_LAYER, cudaMemcpyHostToDevice));
    CK(cudaMemset(b.dev_spins, 0, sizeof(uint64_t)));
    CK(cudaMemset(b.dev_bad, 0, sizeof(unsigned int)));
    b.ctl->err = 0;

    cudaGraph_t     graph;
    cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(b.s, cudaStreamCaptureModeGlobal));
    enqueue_token(b, exchange, exp_mode, resp_mode, n_rows, work_iters);
    CK(cudaStreamEndCapture(b.s, &graph));
    CK(cudaGraphInstantiate(&exec, graph, 0));

    server_cfg c;
    c.ctl       = b.ctl;
    c.host_x    = b.host_x;
    c.host_resp = b.host_resp;
    c.n_rows    = n_rows;
    c.delay_us  = delay_us;
    if (upload) {
        c.up_stream = up_stream;
        c.up_dst    = up_dst;
        c.up_src    = up_src;
        c.up_bytes  = up_bytes;
    }
    c.last0 = b.ctl->head;
    std::thread th(serve, &c);
    while (!c.ready.load(std::memory_order_acquire)) {
        _mm_pause();
    }

    int n_miss = 0;
    for (unsigned char m : miss) {
        n_miss += m;
    }

    std::vector<double> per_layer;
    for (int t = 0; t < n_tok + 20; t++) {
        const double t0 = now_us();
        CK(cudaGraphLaunch(exec, b.s));
        CK(cudaStreamSynchronize(b.s));
        const double t1 = now_us();
        if (t >= 20) {
            per_layer.push_back((t1 - t0) / N_LAYER);
        }
        if (b.ctl->err) {
            printf("  %s: a GPU wait TIMED OUT at seq %llu after %d tokens -- stopping this case\n", name,
                   (unsigned long long) b.ctl->err, t + 1);
            break;
        }
    }
    c.stop = true;
    th.join();
    if (per_layer.empty()) {
        per_layer.push_back(-1.0);
    }

    uint64_t spins = 0;
    unsigned bad   = 0;
    CK(cudaMemcpy(&spins, b.dev_spins, sizeof(spins), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&bad, b.dev_bad, sizeof(bad), cudaMemcpyDeviceToHost));
    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(graph));

    const pct p = summarize(per_layer);
    result    r;
    r.name              = name;
    r.us_per_layer_p50  = p.p50;
    r.us_per_layer_p90  = p.p90;
    r.us_per_layer_mean = p.mean;
    r.served            = c.served;
    r.bad_host          = c.bad;
    r.bad_dev           = bad;
    r.timeouts          = b.ctl->err ? 1 : 0;
    r.spins_per_wait    = c.served ? (double) spins / (double) c.served : 0.0;
    r.up_us             = c.up_n ? c.up_us_sum / (double) c.up_n : 0.0;
    printf("  %-44s %7.2f us/layer p50  %7.2f p90   served %7llu  bad host/dev %llu/%u  timeout %llu"
           "  spins/wait %.0f%s\n",
           name, p.p50, p.p90, (unsigned long long) c.served, (unsigned long long) c.bad, bad,
           (unsigned long long) r.timeouts, r.spins_per_wait,
           (exchange && !b.ctl->err && c.served != (uint64_t) n_miss * (n_tok + 20)) ? "  (SERVED COUNT MISMATCH)" : "");
    if (upload) {
        printf("      upload on a second stream while the GPU waits: %.1f us per %.2f MiB copy\n", r.up_us,
               up_bytes / 1048576.0);
    }
    return r;
}

// What the scheduler does today, reduced to its synchronising calls: wait for
// the GPU, two device-to-host copies, host work, one host-to-device copy.
static result run_conventional(const char * name, bufs & b, int n_rows, int work_iters, int n_tok, bool do_hop) {
    std::vector<double> per_layer;
    int32_t * ids_host = nullptr;
    int32_t * dev_ids  = nullptr;
    hmalloc(ids_host, 8 * sizeof(int32_t), cudaHostAllocDefault);
    dmalloc(dev_ids, 8 * sizeof(int32_t));
    CK(cudaMemset(dev_ids, 0, 8 * sizeof(int32_t)));
    CK(cudaMemset(b.dev_miss, 1, N_LAYER));
    CK(cudaMemset(b.dev_bad, 0, sizeof(unsigned int)));
    uint64_t bad_host = 0;
    uint64_t tok      = 0;
    CK(cudaMemcpy(&tok, b.dev_tok, sizeof(tok), cudaMemcpyDeviceToHost));

    for (int t = 0; t < n_tok + 20; t++) {
        const double t0 = now_us();
        k_token<<<1, 1, 0, b.s>>>(b.dev_tok);
        tok++;
        for (int l = 0; l < N_LAYER; l++) {
            k_make_x<<<N_EMBD / 256, 256, 0, b.s>>>(b.dev_x, b.dev_tok, l);
            if (do_hop) {
                CK(cudaMemcpyAsync(b.host_x, b.dev_x, N_EMBD * sizeof(float), cudaMemcpyDeviceToHost, b.s));
                CK(cudaStreamSynchronize(b.s));
                CK(cudaMemcpyAsync(ids_host, dev_ids, 8 * sizeof(int32_t), cudaMemcpyDeviceToHost, b.s));
                CK(cudaStreamSynchronize(b.s));
                const uint64_t seq = seq_of(tok, l);
                for (int i = 0; i < N_EMBD; i++) {
                    if (b.host_x[i] != pat(seq, i)) {
                        bad_host++;
                    }
                }
                const uint64_t rs = seq ^ 0x5bd1e995ull;
                for (int i = 0; i < n_rows * N_EMBD; i++) {
                    b.host_resp[i] = pat(rs, i);
                }
                CK(cudaMemcpyAsync(b.dev_resp, b.host_resp, (size_t) n_rows * N_EMBD * sizeof(float),
                                   cudaMemcpyHostToDevice, b.s));
                CK(cudaStreamSynchronize(b.s));
            }
            if (work_iters > 0) {
                k_work<<<8, 256, 0, b.s>>>(b.dev_work, work_iters);
            }
            if (do_hop) {
                const int blocks = (n_rows * N_EMBD + 255) / 256;
                k_join<<<blocks, 256, 0, b.s>>>(b.dev_resp, b.dev_out, b.dev_tok, b.dev_miss, l, n_rows, b.dev_bad);
            }
        }
        CK(cudaStreamSynchronize(b.s));
        const double t1 = now_us();
        if (t >= 20) {
            per_layer.push_back((t1 - t0) / N_LAYER);
        }
    }
    unsigned bad = 0;
    CK(cudaMemcpy(&bad, b.dev_bad, sizeof(bad), cudaMemcpyDeviceToHost));
    CK(cudaFree(dev_ids));
    CK(cudaFreeHost(ids_host));
    const pct p = summarize(per_layer);
    printf("  %-44s %7.2f us/layer p50  %7.2f p90   bad host/dev %llu/%u\n", name, p.p50, p.p90,
           (unsigned long long) bad_host, bad);
    result r{};
    r.name              = name;
    r.us_per_layer_p50  = p.p50;
    r.us_per_layer_p90  = p.p90;
    r.us_per_layer_mean = p.mean;
    r.bad_host          = bad_host;
    r.bad_dev           = bad;
    return r;
}

int main(int argc, char ** argv) {
    const char * json_out = argc > 1 ? argv[1] : nullptr;
    const int    n_tok    = getenv("X1_TOKENS") ? atoi(getenv("X1_TOKENS")) : 400;

    if (getenv("X1_SPIN")) {
        CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    }
    CK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s  (unified addressing %d, can map host memory %d)\n", prop.name, prop.unifiedAddressing,
           prop.canMapHostMemory);

    bufs b{};
    CK(cudaStreamCreate(&b.s));
    dmalloc(b.dev_tok, sizeof(uint64_t));
    dmalloc(b.dev_spins, sizeof(uint64_t));
    dmalloc(b.dev_bad, sizeof(unsigned int));
    dmalloc(b.dev_miss, N_LAYER);
    dmalloc(b.dev_x, N_EMBD * sizeof(float));
    dmalloc(b.dev_out, MAX_ROWS * N_EMBD * sizeof(float));
    dmalloc(b.dev_work, 4096 * sizeof(float));
    dmalloc(b.dev_resp, MAX_ROWS * N_EMBD * sizeof(float));
    CK(cudaMemset(b.dev_tok, 0, sizeof(uint64_t)));

    hmalloc(b.ctl, sizeof(ctl_t), cudaHostAllocMapped);
    hmalloc(b.host_x, N_EMBD * sizeof(float), cudaHostAllocMapped);
    hmalloc(b.host_resp, MAX_ROWS * N_EMBD * sizeof(float), cudaHostAllocMapped);
    hmalloc(b.host_miss, N_LAYER, cudaHostAllocDefault);
    memset(b.ctl, 0, sizeof(ctl_t));
    memset(b.host_x, 0, N_EMBD * sizeof(float));
    memset(b.host_resp, 0, MAX_ROWS * N_EMBD * sizeof(float));
    CK(cudaHostGetDevicePointer((void **) &b.ctl_d, b.ctl, 0));
    CK(cudaHostGetDevicePointer((void **) &b.host_x_d, b.host_x, 0));
    CK(cudaHostGetDevicePointer((void **) &b.host_resp_d, b.host_resp, 0));
    printf("host ctl %p, device view %p  (%s)\n", (void *) b.ctl, (void *) b.ctl_d,
           (void *) b.ctl == (void *) b.ctl_d ? "same address: unified" : "different addresses");

    // -- calibrate the stand-in for the cached-expert chain to ~40 us ---------
    int work_iters = 20000;
    {
        for (int rep = 0; rep < 6; rep++) {
            std::vector<double> v;
            for (int i = 0; i < 300; i++) {
                const double t0 = now_us();
                k_work<<<8, 256, 0, b.s>>>(b.dev_work, work_iters);
                CK(cudaStreamSynchronize(b.s));
                v.push_back(now_us() - t0);
            }
            const double got = summarize(v).p50;
            if (rep == 5) {
                printf("work kernel: %d iterations, %.1f us from launch to done "
                       "(about 40 us of GPU time above the launch floor)\n", work_iters, got);
                break;
            }
            // launch+sync has a floor of its own; scale only the part above it
            std::vector<double> f;
            for (int i = 0; i < 300; i++) {
                const double t0 = now_us();
                k_work<<<8, 256, 0, b.s>>>(b.dev_work, 1);
                CK(cudaStreamSynchronize(b.s));
                f.push_back(now_us() - t0);
            }
            const double floor_us = summarize(f).p50;
            const double body     = std::max(1.0, got - floor_us);
            work_iters            = std::max(100, (int) (work_iters * 40.0 / body));
        }
    }

    std::vector<result> res;
    std::vector<unsigned char> all(N_LAYER, 1), none(N_LAYER, 0), some(N_LAYER, 0);
    for (int l = 0; l < N_LAYER; l++) {
        some[l] = (l * 7 + 3) % 16 < 9;   // 56% of layers, the share that miss at a 90% hit rate
    }

    printf("\n[1] launch + synchronise of one trivial kernel (the floor under every hop today)\n");
    {
        std::vector<double> v;
        for (int i = 0; i < 4000; i++) {
            const double t0 = now_us();
            k_token<<<1, 1, 0, b.s>>>(b.dev_tok);
            CK(cudaStreamSynchronize(b.s));
            v.push_back(now_us() - t0);
        }
        const pct p = summarize(v);
        printf("  p50 %.2f us   p90 %.2f   p99 %.2f   min %.2f\n", p.p50, p.p90, p.p99, p.mn);
        result r{};
        r.name = "launch_sync_floor";
        r.us_per_layer_p50 = p.p50; r.us_per_layer_p90 = p.p90; r.us_per_layer_mean = p.mean;
        res.push_back(r);
    }

    printf("\n[2] one token = %d layers, wall time per layer. GPU work per layer is the %d-iteration kernel.\n",
           N_LAYER, work_iters);
    printf("  -- no host involvement (what an all-on-GPU layer costs in this harness)\n");
    res.push_back(run_conventional("direct launches, no hop", b, 1, work_iters, n_tok, false));
    res.push_back(run_graph("captured graph, no exchange", b, false, 0, 0, 1, work_iters, 0, none, n_tok));

    printf("  -- today: synchronise, 2 copies down, host work, 1 copy up (every layer)\n");
    res.push_back(run_conventional("sync round trip, 1 row back", b, 1, work_iters, n_tok, true));
    res.push_back(run_conventional("sync round trip, 8 rows back", b, 8, work_iters, n_tok, true));

    printf("  -- mapped exchange in one captured graph, every layer misses, host answers at once\n");
    res.push_back(run_graph("export kernel / join reads host, 1 row", b, true, EXPORT_KERNEL, RESP_KERNEL, 1, work_iters, 0, all, n_tok));
    res.push_back(run_graph("export kernel / join reads host, 2 rows", b, true, EXPORT_KERNEL, RESP_KERNEL, 2, work_iters, 0, all, n_tok));
    res.push_back(run_graph("export kernel / join reads host, 8 rows", b, true, EXPORT_KERNEL, RESP_KERNEL, 8, work_iters, 0, all, n_tok));
    res.push_back(run_graph("export memcpy / join reads host, 1 row", b, true, EXPORT_MEMCPY, RESP_KERNEL, 1, work_iters, 0, all, n_tok));
    res.push_back(run_graph("export kernel / response memcpy, 1 row", b, true, EXPORT_KERNEL, RESP_MEMCPY, 1, work_iters, 0, all, n_tok));
    res.push_back(run_graph("export memcpy / response memcpy, 8 rows", b, true, EXPORT_MEMCPY, RESP_MEMCPY, 8, work_iters, 0, all, n_tok));

    printf("  -- same graph, but no layer misses (kernels return at once; the host is never involved)\n");
    res.push_back(run_graph("exchange kernels present, 0 of 48 miss", b, true, EXPORT_KERNEL, RESP_KERNEL, 1, work_iters, 0, none, n_tok));
    printf("  -- 27 of 48 layers miss (a 90%% hit rate)\n");
    res.push_back(run_graph("exchange, 27 of 48 layers miss", b, true, EXPORT_KERNEL, RESP_KERNEL, 1, work_iters, 0, some, n_tok));

    printf("\n[3] does host time hide behind GPU work? every layer misses, host takes d us to answer\n");
    for (double d : {10.0, 20.0, 30.0, 40.0, 60.0, 100.0}) {
        char nm[96];
        snprintf(nm, sizeof(nm), "host answers after %3.0f us", d);
        res.push_back(run_graph(nm, b, true, EXPORT_KERNEL, RESP_KERNEL, 1, work_iters, d, all, n_tok / 2));
    }

    printf("\n[4] a 2.78 MiB upload on a second stream while the main stream's kernel is spinning\n");
    {
        const size_t up_bytes = (size_t) (2.78 * 1048576);
        void *       up_src   = nullptr;
        void *       up_dst   = nullptr;
        cudaStream_t s2;
        CK(cudaStreamCreate(&s2));
        CK(cudaHostAlloc(&up_src, up_bytes, cudaHostAllocDefault));
        CK(cudaMalloc(&up_dst, up_bytes));
        memset(up_src, 7, up_bytes);
        res.push_back(run_graph("exchange with an upload inside each wait", b, true, EXPORT_KERNEL, RESP_KERNEL, 1,
                                work_iters, 0, all, n_tok / 4, true, up_dst, up_src, up_bytes, s2));
        // the same copy with nothing else running, for comparison
        std::vector<double> v;
        for (int i = 0; i < 400; i++) {
            const double t0 = now_us();
            CK(cudaMemcpyAsync(up_dst, up_src, up_bytes, cudaMemcpyHostToDevice, s2));
            CK(cudaStreamSynchronize(s2));
            v.push_back(now_us() - t0);
        }
        const pct p = summarize(v);
        printf("      the same copy on an idle GPU: %.1f us p50 (%.1f GB/s)\n", p.p50, up_bytes / p.p50 / 1e3);
        result r{};
        r.name = "upload_idle_2.78MiB";
        r.us_per_layer_p50 = p.p50; r.us_per_layer_p90 = p.p90; r.us_per_layer_mean = p.mean;
        res.push_back(r);
        CK(cudaFreeHost(up_src));
        CK(cudaFree(up_dst));
        CK(cudaStreamDestroy(s2));
    }

    printf("\n[5] bulk read of host memory by a kernel (a device-initiated load of one missed expert)\n");
    for (size_t bytes : {(size_t) (2.65 * 1048576), (size_t) (32 * 1048576)}) {
        bytes &= ~(size_t) 15;
        void * hsrc = nullptr;
        void * hsrc_d = nullptr;
        void * ddst = nullptr;
        CK(cudaHostAlloc(&hsrc, bytes, cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer(&hsrc_d, hsrc, 0));
        CK(cudaMalloc(&ddst, bytes));
        memset(hsrc, 3, bytes);
        const size_t n16    = bytes / 16;
        const int    blocks = (int) ((n16 + 255) / 256);
        std::vector<double> vk, vm;
        for (int i = 0; i < 60; i++) {
            double t0 = now_us();
            k_copy16<<<blocks, 256, 0, b.s>>>((const uint4 *) hsrc_d, (uint4 *) ddst, n16);
            CK(cudaStreamSynchronize(b.s));
            vk.push_back(now_us() - t0);
            t0 = now_us();
            CK(cudaMemcpyAsync(ddst, hsrc, bytes, cudaMemcpyHostToDevice, b.s));
            CK(cudaStreamSynchronize(b.s));
            vm.push_back(now_us() - t0);
        }
        const pct pk = summarize(vk), pm = summarize(vm);
        printf("  %6.2f MiB   kernel read %8.1f us (%5.1f GB/s)    cudaMemcpyAsync %8.1f us (%5.1f GB/s)\n",
               bytes / 1048576.0, pk.p50, bytes / pk.p50 / 1e3, pm.p50, bytes / pm.p50 / 1e3);
        result rk{}, rm{};
        char   nm[96];
        snprintf(nm, sizeof(nm), "kernel_read_%.2fMiB", bytes / 1048576.0);
        rk.name = nm; rk.us_per_layer_p50 = pk.p50; rk.us_per_layer_p90 = pk.p90; rk.us_per_layer_mean = pk.mean;
        snprintf(nm, sizeof(nm), "memcpy_h2d_%.2fMiB", bytes / 1048576.0);
        rm.name = nm; rm.us_per_layer_p50 = pm.p50; rm.us_per_layer_p90 = pm.p90; rm.us_per_layer_mean = pm.mean;
        res.push_back(rk);
        res.push_back(rm);
        CK(cudaFreeHost(hsrc));
        CK(cudaFree(ddst));
    }

    // A fallback design keeps the host in charge but never calls a blocking
    // synchronise: it queues the layer, queues an asynchronous copy of the
    // request to pinned memory, and learns the copy has landed some other way.
    printf("\n[6] how soon the host learns that queued GPU work plus a copy-out has finished, three ways\n");
    {
        uint64_t * dev_w  = nullptr;
        uint64_t * host_w = nullptr;
        dmalloc(dev_w, sizeof(uint64_t));
        hmalloc(host_w, sizeof(uint64_t), cudaHostAllocDefault);
        *host_w = 0;
        cudaEvent_t ev;
        CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        const char * names[3] = {"cudaStreamSynchronize", "spin on cudaEventQuery", "spin on the copied word in pinned memory"};
        uint64_t v = 1000;
        for (int mode = 0; mode < 3; mode++) {
            for (int with_work = 0; with_work < 2; with_work++) {
                std::vector<double> t;
                for (int i = 0; i < 3000; i++) {
                    v++;
                    const double t0 = now_us();
                    k_make_x<<<N_EMBD / 256, 256, 0, b.s>>>(b.dev_x, b.dev_tok, 0);
                    if (with_work) {
                        k_work<<<8, 256, 0, b.s>>>(b.dev_work, work_iters);
                    }
                    k_setword<<<1, 1, 0, b.s>>>(dev_w, v);
                    CK(cudaMemcpyAsync(b.host_x, b.dev_x, N_EMBD * sizeof(float), cudaMemcpyDeviceToHost, b.s));
                    CK(cudaMemcpyAsync(host_w, dev_w, sizeof(uint64_t), cudaMemcpyDeviceToHost, b.s));
                    if (mode == 0) {
                        CK(cudaStreamSynchronize(b.s));
                    } else if (mode == 1) {
                        CK(cudaEventRecord(ev, b.s));
                        while (cudaEventQuery(ev) == cudaErrorNotReady) {
                        }
                    } else {
                        while (*(volatile uint64_t *) host_w != v) {
                            _mm_pause();
                        }
                    }
                    t.push_back(now_us() - t0);
                    if (mode != 0) {
                        CK(cudaStreamSynchronize(b.s));
                    }
                }
                const pct p = summarize(t);
                printf("  %-44s %s  %7.2f us p50  %7.2f p90  %7.2f p99\n", names[mode],
                       with_work ? "after 40 us of GPU work" : "after a trivial kernel    ", p.p50, p.p90, p.p99);
                result r{};
                r.name = std::string("observe_") + (mode == 0 ? "sync" : mode == 1 ? "eventquery" : "memspin") +
                         (with_work ? "_work" : "_trivial");
                r.us_per_layer_p50 = p.p50; r.us_per_layer_p90 = p.p90; r.us_per_layer_mean = p.mean;
                res.push_back(r);
            }
        }
        CK(cudaEventDestroy(ev));
        CK(cudaFree(dev_w));
        CK(cudaFreeHost(host_w));
    }

    if (json_out) {
        FILE * f = fopen(json_out, "w");
        if (f) {
            fprintf(f, "{\n  \"device\": \"%s\",\n  \"layers_per_token\": %d,\n  \"work_iters\": %d,\n  \"results\": [\n",
                    prop.name, N_LAYER, work_iters);
            for (size_t i = 0; i < res.size(); i++) {
                const result & r = res[i];
                fprintf(f,
                        "    {\"name\": \"%s\", \"us_p50\": %.3f, \"us_p90\": %.3f, \"us_mean\": %.3f, "
                        "\"served\": %llu, \"bad_host\": %llu, \"bad_dev\": %u, \"timeouts\": %llu, "
                        "\"spins_per_wait\": %.1f, \"upload_us\": %.2f}%s\n",
                        r.name.c_str(), r.us_per_layer_p50, r.us_per_layer_p90, r.us_per_layer_mean,
                        (unsigned long long) r.served, (unsigned long long) r.bad_host, r.bad_dev,
                        (unsigned long long) r.timeouts, r.spins_per_wait, r.up_us, i + 1 < res.size() ? "," : "");
            }
            fprintf(f, "  ]\n}\n");
            fclose(f);
        }
    }
    return 0;
}
