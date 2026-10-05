// Unit tests for the just-in-time streaming planner (engine/src/llama-moe-planner.h).
// Pure host logic; no GPU.
#include "llama-moe-planner.h"

#include <cstdio>
#include <cstdlib>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

static const size_t EXPERT = 2831155;          // ~2.7 MiB, the three slices of one Qwen3-30B expert
static const double FAST   = 1e9;              // bytes per ns: nothing is ever late
static const double PCIE   = 45.7;             // 45.7 GB/s

static void test_dedupe_and_refresh() {
    moe_planner p(512 * 1024);
    CHECK(p.offer(5, 10, EXPERT, 1000) == 0);
    CHECK(p.offer(5, 10, EXPERT, 800) == -1);   // the d = 1 refresh of a d = 2 job
    CHECK(p.get(0).deadline_ns == 800);
    CHECK(p.offer(5, 10, EXPERT, 900) == -1);   // a later deadline never loosens it
    CHECK(p.get(0).deadline_ns == 800);
    CHECK(p.offer(6, 10, EXPERT, 900) == 1);    // same expert, other layer: a different job
    CHECK(p.size() == 2);
}

static void test_edf_order() {
    moe_planner p(512 * 1024);
    const int a = p.offer(7, 1, EXPERT, 500000);
    const int b = p.offer(6, 2, EXPERT, 300000);
    moe_planner::chunk c;
    CHECK(p.next(0, FAST, -1, c) && c.job == b);
    // b runs to completion before a gets a chunk
    while (!c.last) {
        CHECK(p.next(0, FAST, -1, c) && c.job == b);
    }
    CHECK(p.next(0, FAST, -1, c) && c.job == a);
}

static void test_chunks() {
    moe_planner p(512 * 1024);
    p.offer(3, 4, EXPERT, 1000000);
    moe_planner::chunk c;
    size_t off = 0;
    int n = 0;
    while (p.next(0, FAST, -1, c)) {
        CHECK(c.job == 0 && c.offset == off && c.len <= 512 * 1024 && c.len > 0);
        off += c.len;
        ++n;
        CHECK(c.last == (off == EXPERT));
    }
    CHECK(off == EXPERT && n == 6);
    CHECK(p.get(0).sent == EXPERT && !p.get(0).dropped);
}

static void test_drop_late() {
    moe_planner p(512 * 1024);
    p.offer(3, 4, EXPERT, 10000);               // needs ~62 us at 45.7 GB/s, has 10 us
    moe_planner::chunk c;
    CHECK(!p.next(0, PCIE, -1, c));
    CHECK(p.get(0).dropped && p.get(0).drop == moe_planner::DROP_LATE);
    CHECK(p.n_drop_late == 1);
    // a job that fits is still sent; the stream being busy until free_at counts against it
    p.offer(3, 5, EXPERT, 70000);
    CHECK(p.next(0, PCIE, -1, c) && c.job == 1);
    moe_planner q(512 * 1024);
    q.offer(3, 5, EXPERT, 70000);
    CHECK(!q.next(20000, PCIE, -1, c));         // the same job behind 20 us of queued copies
}

static void test_posted() {
    moe_planner p(512 * 1024);
    p.offer(9, 1, EXPERT, 1000000);
    p.offer(10, 2, EXPERT, 2000000);
    moe_planner::chunk c;
    CHECK(p.next(0, FAST, -1, c) && c.job == 0);
    p.posted(9);                                // layer 9 has read its table: too late for job 0
    CHECK(p.get(0).dropped && p.get(0).drop == moe_planner::DROP_POSTED);
    CHECK(!p.get(1).dropped);
    CHECK(p.next(0, FAST, -1, c) && c.job == 1);
    CHECK(p.offer(9, 3, EXPERT, 3000000) == -1); // nothing new for a layer that already posted
    p.posted(10);
    CHECK(p.n_drop_posted == 2);
}

static void test_hold() {
    moe_planner p(512 * 1024);
    const int near = p.offer(5, 1, EXPERT, 100000);
    const int far  = p.offer(8, 2, EXPERT, 50000);
    moe_planner::chunk c;
    CHECK(p.next(0, FAST, 6, c) && c.job == near);   // the far target waits while a miss computes
    CHECK(p.next(0, FAST, -1, c) && c.job == far);   // no hold: earliest deadline
    moe_planner q(512 * 1024);
    q.offer(8, 2, EXPERT, 50000);
    CHECK(!q.next(0, FAST, 6, c) && !q.get(0).dropped);  // held, not dropped
}

static void test_preempt_at_chunk_boundary() {
    moe_planner p(512 * 1024);
    const int a = p.offer(20, 1, EXPERT, 1000000);
    moe_planner::chunk c;
    CHECK(p.next(0, FAST, -1, c) && c.job == a && !c.last);
    const int b = p.offer(15, 2, EXPERT, 200000);
    CHECK(p.next(0, FAST, -1, c) && c.job == b && c.offset == 0);
}

static void test_reset() {
    moe_planner p(512 * 1024);
    p.offer(1, 1, EXPERT, 1);
    p.posted(4);
    p.reset();
    CHECK(p.size() == 0);
    CHECK(p.offer(1, 1, EXPERT, 1000) == 0);
    CHECK(p.offer(4, 1, EXPERT, 1000) == 1);     // reset also forgets which layers posted
}

int main() {
    test_dedupe_and_refresh();
    test_edf_order();
    test_chunks();
    test_drop_late();
    test_posted();
    test_hold();
    test_preempt_at_chunk_boundary();
    test_reset();
    if (failures) {
        std::fprintf(stderr, "%d planner check(s) failed\n", failures);
        return 1;
    }
    std::printf("PASS: planner (dedupe, refresh, EDF, chunks, late drop, posted drop, hold, preemption, reset)\n");
    return 0;
}
