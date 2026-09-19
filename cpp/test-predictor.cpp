// Scores a list of cases with the C++ loader and writes the raw floats out, so
// Python can compare them against what it would have produced itself.
//
// The check that matters is not "does it load" -- a wrong feature offset loads
// fine. It is that the same (layer, prev, cur) gives the same 128 numbers in
// both, which is the only way to catch the block table being read in the wrong
// order, the prior being applied before the z-score instead of after, or the
// weight matrix being transposed.
#include "moe-predictor.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s model.bin cases.bin scores.bin\n", argv[0]);
        return 2;
    }
    moe_predictor p;
    std::string err;
    if (!p.load(argv[1], err)) {
        fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "loaded: n_expert=%u feat_dim=%u layers=%zu "
            "off_prev=%d off_cur=%d off_pca=%d\n",
            p.n_expert, p.feat_dim, p.layer_of.size(), p.off_prev, p.off_cur, p.off_pca);

    FILE * fi = fopen(argv[2], "rb");
    FILE * fo = fopen(argv[3], "wb");
    if (!fi || !fo) {
        fprintf(stderr, "cannot open cases/scores\n");
        return 1;
    }
    int32_t n_cases = 0;
    if (fread(&n_cases, 4, 1, fi) != 1) {
        return 1;
    }
    std::vector<float> s(p.n_expert);
    std::vector<int32_t> ids[4];
    std::vector<int32_t> top(8);
    for (int32_t c = 0; c < n_cases; ++c) {
        int32_t il = 0;
        if (fread(&il, 4, 1, fi) != 1) return 1;
        // four id lists, in the fixed order prev, cur, below, self_prev
        for (int b = 0; b < 4; ++b) {
            int32_t n = 0;
            if (fread(&n, 4, 1, fi) != 1) return 1;
            ids[b].resize(n);
            if (n && fread(ids[b].data(), 4, n, fi) != (size_t) n) return 1;
        }
        if (!p.has_layer(il)) {
            fprintf(stderr, "case %d: layer %d absent\n", c, il);
            return 1;
        }
        moe_predictor::inputs in;
        in.prev      = ids[0].data(); in.n_prev      = ids[0].size();
        in.cur       = ids[1].data(); in.n_cur       = ids[1].size();
        in.below     = ids[2].data(); in.n_below     = ids[2].size();
        in.self_prev = ids[3].data(); in.n_self_prev = ids[3].size();
        p.score(il, in, s.data());
        fwrite(s.data(), 4, p.n_expert, fo);
        p.top_k_ids(s.data(), 8, top.data());
        fwrite(top.data(), 4, 8, fo);
    }
    fclose(fi);
    fclose(fo);
    return 0;
}
