#include "moe-predictor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

namespace {

bool rd(FILE * f, void * dst, size_t n) {
    return fread(dst, 1, n, f) == n;
}

bool rd_u32(FILE * f, uint32_t * v) {
    return rd(f, v, sizeof(uint32_t));
}

} // namespace

bool moe_predictor::load(const std::string & path, std::string & err) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open " + path;
        return false;
    }

    char magic[4];
    uint32_t ver = 0, n_present = 0, n_blocks = 0, pca_flag = 0, reserved = 0;
    if (!rd(f, magic, 4) || memcmp(magic, "MOEP", 4) != 0) {
        err = path + ": not a MOEP file";
        fclose(f);
        return false;
    }
    bool ok = rd_u32(f, &ver) && rd_u32(f, &d_model) && rd_u32(f, &n_comp) &&
              rd_u32(f, &n_expert) && rd_u32(f, &n_layers) && rd_u32(f, &top_k) &&
              rd_u32(f, &feat_dim) && rd_u32(f, &n_present) && rd_u32(f, &n_blocks) &&
              rd_u32(f, &pca_flag) && rd_u32(f, &reserved);
    if (!ok) {
        err = path + ": truncated header";
        fclose(f);
        return false;
    }
    if (ver != 3) {
        err = path + ": version " + std::to_string(ver) + ", expected 3";
        fclose(f);
        return false;
    }
    has_pca = pca_flag != 0;

    // The block table is the whole point of the format: it says which columns
    // of x mean what, so the loader cannot silently disagree with the trainer
    // about the feature order. A wrong order does not crash, it just predicts
    // worse -- so it is checked here rather than discovered in a benchmark.
    uint32_t expect_off = 0;
    for (uint32_t i = 0; i < n_blocks; ++i) {
        uint32_t kind = 0, off = 0, size = 0;
        if (!rd_u32(f, &kind) || !rd_u32(f, &off) || !rd_u32(f, &size)) {
            err = path + ": truncated block table";
            fclose(f);
            return false;
        }
        if (off != expect_off) {
            err = path + ": block " + std::to_string(i) + " starts at " +
                  std::to_string(off) + ", expected " + std::to_string(expect_off);
            fclose(f);
            return false;
        }
        expect_off += size;
        switch (kind) {
            case MOE_PRED_BLOCK_PCA:
                off_pca = (int32_t) off;
                if (size != n_comp) { err = path + ": PCA block size != n_comp"; fclose(f); return false; }
                break;
            case MOE_PRED_BLOCK_PREV:
                off_prev = (int32_t) off;
                if (size != n_expert) { err = path + ": prev block size != n_expert"; fclose(f); return false; }
                break;
            case MOE_PRED_BLOCK_CUR:
                off_cur = (int32_t) off;
                if (size != n_expert) { err = path + ": cur block size != n_expert"; fclose(f); return false; }
                break;
            case MOE_PRED_BLOCK_BELOW:
                off_below = (int32_t) off;
                if (size != n_expert) { err = path + ": below block size != n_expert"; fclose(f); return false; }
                break;
            case MOE_PRED_BLOCK_SELF_PREV:
                off_self_prev = (int32_t) off;
                if (size != n_expert) { err = path + ": self_prev block size != n_expert"; fclose(f); return false; }
                break;
            default:
                err = path + ": unknown feature block kind " + std::to_string(kind);
                fclose(f);
                return false;
        }
    }
    if (expect_off != feat_dim) {
        err = path + ": blocks cover " + std::to_string(expect_off) +
              " columns, feat_dim is " + std::to_string(feat_dim);
        fclose(f);
        return false;
    }
    if (has_pca != (off_pca >= 0)) {
        err = path + ": has_pca disagrees with the block table";
        fclose(f);
        return false;
    }

    if (has_pca) {
        mean.resize(d_model);
        comp.resize((size_t) d_model*n_comp);
        scale.resize(n_comp);
        if (!rd(f, mean.data(), mean.size()*4) || !rd(f, comp.data(), comp.size()*4) ||
            !rd(f, scale.data(), scale.size()*4)) {
            err = path + ": truncated PCA block";
            fclose(f);
            return false;
        }
    }

    layer_of.resize(n_present);
    prior.resize(n_present);
    bias.resize((size_t) n_present*n_expert);
    if (!rd(f, layer_of.data(), n_present*4) || !rd(f, prior.data(), n_present*4) ||
        !rd(f, bias.data(), bias.size()*4)) {
        err = path + ": truncated layer table";
        fclose(f);
        return false;
    }

    w.resize((size_t) n_present*feat_dim*n_expert);
    if (!rd(f, w.data(), w.size()*4)) {
        err = path + ": truncated weights (wanted " + std::to_string(w.size()*4) + " bytes)";
        fclose(f);
        return false;
    }
    char tail;
    if (fread(&tail, 1, 1, f) != 0) {
        err = path + ": trailing bytes after the weights";
        fclose(f);
        return false;
    }
    fclose(f);

    index_of.assign(n_layers, -1);
    for (uint32_t i = 0; i < n_present; ++i) {
        const int32_t il = layer_of[i];
        if (il < 0 || il >= (int32_t) n_layers) {
            err = path + ": layer index " + std::to_string(il) + " out of range";
            return false;
        }
        index_of[il] = (int32_t) i;
    }
    return true;
}

void moe_predictor::score(int il, const int32_t * prev, size_t n_prev,
                          const int32_t * cur, size_t n_cur, float * out) const {
    inputs in;
    in.prev = prev; in.n_prev = n_prev;
    in.cur  = cur;  in.n_cur  = n_cur;
    score(il, in, out);
}

void moe_predictor::score(int il, const inputs & in, float * out) const {
    const int32_t slot = index_of[il];
    const float * W = w.data() + (size_t) slot*feat_dim*n_expert;
    const uint32_t E = n_expert;

    // start from the layer's bias rather than zero: the fresh model is trained
    // by SGD and needs an intercept, where the ridge solution absorbed one
    // implicitly
    const float * b = bias.data() + (size_t) slot*E;
    std::copy(b, b + E, out);

    // x is multi-hot, so x . W is a sum of the rows x selects. 16 rows of 128
    // floats: no multiplies, and it vectorises.
    auto add_rows = [&](const int32_t * ids, size_t n, int32_t off) {
        if (off < 0) {
            return;
        }
        for (size_t i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            if (e < 0 || e >= (int32_t) E) {
                continue;
            }
            const float * row = W + (size_t) (off + e)*E;
            for (uint32_t j = 0; j < E; ++j) {
                out[j] += row[j];
            }
        }
    };
    add_rows(in.prev,      in.n_prev,      off_prev);
    add_rows(in.cur,       in.n_cur,       off_cur);
    add_rows(in.below,     in.n_below,     off_below);
    add_rows(in.self_prev, in.n_self_prev, off_self_prev);

    // z-score, then the repeat prior -- in that order, which is how the weight
    // was chosen at fit time. Swapping them changes what the weight means.
    float mu = 0.0f;
    for (uint32_t j = 0; j < E; ++j) {
        mu += out[j];
    }
    mu /= (float) E;
    float var = 0.0f;
    for (uint32_t j = 0; j < E; ++j) {
        const float d = out[j] - mu;
        var += d*d;
    }
    const float sd = std::sqrt(var / (float) E) + 1e-9f;
    for (uint32_t j = 0; j < E; ++j) {
        out[j] = (out[j] - mu) / sd;
    }

    const float pw = prior[slot];
    if (pw != 0.0f) {
        for (size_t i = 0; i < in.n_prev; ++i) {
            const int32_t e = in.prev[i];
            if (e >= 0 && e < (int32_t) E) {
                out[e] += pw;
            }
        }
    }
}

size_t moe_predictor::top_k_ids(const float * scores, size_t k, int32_t * out) const {
    const size_t E = n_expert;
    k = std::min(k, E);
    std::vector<int32_t> idx(E);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int32_t a, int32_t b) {
                          if (scores[a] != scores[b]) {
                              return scores[a] > scores[b];
                          }
                          return a < b;      // deterministic ties
                      });
    std::copy(idx.begin(), idx.begin() + k, out);
    return k;
}
