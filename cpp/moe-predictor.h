// Expert predictor for the llama.cpp MoE cache: loads a MOEP file and scores
// one layer's 128 experts from what the cache already knows on the host.
//
// The model is per-layer ridge over a multi-hot feature vector:
//
//     x = [ prev: experts layer L+1 used on the PREVIOUS token | 128 ]
//         [ cur:  experts layer L   used on THIS token         | 128 ]
//     s = z(x . W_L) ; s[prev] += prior_L ; take the top k
//
// Both blocks come free: llama.cpp's MoE observation callback already reports
// every routed expert id on the host, so nothing is read back from the device
// and nothing is synchronised. x has exactly 16 non-zeros out of 256, so the
// "matvec" is 16 row additions -- ~2K adds per layer, no multiplies -- against
// the 2.92 MiB fetch it decides.
//
// The file may also carry a PCA block over the hidden state (65.3% recall
// against 61.8% without). It is supported here and deliberately not used by the
// cache: measured end to end it was worth 2.4 tok/s, and reconstructing the
// residual stream from the tensor that is actually in scope at the callback
// (ffn_norm's output, not ffn_inp) is exactly the kind of silent disagreement
// that shows up as a mildly worse hit rate rather than an error.

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

enum moe_pred_block {
    MOE_PRED_BLOCK_PCA  = 0,
    MOE_PRED_BLOCK_PREV = 1,
    MOE_PRED_BLOCK_CUR  = 2,
};

struct moe_predictor {
    uint32_t n_expert = 0;
    uint32_t n_layers = 0;
    uint32_t feat_dim = 0;
    uint32_t top_k    = 0;

    uint32_t d_model = 0;
    uint32_t n_comp  = 0;
    bool     has_pca = false;

    // where each feature block starts inside x
    int32_t off_prev = -1;
    int32_t off_cur  = -1;
    int32_t off_pca  = -1;

    std::vector<float> mean, comp, scale;          // PCA, empty unless has_pca

    std::vector<int32_t> layer_of;                 // present layers, ascending
    std::vector<int32_t> index_of;                 // layer -> slot in `w`, or -1
    std::vector<float>   prior;                    // per present layer
    std::vector<float>   w;                        // [n_present][feat_dim*n_expert]

    // Load a MOEP file. Returns false and fills `err` on any mismatch; a
    // predictor that half-loaded is never returned.
    bool load(const std::string & path, std::string & err);

    bool has_layer(int il) const {
        return il >= 0 && il < (int) index_of.size() && index_of[il] >= 0;
    }

    // Score layer `il`'s weights into `out` (n_expert floats) from the two
    // expert-id lists. `prev`/`cur` may be empty.
    void score(int il, const int32_t * prev, size_t n_prev,
               const int32_t * cur, size_t n_cur, float * out) const;

    // Top-k expert ids by score, best first. Returns how many were written.
    size_t top_k_ids(const float * scores, size_t k, int32_t * out) const;
};
