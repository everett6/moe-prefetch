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
    MOE_PRED_BLOCK_PCA       = 0,
    MOE_PRED_BLOCK_PREV      = 1,   // experts layer L+1 used on the previous token
    MOE_PRED_BLOCK_CUR       = 2,   // experts layer L   used on this token
    MOE_PRED_BLOCK_BELOW     = 3,   // experts layer L-1 used on this token
    MOE_PRED_BLOCK_SELF_PREV = 4,   // experts layer L   used on the previous token
};

struct moe_predictor {
    uint32_t n_expert = 0;
    uint32_t n_layers = 0;
    uint32_t feat_dim = 0;
    uint32_t top_k    = 0;

    uint32_t d_model = 0;
    uint32_t n_comp  = 0;
    bool     has_pca = false;

    // where each feature block starts inside x, -1 when the model does not use it
    int32_t off_prev      = -1;
    int32_t off_cur       = -1;
    int32_t off_below     = -1;
    int32_t off_self_prev = -1;
    int32_t off_pca       = -1;

    std::vector<float> mean, comp, scale;          // PCA, empty unless has_pca

    std::vector<int32_t> layer_of;                 // present layers, ascending
    std::vector<int32_t> index_of;                 // layer -> slot in `w`, or -1
    std::vector<float>   prior;                    // per present layer
    std::vector<float>   bias;                     // [n_present][n_expert]
    std::vector<float>   w;                        // [n_present][feat_dim*n_expert]

    // Load a MOEP file. Returns false and fills `err` on any mismatch; a
    // predictor that half-loaded is never returned.
    bool load(const std::string & path, std::string & err);

    bool has_layer(int il) const {
        return il >= 0 && il < (int) index_of.size() && index_of[il] >= 0;
    }

    // Score layer `il` into `out` (n_expert floats). Any list may be null/empty;
    // a block the model does not declare is ignored even if ids are supplied.
    struct inputs {
        const int32_t * prev = nullptr;      size_t n_prev = 0;
        const int32_t * cur  = nullptr;      size_t n_cur = 0;
        const int32_t * below = nullptr;     size_t n_below = 0;
        const int32_t * self_prev = nullptr; size_t n_self_prev = 0;
    };
    void score(int il, const inputs & in, float * out) const;

    // two-block convenience overload, for v2 models
    void score(int il, const int32_t * prev, size_t n_prev,
               const int32_t * cur, size_t n_cur, float * out) const;

    // Top-k expert ids by score, best first. Returns how many were written.
    size_t top_k_ids(const float * scores, size_t k, int32_t * out) const;

    // ---- online learning -------------------------------------------------
    // One SGD step on the layer's weights from a prediction that has since been
    // resolved: `in` are the features the prediction was made from, `truth` the
    // experts the target layer actually routed to.
    //
    // The features are multi-hot with ~32 non-zeros out of 512, so the gradient
    // touches only those 32 rows: 32*128 updates per layer per token, the same
    // order as scoring. Logistic loss, because the target is a set membership
    // and squared error on an unbounded score has no reason to be calibrated.
    //
    // Returns the recall@8 of the prediction being learned from, so the caller
    // can watch whether online updates are helping without a second pass.
    float update(int il, const inputs & in, const int32_t * truth, size_t n_truth,
                 float lr);

    // Write the current weights back out in MOEP format, so a run's learning
    // survives it.
    bool save(const std::string & path, std::string & err) const;
};
