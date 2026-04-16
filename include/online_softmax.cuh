#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

// Online softmax state for streaming computation
struct OnlineSoftmaxState {
    float max_val;    // Current maximum value m_i
    float sum_exp;    // Current sum of exponentials l_i
    
    __device__ __forceinline__ OnlineSoftmaxState() 
        : max_val(-FLT_MAX), sum_exp(0.0f) {}
    
    __device__ __forceinline__ OnlineSoftmaxState(float m, float l) 
        : max_val(m), sum_exp(l) {}
};

// Update online softmax state with a new value
__device__ __forceinline__ void online_softmax_update(
    OnlineSoftmaxState& state,
    float new_val
) {
    float new_max = fmaxf(state.max_val, new_val);
    float old_scale = expf(state.max_val - new_max);
    float new_scale = expf(new_val - new_max);
    
    state.sum_exp = state.sum_exp * old_scale + new_scale;
    state.max_val = new_max;
}

// Update online softmax state with a block of values
__device__ __forceinline__ void online_softmax_update_block(
    OnlineSoftmaxState& state,
    const float* values,
    int count
) {
    // Find max in block
    float block_max = -FLT_MAX;
    for (int i = 0; i < count; i++) {
        block_max = fmaxf(block_max, values[i]);
    }
    
    // Update state
    float new_max = fmaxf(state.max_val, block_max);
    float old_scale = expf(state.max_val - new_max);
    
    // Compute sum of exp for new block
    float block_sum = 0.0f;
    for (int i = 0; i < count; i++) {
        block_sum += expf(values[i] - new_max);
    }
    
    state.sum_exp = state.sum_exp * old_scale + block_sum;
    state.max_val = new_max;
}

// Merge two online softmax states
__device__ __forceinline__ OnlineSoftmaxState online_softmax_merge(
    const OnlineSoftmaxState& a,
    const OnlineSoftmaxState& b
) {
    float new_max = fmaxf(a.max_val, b.max_val);
    float scale_a = expf(a.max_val - new_max);
    float scale_b = expf(b.max_val - new_max);
    float new_sum = a.sum_exp * scale_a + b.sum_exp * scale_b;
    
    return OnlineSoftmaxState(new_max, new_sum);
}

// Finalize online softmax - get normalization factor
__device__ __forceinline__ float online_softmax_finalize(
    const OnlineSoftmaxState& state
) {
    // Protect against divide by zero (sum_exp can be 0 for empty/fully-masked inputs)
    return state.sum_exp > 0.0f ? 1.0f / state.sum_exp : 0.0f;
}

// Compute softmax weight for a value given the final state
__device__ __forceinline__ float online_softmax_weight(
    float value,
    const OnlineSoftmaxState& state
) {
    // Protect against divide by zero
    float inv_sum = state.sum_exp > 0.0f ? 1.0f / state.sum_exp : 0.0f;
    return expf(value - state.max_val) * inv_sum;
}

// Update output accumulator with rescaling for online softmax
// O_new = (l_old * exp(m_old - m_new) * O_old + exp_weights @ V_block) / l_new
//
// output:      [dim]           — running output accumulator (updated in-place)
// v_block:     [block_n, dim]  — V values for current KV block
// exp_weights: [block_n]       — exp(score - new_max) for each KV position
// dim:         head dimension
// block_n:     number of KV positions in current block
__device__ __forceinline__ void online_softmax_update_output(
    float* output,
    const float* v_block,
    const float* exp_weights,
    int dim,
    int block_n,
    float old_max,
    float new_max,
    float old_sum,
    float new_sum
) {
    // Protect against divide by zero
    float rescale = new_sum > 0.0f ? expf(old_max - new_max) * old_sum / new_sum : 0.0f;
    float inv_new_sum = new_sum > 0.0f ? 1.0f / new_sum : 0.0f;

    for (int d = 0; d < dim; d++) {
        // Rescale old accumulator
        float val = output[d] * rescale;

        // Add new contribution: exp_weights @ V_block[:, d]
        float new_val = 0.0f;
        for (int j = 0; j < block_n; j++) {
            new_val += exp_weights[j] * v_block[j * dim + d];
        }
        output[d] = val + new_val * inv_new_sum;
    }
}

// FlashAttention-style online softmax update for output
// O_new = (l_old * exp(m_old - m_new) * O_old + exp(S - m_new) @ V) / l_new
template<int HEAD_DIM, int BLOCK_N>
__device__ __forceinline__ void flash_attention_update_output(
    float* __restrict__ output,           // [HEAD_DIM]
    const float* __restrict__ scores,     // [BLOCK_N] - attention scores for current K block
    const float* __restrict__ v_block,    // [BLOCK_N, HEAD_DIM] - V values for current block
    int block_n,                          // Actual size of current block (<= BLOCK_N)
    OnlineSoftmaxState& state             // Online softmax state
) {
    // Find max in current scores
    float block_max = -FLT_MAX;
    for (int j = 0; j < block_n; j++) {
        block_max = fmaxf(block_max, scores[j]);
    }

    // Compute new max and scaling factors
    float new_max = fmaxf(state.max_val, block_max);
    float old_scale = expf(state.max_val - new_max);

    // Compute exp(scores - new_max) and their sum
    float exp_scores[BLOCK_N];
    float block_sum = 0.0f;
    for (int j = 0; j < block_n; j++) {
        exp_scores[j] = expf(scores[j] - new_max);
        block_sum += exp_scores[j];
    }

    // New sum
    float new_sum = state.sum_exp * old_scale + block_sum;

    // Rescale old output and add new contribution (protect against divide by zero)
    float rescale_old = new_sum > 0.0f ? old_scale * state.sum_exp / new_sum : 0.0f;
    float scale_new = new_sum > 0.0f ? 1.0f / new_sum : 0.0f;

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        float new_val = 0.0f;
        for (int j = 0; j < block_n; j++) {
            new_val += exp_scores[j] * v_block[j * HEAD_DIM + d];
        }
        output[d] = output[d] * rescale_old + new_val * scale_new;
    }

    // Update state
    state.max_val = new_max;
    state.sum_exp = new_sum;
}
