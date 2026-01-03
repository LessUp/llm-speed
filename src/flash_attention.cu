#include "common.cuh"
#include "shared_memory.cuh"
#include "warp_primitives.cuh"
#include "online_softmax.cuh"
#include <cfloat>

// FlashAttention forward kernel
// Implements online softmax to avoid storing N×N attention matrix
template<typename T, int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void flash_attention_forward_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    float* __restrict__ L,  // logsumexp for backward (optional)
    int batch_size,
    int num_heads,
    int seq_len,
    float scale,
    bool is_causal
) {
    // Block handles BLOCK_M rows of Q
    int batch_head_idx = blockIdx.z;
    int batch_idx = batch_head_idx / num_heads;
    int head_idx = batch_head_idx % num_heads;
    int block_row = blockIdx.y;
    
    if (batch_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    int row_start = block_row * BLOCK_M;
    
    // Shared memory
    extern __shared__ float smem[];
    float* smem_Q = smem;                                    // [BLOCK_M, HEAD_DIM]
    float* smem_K = smem_Q + BLOCK_M * HEAD_DIM;            // [BLOCK_N, HEAD_DIM]
    float* smem_V = smem_K + BLOCK_N * HEAD_DIM;            // [BLOCK_N, HEAD_DIM]
    float* smem_S = smem_V + BLOCK_N * HEAD_DIM;            // [BLOCK_M, BLOCK_N]
    
    // Per-row state in registers
    float row_max[BLOCK_M];
    float row_sum[BLOCK_M];
    float output[BLOCK_M * HEAD_DIM];
    
    // Initialize
    #pragma unroll
    for (int m = 0; m < BLOCK_M; m++) {
        row_max[m] = -FLT_MAX;
        row_sum[m] = 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < BLOCK_M * HEAD_DIM; i++) {
        output[i] = 0.0f;
    }
    
    // Pointer offsets
    int offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM;
    const T* q_ptr = Q + offset;
    const T* k_ptr = K + offset;
    const T* v_ptr = V + offset;
    T* o_ptr = O + offset;
    
    // Load Q tile to shared memory
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
        int m = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        int global_row = row_start + m;
        if (global_row < seq_len) {
            smem_Q[m * HEAD_DIM + d] = static_cast<float>(q_ptr[global_row * HEAD_DIM + d]);
        } else {
            smem_Q[m * HEAD_DIM + d] = 0.0f;
        }
    }
    __syncthreads();
    
    // Determine K/V block range for causal masking
    int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    int max_kv_block = num_kv_blocks;
    
    if (is_causal) {
        // For causal, only attend to positions <= current position
        max_kv_block = (row_start + BLOCK_M + BLOCK_N - 1) / BLOCK_N;
        max_kv_block = min(max_kv_block, num_kv_blocks);
    }
    
    // Iterate over K/V blocks
    for (int kv_block = 0; kv_block < max_kv_block; kv_block++) {
        int col_start = kv_block * BLOCK_N;
        
        // Load K tile
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
            int n = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            int global_col = col_start + n;
            if (global_col < seq_len) {
                smem_K[n * HEAD_DIM + d] = static_cast<float>(k_ptr[global_col * HEAD_DIM + d]);
            } else {
                smem_K[n * HEAD_DIM + d] = 0.0f;
            }
        }
        
        // Load V tile
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
            int n = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            int global_col = col_start + n;
            if (global_col < seq_len) {
                smem_V[n * HEAD_DIM + d] = static_cast<float>(v_ptr[global_col * HEAD_DIM + d]);
            } else {
                smem_V[n * HEAD_DIM + d] = 0.0f;
            }
        }
        __syncthreads();
        
        // Compute attention scores: Q @ K^T * scale
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += blockDim.x) {
            int m = i / BLOCK_N;
            int n = i % BLOCK_N;
            int global_row = row_start + m;
            int global_col = col_start + n;
            
            float score = 0.0f;
            if (global_row < seq_len && global_col < seq_len) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) {
                    score += smem_Q[m * HEAD_DIM + d] * smem_K[n * HEAD_DIM + d];
                }
                score *= scale;
                
                // Apply causal mask
                if (is_causal && global_col > global_row) {
                    score = -FLT_MAX;
                }
            } else {
                score = -FLT_MAX;
            }
            
            smem_S[m * BLOCK_N + n] = score;
        }
        __syncthreads();
        
        // Online softmax update for each row handled by this thread
        for (int m = 0; m < BLOCK_M; m++) {
            int global_row = row_start + m;
            if (global_row >= seq_len) continue;
            
            // Find max in this block
            float block_max = -FLT_MAX;
            for (int n = 0; n < BLOCK_N; n++) {
                block_max = fmaxf(block_max, smem_S[m * BLOCK_N + n]);
            }
            
            // Update running max
            float new_max = fmaxf(row_max[m], block_max);
            float old_scale = expf(row_max[m] - new_max);
            
            // Compute exp(scores - new_max) and sum
            float exp_scores[BLOCK_N];
            float block_sum = 0.0f;
            for (int n = 0; n < BLOCK_N; n++) {
                exp_scores[n] = expf(smem_S[m * BLOCK_N + n] - new_max);
                block_sum += exp_scores[n];
            }
            
            // Update running sum
            float new_sum = row_sum[m] * old_scale + block_sum;
            
            // Rescale old output
            float rescale = (row_sum[m] > 0.0f) ? (old_scale * row_sum[m] / new_sum) : 0.0f;
            
            // Update output: O = rescale * O_old + (1/new_sum) * exp_scores @ V
            for (int d = 0; d < HEAD_DIM; d++) {
                float new_val = 0.0f;
                for (int n = 0; n < BLOCK_N; n++) {
                    new_val += exp_scores[n] * smem_V[n * HEAD_DIM + d];
                }
                output[m * HEAD_DIM + d] = output[m * HEAD_DIM + d] * rescale + new_val / new_sum;
            }
            
            row_max[m] = new_max;
            row_sum[m] = new_sum;
        }
        __syncthreads();
    }
    
    // Write output
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
        int m = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        int global_row = row_start + m;
        if (global_row < seq_len) {
            o_ptr[global_row * HEAD_DIM + d] = static_cast<T>(output[m * HEAD_DIM + d]);
        }
    }
    
    // Write logsumexp for backward pass (optional)
    if (L != nullptr) {
        float* l_ptr = L + (batch_idx * num_heads + head_idx) * seq_len;
        for (int m = tid; m < BLOCK_M; m += blockDim.x) {
            int global_row = row_start + m;
            if (global_row < seq_len) {
                l_ptr[global_row] = row_max[m] + logf(row_sum[m]);
            }
        }
    }
}

// Simplified FlashAttention for variable head dimensions
template<typename T>
__global__ void flash_attention_forward_dynamic_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool is_causal
) {
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    
    int batch_head_idx = blockIdx.z;
    int batch_idx = batch_head_idx / num_heads;
    int head_idx = batch_head_idx % num_heads;
    int block_row = blockIdx.y;
    
    if (batch_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    int row_start = block_row * BLOCK_M;
    
    // Dynamic shared memory
    extern __shared__ float smem[];
    float* smem_Q = smem;
    float* smem_K = smem_Q + BLOCK_M * head_dim;
    float* smem_V = smem_K + BLOCK_N * head_dim;
    float* smem_S = smem_V + BLOCK_N * head_dim;
    
    // Pointer offsets
    int offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const T* q_ptr = Q + offset;
    const T* k_ptr = K + offset;
    const T* v_ptr = V + offset;
    T* o_ptr = O + offset;
    
    // Per-row state
    float row_max[BLOCK_M];
    float row_sum[BLOCK_M];
    
    for (int m = 0; m < BLOCK_M; m++) {
        row_max[m] = -FLT_MAX;
        row_sum[m] = 0.0f;
    }
    
    // Output accumulator (in shared memory for large head_dim)
    float* output = smem_S + BLOCK_M * BLOCK_N;  // Reuse space after scores
    for (int i = tid; i < BLOCK_M * head_dim; i += blockDim.x) {
        output[i] = 0.0f;
    }
    __syncthreads();
    
    // Load Q tile
    for (int i = tid; i < BLOCK_M * head_dim; i += blockDim.x) {
        int m = i / head_dim;
        int d = i % head_dim;
        int global_row = row_start + m;
        smem_Q[i] = (global_row < seq_len) ? static_cast<float>(q_ptr[global_row * head_dim + d]) : 0.0f;
    }
    __syncthreads();
    
    int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int col_start = kv_block * BLOCK_N;
        
        // Early exit for causal
        if (is_causal && col_start > row_start + BLOCK_M) break;
        
        // Load K, V tiles
        for (int i = tid; i < BLOCK_N * head_dim; i += blockDim.x) {
            int n = i / head_dim;
            int d = i % head_dim;
            int global_col = col_start + n;
            smem_K[i] = (global_col < seq_len) ? static_cast<float>(k_ptr[global_col * head_dim + d]) : 0.0f;
            smem_V[i] = (global_col < seq_len) ? static_cast<float>(v_ptr[global_col * head_dim + d]) : 0.0f;
        }
        __syncthreads();
        
        // Compute scores
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += blockDim.x) {
            int m = i / BLOCK_N;
            int n = i % BLOCK_N;
            int global_row = row_start + m;
            int global_col = col_start + n;
            
            float score = 0.0f;
            if (global_row < seq_len && global_col < seq_len) {
                for (int d = 0; d < head_dim; d++) {
                    score += smem_Q[m * head_dim + d] * smem_K[n * head_dim + d];
                }
                score *= scale;
                if (is_causal && global_col > global_row) {
                    score = -FLT_MAX;
                }
            } else {
                score = -FLT_MAX;
            }
            smem_S[i] = score;
        }
        __syncthreads();
        
        // Online softmax update (single thread per row for simplicity)
        if (tid < BLOCK_M) {
            int m = tid;
            int global_row = row_start + m;
            if (global_row < seq_len) {
                float block_max = -FLT_MAX;
                for (int n = 0; n < BLOCK_N; n++) {
                    block_max = fmaxf(block_max, smem_S[m * BLOCK_N + n]);
                }
                
                float new_max = fmaxf(row_max[m], block_max);
                float old_scale = expf(row_max[m] - new_max);
                
                float block_sum = 0.0f;
                for (int n = 0; n < BLOCK_N; n++) {
                    smem_S[m * BLOCK_N + n] = expf(smem_S[m * BLOCK_N + n] - new_max);
                    block_sum += smem_S[m * BLOCK_N + n];
                }
                
                float new_sum = row_sum[m] * old_scale + block_sum;
                float rescale = (row_sum[m] > 0.0f) ? (old_scale * row_sum[m] / new_sum) : 0.0f;
                
                for (int d = 0; d < head_dim; d++) {
                    float new_val = 0.0f;
                    for (int n = 0; n < BLOCK_N; n++) {
                        new_val += smem_S[m * BLOCK_N + n] * smem_V[n * head_dim + d];
                    }
                    output[m * head_dim + d] = output[m * head_dim + d] * rescale + new_val / new_sum;
                }
                
                row_max[m] = new_max;
                row_sum[m] = new_sum;
            }
        }
        __syncthreads();
    }
    
    // Write output
    for (int i = tid; i < BLOCK_M * head_dim; i += blockDim.x) {
        int m = i / head_dim;
        int d = i % head_dim;
        int global_row = row_start + m;
        if (global_row < seq_len) {
            o_ptr[global_row * head_dim + d] = static_cast<T>(output[i]);
        }
    }
}

// Host wrappers
void flash_attention_fp32(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool is_causal, cudaStream_t stream
) {
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    
    dim3 grid(1, (seq_len + BLOCK_M - 1) / BLOCK_M, batch_size * num_heads);
    dim3 block(128);
    
    size_t smem_size = (BLOCK_M * head_dim + 2 * BLOCK_N * head_dim + BLOCK_M * BLOCK_N + BLOCK_M * head_dim) * sizeof(float);
    
    flash_attention_forward_dynamic_kernel<float><<<grid, block, smem_size, stream>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, head_dim, scale, is_causal
    );
    CUDA_CHECK(cudaGetLastError());
}

void flash_attention_fp16(
    const half* Q, const half* K, const half* V, half* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool is_causal, cudaStream_t stream
) {
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    
    dim3 grid(1, (seq_len + BLOCK_M - 1) / BLOCK_M, batch_size * num_heads);
    dim3 block(128);
    
    size_t smem_size = (BLOCK_M * head_dim + 2 * BLOCK_N * head_dim + BLOCK_M * BLOCK_N + BLOCK_M * head_dim) * sizeof(float);
    
    flash_attention_forward_dynamic_kernel<half><<<grid, block, smem_size, stream>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, head_dim, scale, is_causal
    );
    CUDA_CHECK(cudaGetLastError());
}
