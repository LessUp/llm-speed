#include "common.cuh"
#include "shared_memory.cuh"
#include "warp_primitives.cuh"
#include "online_softmax.cuh"
#include "pipeline.cuh"
#include <cfloat>

// FlashAttention forward kernel with online softmax and double buffering
// Optimized: all threads cooperate on the expensive output update phase
// Double buffering: K/V tiles use two buffers for compute/load overlap
//
// Shared memory layout (with +1 padding to avoid bank conflicts):
//   smem_Q    [BLOCK_M * (head_dim+1)]
//   smem_K[2] [2 * BLOCK_N * (head_dim+1)]   — double buffer for K tiles
//   smem_V[2] [2 * BLOCK_N * (head_dim+1)]   — double buffer for V tiles
//   smem_S    [BLOCK_M * (BLOCK_N+1)]         — attention scores, then exp(scores)
//   output    [BLOCK_M * head_dim]            — output accumulator
//   row_max   [BLOCK_M]                       — per-row running max
//   row_sum   [BLOCK_M]                       — per-row running sum
//   rescale   [BLOCK_M]                       — per-row rescale factor
template<typename T, int BLOCK_M, int BLOCK_N>
__global__ void flash_attention_forward_kernel(
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
    int batch_head_idx = blockIdx.z;
    int batch_idx = batch_head_idx / num_heads;
    int head_idx = batch_head_idx % num_heads;
    int block_row = blockIdx.y;
    
    if (batch_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    int row_start = block_row * BLOCK_M;
    
    // Shared memory layout with padding and double buffering for K/V
    extern __shared__ float smem[];
    int hd_stride = head_dim + 1;  // padded stride for bank conflict avoidance
    int sn_stride = BLOCK_N + 1;
    int kv_buf_size = BLOCK_N * hd_stride;  // size of one K or V buffer
    
    float* smem_Q     = smem;
    float* smem_K_buf = smem_Q + BLOCK_M * hd_stride;            // K double buffer [2 * kv_buf_size]
    float* smem_V_buf = smem_K_buf + 2 * kv_buf_size;            // V double buffer [2 * kv_buf_size]
    float* smem_S     = smem_V_buf + 2 * kv_buf_size;
    float* output     = smem_S + BLOCK_M * sn_stride;
    float* row_max    = output + BLOCK_M * head_dim;
    float* row_sum    = row_max + BLOCK_M;
    float* rescale    = row_sum + BLOCK_M;
    
    // Double buffer accessors
    #define SMEM_K(buf) (smem_K_buf + (buf) * kv_buf_size)
    #define SMEM_V(buf) (smem_V_buf + (buf) * kv_buf_size)
    
    // Pointer offsets (use int64 to avoid overflow for large tensors)
    int64_t offset = (static_cast<int64_t>(batch_idx) * num_heads + head_idx) * seq_len * head_dim;
    const T* q_ptr = Q + offset;
    const T* k_ptr = K + offset;
    const T* v_ptr = V + offset;
    T* o_ptr = O + offset;
    
    // Initialize output accumulator and per-row state (cooperative)
    for (int i = tid; i < BLOCK_M * head_dim; i += blockDim.x) {
        output[i] = 0.0f;
    }
    if (tid < BLOCK_M) {
        row_max[tid] = -FLT_MAX;
        row_sum[tid] = 0.0f;
    }
    __syncthreads();
    
    // Load Q tile to shared memory
    for (int i = tid; i < BLOCK_M * head_dim; i += blockDim.x) {
        int m = i / head_dim;
        int d = i % head_dim;
        int global_row = row_start + m;
        smem_Q[m * hd_stride + d] = (global_row < seq_len)
            ? static_cast<float>(q_ptr[global_row * head_dim + d]) : 0.0f;
    }
    __syncthreads();
    
    int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    
    // Helper: load K/V tile into the specified buffer
    #define LOAD_KV_TILE(buf_idx, col_start_val) \
        for (int i = tid; i < BLOCK_N * head_dim; i += blockDim.x) { \
            int n = i / head_dim; \
            int d = i % head_dim; \
            int global_col = (col_start_val) + n; \
            float kval = (global_col < seq_len) \
                ? static_cast<float>(k_ptr[global_col * head_dim + d]) : 0.0f; \
            float vval = (global_col < seq_len) \
                ? static_cast<float>(v_ptr[global_col * head_dim + d]) : 0.0f; \
            SMEM_K(buf_idx)[n * hd_stride + d] = kval; \
            SMEM_V(buf_idx)[n * hd_stride + d] = vval; \
        }
    
    // Prologue: load first KV tile into buffer 0
    if (num_kv_blocks > 0) {
        int col_start_0 = 0;
        LOAD_KV_TILE(0, col_start_0);
    }
    __syncthreads();
    
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int col_start = kv_block * BLOCK_N;
        int cur_buf = kv_block % 2;
        int next_buf = 1 - cur_buf;
        
        // Early exit for causal: max query pos is row_start + BLOCK_M - 1
        if (is_causal && col_start >= row_start + BLOCK_M) break;
        
        // Prefetch next KV tile into alternate buffer (if available)
        int next_kv_block = kv_block + 1;
        bool has_next = (next_kv_block < num_kv_blocks);
        if (is_causal && has_next) {
            int next_col_start = next_kv_block * BLOCK_N;
            if (next_col_start >= row_start + BLOCK_M) has_next = false;
        }
        if (has_next) {
            int next_col_start = next_kv_block * BLOCK_N;
            LOAD_KV_TILE(next_buf, next_col_start);
        }
        
        // Compute attention scores: Q @ K^T * scale using current buffer
        float* cur_K = SMEM_K(cur_buf);
        float* cur_V = SMEM_V(cur_buf);
        
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += blockDim.x) {
            int m = i / BLOCK_N;
            int n = i % BLOCK_N;
            int global_row = row_start + m;
            int global_col = col_start + n;
            
            float score = 0.0f;
            if (global_row < seq_len && global_col < seq_len) {
                for (int d = 0; d < head_dim; d++) {
                    score += smem_Q[m * hd_stride + d] * cur_K[n * hd_stride + d];
                }
                score *= scale;
                if (is_causal && global_col > global_row) {
                    score = -FLT_MAX;
                }
            } else {
                score = -FLT_MAX;
            }
            smem_S[m * sn_stride + n] = score;
        }
        __syncthreads();
        
        // Phase 1: Online softmax - compute exp(scores) and rescale factors
        // One thread per row (cheap: only BLOCK_N iterations per row)
        if (tid < BLOCK_M) {
            int m = tid;
            int global_row = row_start + m;
            
            if (global_row < seq_len) {
                // Find block max
                float block_max = -FLT_MAX;
                for (int n = 0; n < BLOCK_N; n++) {
                    block_max = fmaxf(block_max, smem_S[m * sn_stride + n]);
                }
                
                float old_max = row_max[m];
                float old_sum = row_sum[m];
                float new_max = fmaxf(old_max, block_max);
                float old_scale_f = expf(old_max - new_max);
                
                // Compute exp(scores - new_max) in-place and block sum
                float block_sum = 0.0f;
                for (int n = 0; n < BLOCK_N; n++) {
                    float exp_val = expf(smem_S[m * sn_stride + n] - new_max);
                    smem_S[m * sn_stride + n] = exp_val;
                    block_sum += exp_val;
                }
                
                float new_sum = old_sum * old_scale_f + block_sum;
                
                // Store rescale factor for Phase 2
                rescale[m] = (old_sum > 0.0f && new_sum > 0.0f) ? (old_scale_f * old_sum / new_sum) : 0.0f;
                
                row_max[m] = new_max;
                row_sum[m] = new_sum;
            } else {
                rescale[m] = 0.0f;
            }
        }
        __syncthreads();
        
        // Phase 2: Output update - ALL threads cooperate (the expensive part)
        // output[m,d] = output[m,d] * rescale[m] + (1/new_sum) * sum_n(exp_S[m,n] * V[n,d])
        for (int i = tid; i < BLOCK_M * head_dim; i += blockDim.x) {
            int m = i / head_dim;
            int d = i % head_dim;
            int global_row = row_start + m;

            if (global_row < seq_len) {
                // Rescale old output
                float old_val = output[i] * rescale[m];

                // Compute new contribution: exp_scores[m,:] @ V[:,d]
                float new_val = 0.0f;
                for (int n = 0; n < BLOCK_N; n++) {
                    new_val += smem_S[m * sn_stride + n] * cur_V[n * hd_stride + d];
                }

                // Protect against divide by zero (row_sum can be 0 for masked rows)
                float inv_sum = row_sum[m] > 0.0f ? 1.0f / row_sum[m] : 0.0f;
                output[i] = old_val + new_val * inv_sum;
            }
        }
        __syncthreads();
    }
    
    #undef SMEM_K
    #undef SMEM_V
    #undef LOAD_KV_TILE
    
    // Write output to global memory
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

    int hd_stride = head_dim + 1;
    int sn_stride = BLOCK_N + 1;
    size_t smem_size = (BLOCK_M * hd_stride            // smem_Q
                       + 4 * BLOCK_N * hd_stride        // smem_K[2] + smem_V[2] (double buffered)
                       + BLOCK_M * sn_stride             // smem_S
                       + BLOCK_M * head_dim              // output
                       + 3 * BLOCK_M                     // row_max + row_sum + rescale
                       ) * sizeof(float);

    // Validate shared memory requirement against device limit
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int max_smem;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem,
        cudaDevAttrMaxSharedMemoryPerBlock, device));
    if (static_cast<int>(smem_size) > max_smem) {
        throw std::runtime_error(
            "flash_attention: head_dim=" + std::to_string(head_dim) +
            " requires " + std::to_string(smem_size) +
            " bytes shared memory, but device max is " +
            std::to_string(max_smem) + " bytes.");
    }

    flash_attention_forward_kernel<float, BLOCK_M, BLOCK_N><<<grid, block, smem_size, stream>>>(
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

    int hd_stride = head_dim + 1;
    int sn_stride = BLOCK_N + 1;
    size_t smem_size = (BLOCK_M * hd_stride
                       + 4 * BLOCK_N * hd_stride        // double buffered K/V
                       + BLOCK_M * sn_stride
                       + BLOCK_M * head_dim
                       + 3 * BLOCK_M
                       ) * sizeof(float);

    // Validate shared memory requirement against device limit
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int max_smem;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem,
        cudaDevAttrMaxSharedMemoryPerBlock, device));
    if (static_cast<int>(smem_size) > max_smem) {
        throw std::runtime_error(
            "flash_attention: head_dim=" + std::to_string(head_dim) +
            " requires " + std::to_string(smem_size) +
            " bytes shared memory, but device max is " +
            std::to_string(max_smem) + " bytes.");
    }

    flash_attention_forward_kernel<half, BLOCK_M, BLOCK_N><<<grid, block, smem_size, stream>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, head_dim, scale, is_causal
    );
    CUDA_CHECK(cudaGetLastError());
}
