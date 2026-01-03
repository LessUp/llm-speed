#include "common.cuh"
#include "shared_memory.cuh"
#include "warp_primitives.cuh"
#include <cfloat>

// Tiled attention kernel using shared memory
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void tiled_attention_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // Block indices
    int batch_head_idx = blockIdx.z;
    int batch_idx = batch_head_idx / num_heads;
    int head_idx = batch_head_idx % num_heads;
    int block_row = blockIdx.y;
    
    if (batch_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    int row_start = block_row * BLOCK_M;
    
    // Shared memory with padding to avoid bank conflicts
    __shared__ float smem_Q[BLOCK_M][BLOCK_K + 1];
    __shared__ float smem_K[BLOCK_N][BLOCK_K + 1];
    __shared__ float smem_V[BLOCK_N][BLOCK_K + 1];
    __shared__ float smem_S[BLOCK_M][BLOCK_N + 1];  // Attention scores
    __shared__ float smem_max[BLOCK_M];
    __shared__ float smem_sum[BLOCK_M];
    
    // Output accumulator in registers
    float output[BLOCK_M][BLOCK_K];
    float row_max[BLOCK_M];
    float row_sum[BLOCK_M];
    
    // Initialize
    #pragma unroll
    for (int m = 0; m < BLOCK_M; m++) {
        row_max[m] = -FLT_MAX;
        row_sum[m] = 0.0f;
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            output[m][k] = 0.0f;
        }
    }
    
    // Pointer offsets
    int offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const T* q_ptr = Q + offset;
    const T* k_ptr = K + offset;
    const T* v_ptr = V + offset;
    T* o_ptr = O + offset;
    
    // Load Q tile to shared memory
    for (int i = tid; i < BLOCK_M * head_dim; i += blockDim.x) {
        int m = i / head_dim;
        int k = i % head_dim;
        int global_row = row_start + m;
        if (global_row < seq_len && k < head_dim) {
            smem_Q[m][k] = static_cast<float>(q_ptr[global_row * head_dim + k]);
        } else {
            smem_Q[m][k] = 0.0f;
        }
    }
    __syncthreads();
    
    // Iterate over K/V blocks
    int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int col_start = kv_block * BLOCK_N;
        
        // Load K tile
        for (int i = tid; i < BLOCK_N * head_dim; i += blockDim.x) {
            int n = i / head_dim;
            int k = i % head_dim;
            int global_col = col_start + n;
            if (global_col < seq_len && k < head_dim) {
                smem_K[n][k] = static_cast<float>(k_ptr[global_col * head_dim + k]);
            } else {
                smem_K[n][k] = 0.0f;
            }
        }
        
        // Load V tile
        for (int i = tid; i < BLOCK_N * head_dim; i += blockDim.x) {
            int n = i / head_dim;
            int k = i % head_dim;
            int global_col = col_start + n;
            if (global_col < seq_len && k < head_dim) {
                smem_V[n][k] = static_cast<float>(v_ptr[global_col * head_dim + k]);
            } else {
                smem_V[n][k] = 0.0f;
            }
        }
        __syncthreads();
        
        // Compute attention scores: Q @ K^T
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += blockDim.x) {
            int m = i / BLOCK_N;
            int n = i % BLOCK_N;
            
            float score = 0.0f;
            #pragma unroll
            for (int k = 0; k < BLOCK_K && k < head_dim; k++) {
                score += smem_Q[m][k] * smem_K[n][k];
            }
            
            // Handle remaining dimensions if head_dim > BLOCK_K
            for (int k = BLOCK_K; k < head_dim; k++) {
                score += smem_Q[m][k] * smem_K[n][k];
            }
            
            smem_S[m][n] = score * scale;
        }
        __syncthreads();
        
        // Online softmax update for each row
        for (int m = tid; m < BLOCK_M; m += blockDim.x) {
            int global_row = row_start + m;
            if (global_row >= seq_len) continue;
            
            // Find max in this block
            float block_max = -FLT_MAX;
            for (int n = 0; n < BLOCK_N; n++) {
                int global_col = col_start + n;
                if (global_col < seq_len) {
                    block_max = fmaxf(block_max, smem_S[m][n]);
                }
            }
            
            // Update running max
            float new_max = fmaxf(row_max[m], block_max);
            float old_scale = expf(row_max[m] - new_max);
            
            // Compute exp and sum for this block
            float block_sum = 0.0f;
            for (int n = 0; n < BLOCK_N; n++) {
                int global_col = col_start + n;
                if (global_col < seq_len) {
                    smem_S[m][n] = expf(smem_S[m][n] - new_max);
                    block_sum += smem_S[m][n];
                } else {
                    smem_S[m][n] = 0.0f;
                }
            }
            
            // Update running sum
            float new_sum = row_sum[m] * old_scale + block_sum;
            
            // Rescale old output and add new contribution
            float rescale = old_scale * row_sum[m] / fmaxf(new_sum, 1e-6f);
            
            for (int k = 0; k < head_dim; k++) {
                output[m][k] *= rescale;
                
                float new_val = 0.0f;
                for (int n = 0; n < BLOCK_N; n++) {
                    new_val += smem_S[m][n] * smem_V[n][k];
                }
                output[m][k] += new_val / fmaxf(new_sum, 1e-6f);
            }
            
            row_max[m] = new_max;
            row_sum[m] = new_sum;
        }
        __syncthreads();
    }
    
    // Write output
    for (int i = tid; i < BLOCK_M * head_dim; i += blockDim.x) {
        int m = i / head_dim;
        int k = i % head_dim;
        int global_row = row_start + m;
        if (global_row < seq_len && k < head_dim) {
            o_ptr[global_row * head_dim + k] = static_cast<T>(output[m][k]);
        }
    }
}

// Host wrapper
void tiled_attention_fp32(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, cudaStream_t stream
) {
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    constexpr int BLOCK_K = 64;
    
    dim3 grid(1, (seq_len + BLOCK_M - 1) / BLOCK_M, batch_size * num_heads);
    dim3 block(256);
    
    tiled_attention_kernel<float, BLOCK_M, BLOCK_N, BLOCK_K><<<grid, block, 0, stream>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, head_dim, scale
    );
    CUDA_CHECK(cudaGetLastError());
}

void tiled_attention_fp16(
    const half* Q, const half* K, const half* V, half* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, cudaStream_t stream
) {
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    constexpr int BLOCK_K = 64;
    
    dim3 grid(1, (seq_len + BLOCK_M - 1) / BLOCK_M, batch_size * num_heads);
    dim3 block(256);
    
    tiled_attention_kernel<half, BLOCK_M, BLOCK_N, BLOCK_K><<<grid, block, 0, stream>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, head_dim, scale
    );
    CUDA_CHECK(cudaGetLastError());
}
