#include "common.cuh"
#include <cmath>
#include <cfloat>

// Naive softmax kernel for a single row
template<typename T>
__device__ void softmax_row(T* row, int len) {
    // Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int i = 0; i < len; i++) {
        max_val = fmaxf(max_val, static_cast<float>(row[i]));
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        float val = expf(static_cast<float>(row[i]) - max_val);
        row[i] = static_cast<T>(val);
        sum += val;
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < len; i++) {
        row[i] = static_cast<T>(static_cast<float>(row[i]) * inv_sum);
    }
}

// Naive attention kernel - one thread per output element
template<typename T>
__global__ void naive_attention_kernel(
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
    // Global indices
    int batch_head_idx = blockIdx.z;
    int batch_idx = batch_head_idx / num_heads;
    int head_idx = batch_head_idx % num_heads;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || row_idx >= seq_len || col_idx >= head_dim) return;
    
    // Pointer offsets for this batch and head
    int offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const T* q_ptr = Q + offset;
    const T* k_ptr = K + offset;
    const T* v_ptr = V + offset;
    T* o_ptr = O + offset;
    
    // Compute attention scores for this row
    extern __shared__ float smem[];
    float* scores = smem + threadIdx.y * seq_len;
    
    // Only first thread in each row computes scores
    if (col_idx == 0) {
        // Compute Q[row] @ K^T -> scores[seq_len]
        for (int j = 0; j < seq_len; j++) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += static_cast<float>(q_ptr[row_idx * head_dim + d]) *
                         static_cast<float>(k_ptr[j * head_dim + d]);
            }
            scores[j] = score * scale;
        }
        
        // Softmax
        float max_val = -FLT_MAX;
        for (int j = 0; j < seq_len; j++) {
            max_val = fmaxf(max_val, scores[j]);
        }
        
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[j] = expf(scores[j] - max_val);
            sum += scores[j];
        }
        
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < seq_len; j++) {
            scores[j] *= inv_sum;
        }
    }
    
    __syncthreads();
    
    // Compute output: scores @ V
    float out_val = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        out_val += scores[j] * static_cast<float>(v_ptr[j * head_dim + col_idx]);
    }
    
    o_ptr[row_idx * head_dim + col_idx] = static_cast<T>(out_val);
}

// Simpler naive attention - each block handles one (batch, head, row)
template<typename T>
__global__ void naive_attention_simple_kernel(
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
    int batch_idx = blockIdx.z / num_heads;
    int head_idx = blockIdx.z % num_heads;
    int row_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || row_idx >= seq_len) return;
    
    int offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const T* q_ptr = Q + offset + row_idx * head_dim;
    const T* k_ptr = K + offset;
    const T* v_ptr = V + offset;
    T* o_ptr = O + offset + row_idx * head_dim;
    
    extern __shared__ float shared_mem[];
    float* scores = shared_mem;
    
    // Compute attention scores: Q[row] @ K^T
    for (int j = tid; j < seq_len; j += blockDim.x) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += static_cast<float>(q_ptr[d]) * static_cast<float>(k_ptr[j * head_dim + d]);
        }
        scores[j] = score * scale;
    }
    __syncthreads();
    
    // Softmax - find max (reduction)
    float local_max = -FLT_MAX;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        local_max = fmaxf(local_max, scores[j]);
    }
    
    // Warp reduction for max
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    __shared__ float block_max;
    if (tid == 0) block_max = local_max;
    __syncthreads();
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        scores[j] = expf(scores[j] - block_max);
        local_sum += scores[j];
    }
    
    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    __shared__ float block_sum;
    if (tid == 0) block_sum = local_sum;
    __syncthreads();
    
    // Normalize
    float inv_sum = 1.0f / block_sum;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        scores[j] *= inv_sum;
    }
    __syncthreads();
    
    // Compute output: scores @ V
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            out_val += scores[j] * static_cast<float>(v_ptr[j * head_dim + d]);
        }
        o_ptr[d] = static_cast<T>(out_val);
    }
}

// Host wrapper functions
void naive_attention_fp32(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, cudaStream_t stream
) {
    dim3 grid(1, seq_len, batch_size * num_heads);
    dim3 block(256);
    size_t smem_size = seq_len * sizeof(float);
    
    naive_attention_simple_kernel<float><<<grid, block, smem_size, stream>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, head_dim, scale
    );
    CUDA_CHECK(cudaGetLastError());
}

void naive_attention_fp16(
    const half* Q, const half* K, const half* V, half* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, cudaStream_t stream
) {
    dim3 grid(1, seq_len, batch_size * num_heads);
    dim3 block(256);
    size_t smem_size = seq_len * sizeof(float);
    
    naive_attention_simple_kernel<half><<<grid, block, smem_size, stream>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, head_dim, scale
    );
    CUDA_CHECK(cudaGetLastError());
}
