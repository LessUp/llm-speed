#include "common.cuh"
#include "warp_primitives.cuh"
#include <cmath>
#include <cfloat>

// Naive attention - each block handles one (batch, head, row)
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
    float scale,
    bool is_causal
) {
    int batch_idx = blockIdx.z / num_heads;
    int head_idx = blockIdx.z % num_heads;
    int row_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size || row_idx >= seq_len) return;

    // Pointer offsets (use int64 to avoid overflow for large tensors)
    int64_t offset = (static_cast<int64_t>(batch_idx) * num_heads + head_idx) * seq_len * head_dim;
    const T* q_ptr = Q + offset + row_idx * head_dim;
    const T* k_ptr = K + offset;
    const T* v_ptr = V + offset;
    T* o_ptr = O + offset + row_idx * head_dim;

    constexpr int BLOCK_THREADS = 256;
    extern __shared__ float shared_mem[];
    float* scores = shared_mem;
    float* reduce_smem = shared_mem + seq_len;

    // Compute attention scores: Q[row] @ K^T
    for (int j = tid; j < seq_len; j += blockDim.x) {
        float score = 0.0f;
        // Apply causal mask: set score to -inf if j > row_idx
        if (is_causal && j > row_idx) {
            score = -FLT_MAX;
        } else {
            for (int d = 0; d < head_dim; d++) {
                score += static_cast<float>(q_ptr[d]) * static_cast<float>(k_ptr[j * head_dim + d]);
            }
            score *= scale;
        }
        scores[j] = score;
    }
    __syncthreads();

    // Softmax - find max (block reduction)
    float local_max = -FLT_MAX;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        local_max = fmaxf(local_max, scores[j]);
    }

    __shared__ float block_max;
    float reduced_max = block_reduce_max<float, BLOCK_THREADS>(local_max, reduce_smem);
    if (tid == 0) {
        block_max = reduced_max;
    }
    __syncthreads();

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        scores[j] = expf(scores[j] - block_max);
        local_sum += scores[j];
    }

    __shared__ float block_sum;
    float reduced_sum = block_reduce_sum<float, BLOCK_THREADS>(local_sum, reduce_smem);
    if (tid == 0) {
        block_sum = reduced_sum;
    }
    __syncthreads();

    // Normalize (protect against divide by zero)
    float inv_sum = block_sum > 0.0f ? 1.0f / block_sum : 0.0f;
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
    float scale, bool is_causal, cudaStream_t stream
) {
    dim3 grid(1, seq_len, batch_size * num_heads);
    constexpr int BLOCK_THREADS = 256;
    constexpr int REDUCE_SMEM = BLOCK_THREADS / 32;
    dim3 block(BLOCK_THREADS);
    size_t smem_size = (seq_len + REDUCE_SMEM) * sizeof(float);

    // Validate shared memory requirement against device limit
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int max_smem;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem,
        cudaDevAttrMaxSharedMemoryPerBlock, device));
    if (static_cast<int>(smem_size) > max_smem) {
        throw std::runtime_error(
            "naive_attention: seq_len=" + std::to_string(seq_len) +
            " requires " + std::to_string(smem_size) +
            " bytes shared memory, but device max is " +
            std::to_string(max_smem) +
            " bytes. Use tiled or flash attention for long sequences.");
    }

    naive_attention_simple_kernel<float><<<grid, block, smem_size, stream>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, head_dim, scale, is_causal
    );
    CUDA_CHECK(cudaGetLastError());
}

void naive_attention_fp16(
    const half* Q, const half* K, const half* V, half* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool is_causal, cudaStream_t stream
) {
    dim3 grid(1, seq_len, batch_size * num_heads);
    constexpr int BLOCK_THREADS = 256;
    constexpr int REDUCE_SMEM = BLOCK_THREADS / 32;
    dim3 block(BLOCK_THREADS);
    size_t smem_size = (seq_len + REDUCE_SMEM) * sizeof(float);

    // Validate shared memory requirement against device limit
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int max_smem;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem,
        cudaDevAttrMaxSharedMemoryPerBlock, device));
    if (static_cast<int>(smem_size) > max_smem) {
        throw std::runtime_error(
            "naive_attention: seq_len=" + std::to_string(seq_len) +
            " requires " + std::to_string(smem_size) +
            " bytes shared memory, but device max is " +
            std::to_string(max_smem) +
            " bytes. Use tiled or flash attention for long sequences.");
    }

    naive_attention_simple_kernel<half><<<grid, block, smem_size, stream>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, head_dim, scale, is_causal
    );
    CUDA_CHECK(cudaGetLastError());
}
