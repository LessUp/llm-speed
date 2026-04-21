#include "common.cuh"
#include "shared_memory.cuh"
#include "warp_primitives.cuh"
#include <cfloat>

// Tiled attention kernel using shared memory
template<typename T, int BLOCK_M, int BLOCK_N>
__global__ void tiled_attention_kernel(
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
    // Block indices
    int batch_head_idx = blockIdx.z;
    int batch_idx = batch_head_idx / num_heads;
    int head_idx = batch_head_idx % num_heads;
    int block_row = blockIdx.y;

    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;
    int row_start = block_row * BLOCK_M;

    // Dynamic shared memory layout (sized by actual head_dim, with +1 padding for bank conflicts)
    // Layout: smem_Q[BLOCK_M * (head_dim+1)] | smem_K[BLOCK_N * (head_dim+1)] |
    //         smem_V[BLOCK_N * (head_dim+1)] | smem_S[BLOCK_M * (BLOCK_N+1)]
    extern __shared__ float smem[];
    int hd_stride = head_dim + 1;  // padded stride for bank conflict avoidance
    float* smem_Q = smem;
    float* smem_K = smem_Q + BLOCK_M * hd_stride;
    float* smem_V = smem_K + BLOCK_N * hd_stride;
    float* smem_S = smem_V + BLOCK_N * hd_stride;
    int sn_stride = BLOCK_N + 1;  // padded stride for scores

    // Per-row state in registers (small arrays, BLOCK_M is typically 32)
    float row_max[BLOCK_M];
    float row_sum[BLOCK_M];

    // Output accumulator in shared memory (after smem_S)
    float* output = smem_S + BLOCK_M * sn_stride;
    for (int i = tid; i < BLOCK_M * head_dim; i += blockDim.x) {
        output[i] = 0.0f;
    }
    __syncthreads();

    // Initialize per-row state
    #pragma unroll
    for (int m = 0; m < BLOCK_M; m++) {
        row_max[m] = -FLT_MAX;
        row_sum[m] = 0.0f;
    }

    // Pointer offsets (use int64 to avoid overflow for large tensors)
    int64_t offset = (static_cast<int64_t>(batch_idx) * num_heads + head_idx) * seq_len * head_dim;
    const T* q_ptr = Q + offset;
    const T* k_ptr = K + offset;
    const T* v_ptr = V + offset;
    T* o_ptr = O + offset;

    // Load Q tile to shared memory
    for (int i = tid; i < BLOCK_M * head_dim; i += blockDim.x) {
        int m = i / head_dim;
        int k = i % head_dim;
        int global_row = row_start + m;
        if (global_row < seq_len) {
            smem_Q[m * hd_stride + k] = static_cast<float>(q_ptr[global_row * head_dim + k]);
        } else {
            smem_Q[m * hd_stride + k] = 0.0f;
        }
    }
    __syncthreads();

    // Iterate over K/V blocks
    int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int col_start = kv_block * BLOCK_N;

        // Early exit for causal: skip blocks entirely past the current row block
        if (is_causal && col_start >= row_start + BLOCK_M) break;

        // Load K tile
        for (int i = tid; i < BLOCK_N * head_dim; i += blockDim.x) {
            int n = i / head_dim;
            int k = i % head_dim;
            int global_col = col_start + n;
            if (global_col < seq_len) {
                smem_K[n * hd_stride + k] = static_cast<float>(k_ptr[global_col * head_dim + k]);
            } else {
                smem_K[n * hd_stride + k] = 0.0f;
            }
        }

        // Load V tile
        for (int i = tid; i < BLOCK_N * head_dim; i += blockDim.x) {
            int n = i / head_dim;
            int k = i % head_dim;
            int global_col = col_start + n;
            if (global_col < seq_len) {
                smem_V[n * hd_stride + k] = static_cast<float>(v_ptr[global_col * head_dim + k]);
            } else {
                smem_V[n * hd_stride + k] = 0.0f;
            }
        }
        __syncthreads();

        // Compute attention scores: Q @ K^T
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += blockDim.x) {
            int m = i / BLOCK_N;
            int n = i % BLOCK_N;

            float score = 0.0f;
            int global_row = row_start + m;
            int global_col = col_start + n;

            // Apply causal mask
            if (is_causal && global_col > global_row) {
                score = -FLT_MAX;
            } else if (global_row < seq_len && global_col < seq_len) {
                for (int k = 0; k < head_dim; k++) {
                    score += smem_Q[m * hd_stride + k] * smem_K[n * hd_stride + k];
                }
                score *= scale;
            } else {
                score = -FLT_MAX;
            }

            smem_S[m * sn_stride + n] = score;
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
                    block_max = fmaxf(block_max, smem_S[m * sn_stride + n]);
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
                    smem_S[m * sn_stride + n] = expf(smem_S[m * sn_stride + n] - new_max);
                    block_sum += smem_S[m * sn_stride + n];
                } else {
                    smem_S[m * sn_stride + n] = 0.0f;
                }
            }

            // Update running sum
            float new_sum = row_sum[m] * old_scale + block_sum;

            // Rescale old output and add new contribution
            float rescale = (row_sum[m] > 0.0f) ? old_scale * row_sum[m] / fmaxf(new_sum, 1e-6f) : 0.0f;

            for (int k = 0; k < head_dim; k++) {
                output[m * head_dim + k] *= rescale;

                float new_val = 0.0f;
                for (int n = 0; n < BLOCK_N; n++) {
                    new_val += smem_S[m * sn_stride + n] * smem_V[n * hd_stride + k];
                }
                output[m * head_dim + k] += new_val / fmaxf(new_sum, 1e-6f);
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
        if (global_row < seq_len) {
            o_ptr[global_row * head_dim + k] = static_cast<T>(output[m * head_dim + k]);
        }
    }
}

// Host wrapper
void tiled_attention_fp32(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool is_causal, cudaStream_t stream
) {
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;

    dim3 grid(1, (seq_len + BLOCK_M - 1) / BLOCK_M, batch_size * num_heads);
    dim3 block(256);

    int hd_stride = head_dim + 1;
    int sn_stride = BLOCK_N + 1;
    size_t smem_size = (BLOCK_M * hd_stride + 2 * BLOCK_N * hd_stride
                       + BLOCK_M * sn_stride + BLOCK_M * head_dim) * sizeof(float);

    // Validate shared memory requirement against device limit
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int max_smem;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem,
        cudaDevAttrMaxSharedMemoryPerBlock, device));
    if (static_cast<int>(smem_size) > max_smem) {
        throw std::runtime_error(
            "tiled_attention: head_dim=" + std::to_string(head_dim) +
            " requires " + std::to_string(smem_size) +
            " bytes shared memory, but device max is " +
            std::to_string(max_smem) + " bytes.");
    }

    tiled_attention_kernel<float, BLOCK_M, BLOCK_N><<<grid, block, smem_size, stream>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, head_dim, scale, is_causal
    );
    CUDA_CHECK(cudaGetLastError());
}

void tiled_attention_fp16(
    const half* Q, const half* K, const half* V, half* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool is_causal, cudaStream_t stream
) {
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;

    dim3 grid(1, (seq_len + BLOCK_M - 1) / BLOCK_M, batch_size * num_heads);
    dim3 block(256);

    int hd_stride = head_dim + 1;
    int sn_stride = BLOCK_N + 1;
    size_t smem_size = (BLOCK_M * hd_stride + 2 * BLOCK_N * hd_stride
                       + BLOCK_M * sn_stride + BLOCK_M * head_dim) * sizeof(float);

    // Validate shared memory requirement against device limit
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int max_smem;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem,
        cudaDevAttrMaxSharedMemoryPerBlock, device));
    if (static_cast<int>(smem_size) > max_smem) {
        throw std::runtime_error(
            "tiled_attention: head_dim=" + std::to_string(head_dim) +
            " requires " + std::to_string(smem_size) +
            " bytes shared memory, but device max is " +
            std::to_string(max_smem) + " bytes.");
    }

    tiled_attention_kernel<half, BLOCK_M, BLOCK_N><<<grid, block, smem_size, stream>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, head_dim, scale, is_causal
    );
    CUDA_CHECK(cudaGetLastError());
}
