#include "common.cuh"
#include "warp_primitives.cuh"
#include "shared_memory.cuh"
#include <mma.h>

using namespace nvcuda;

// High-performance GEMM with register tiling and double buffering
// C = alpha * A @ B + beta * C
template<
    typename T,
    int BLOCK_M,      // Thread block tile M
    int BLOCK_N,      // Thread block tile N  
    int BLOCK_K,      // Thread block tile K
    int WARP_M,       // Warp tile M
    int WARP_N,       // Warp tile N
    int THREAD_M,     // Thread tile M (register)
    int THREAD_N      // Thread tile N (register)
>
__global__ void hgemm_register_tiled_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta,
    bool trans_a, bool trans_b
) {
    // Double-buffered shared memory with padding for bank conflict avoidance
    __shared__ float smem_A[2][BLOCK_M][BLOCK_K + 1];
    __shared__ float smem_B[2][BLOCK_K][BLOCK_N + 1];
    
    // Thread indices
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Block position
    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;
    
    // Warp position within block
    constexpr int WARPS_N = BLOCK_N / WARP_N;
    int warp_row = (warp_id / WARPS_N) * WARP_M;
    int warp_col = (warp_id % WARPS_N) * WARP_N;
    
    // Thread position within warp
    constexpr int THREADS_N = WARP_N / THREAD_N;
    int thread_row = (lane_id / THREADS_N) * THREAD_M;
    int thread_col = (lane_id % THREADS_N) * THREAD_N;
    
    // Register tiles for output
    float reg_C[THREAD_M][THREAD_N] = {0.0f};
    
    // Register tiles for A and B
    float reg_A[THREAD_M];
    float reg_B[THREAD_N];
    
    // Number of threads for loading
    int num_threads = blockDim.x * blockDim.y;
    int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;
    
    // Helper lambda-like macros for loading tiles into a specific buffer
    #define LOAD_TILE_A(buf, k_tile_offset) \
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += num_threads) { \
            int row = i / BLOCK_K; \
            int col = i % BLOCK_K; \
            int global_row = block_row + row; \
            int global_col = (k_tile_offset) + col; \
            float val = 0.0f; \
            if (global_row < M && global_col < K) { \
                if (trans_a) { \
                    val = static_cast<float>(A[global_col * M + global_row]); \
                } else { \
                    val = static_cast<float>(A[global_row * K + global_col]); \
                } \
            } \
            smem_A[buf][row][col] = val; \
        }
    
    #define LOAD_TILE_B(buf, k_tile_offset) \
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += num_threads) { \
            int row = i / BLOCK_N; \
            int col = i % BLOCK_N; \
            int global_row = (k_tile_offset) + row; \
            int global_col = block_col + col; \
            float val = 0.0f; \
            if (global_row < K && global_col < N) { \
                if (trans_b) { \
                    val = static_cast<float>(B[global_col * K + global_row]); \
                } else { \
                    val = static_cast<float>(B[global_row * N + global_col]); \
                } \
            } \
            smem_B[buf][row][col] = val; \
        }
    
    #define COMPUTE_TILE(buf) \
        _Pragma("unroll") \
        for (int k = 0; k < BLOCK_K; k++) { \
            _Pragma("unroll") \
            for (int m = 0; m < THREAD_M; m++) { \
                reg_A[m] = smem_A[buf][warp_row + thread_row + m][k]; \
            } \
            _Pragma("unroll") \
            for (int n = 0; n < THREAD_N; n++) { \
                reg_B[n] = smem_B[buf][k][warp_col + thread_col + n]; \
            } \
            _Pragma("unroll") \
            for (int m = 0; m < THREAD_M; m++) { \
                _Pragma("unroll") \
                for (int n = 0; n < THREAD_N; n++) { \
                    reg_C[m][n] += reg_A[m] * reg_B[n]; \
                } \
            } \
        }
    
    // Prologue: load first tile into buffer 0
    LOAD_TILE_A(0, 0);
    LOAD_TILE_B(0, 0);
    __syncthreads();
    
    // Main loop with double buffering
    for (int tile = 0; tile < num_k_tiles; tile++) {
        int cur_buf = tile % 2;
        int next_buf = 1 - cur_buf;
        int next_k_tile = (tile + 1) * BLOCK_K;
        
        // Prefetch next tile into alternate buffer (if available)
        if (tile + 1 < num_k_tiles) {
            LOAD_TILE_A(next_buf, next_k_tile);
            LOAD_TILE_B(next_buf, next_k_tile);
        }
        
        // Compute on current buffer
        COMPUTE_TILE(cur_buf);
        
        __syncthreads();
    }
    
    #undef LOAD_TILE_A
    #undef LOAD_TILE_B
    #undef COMPUTE_TILE
    
    // Store results
    #pragma unroll
    for (int m = 0; m < THREAD_M; m++) {
        #pragma unroll
        for (int n = 0; n < THREAD_N; n++) {
            int global_row = block_row + warp_row + thread_row + m;
            int global_col = block_col + warp_col + thread_col + n;
            
            if (global_row < M && global_col < N) {
                float result = alpha * reg_C[m][n];
                if (beta != 0.0f) {
                    result += beta * static_cast<float>(C[global_row * N + global_col]);
                }
                C[global_row * N + global_col] = static_cast<T>(result);
            }
        }
    }
}

// GEMM with different matrix layouts
template<typename T>
void hgemm_impl(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    float alpha, float beta,
    MatrixLayout layout_a, MatrixLayout layout_b,
    cudaStream_t stream
) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 32;
    constexpr int WARP_M = 32;
    constexpr int WARP_N = 64;
    constexpr int THREAD_M = 8;
    constexpr int THREAD_N = 8;
    
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    // BLOCK_M/WARP_M * BLOCK_N/WARP_N = 4*2 = 8 warps needed
    dim3 block(32, 8);  // 256 threads = 8 warps
    
    bool trans_a = (layout_a == MatrixLayout::ColMajor);
    bool trans_b = (layout_b == MatrixLayout::ColMajor);
    
    hgemm_register_tiled_kernel<T, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, THREAD_M, THREAD_N>
        <<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, trans_a, trans_b);
    
    CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiations
template void hgemm_impl<float>(
    const float*, const float*, float*,
    int, int, int, float, float,
    MatrixLayout, MatrixLayout, cudaStream_t
);

template void hgemm_impl<half>(
    const half*, const half*, half*,
    int, int, int, float, float,
    MatrixLayout, MatrixLayout, cudaStream_t
);

// Host wrapper functions
void hgemm_fp32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    MatrixLayout layout_a, MatrixLayout layout_b,
    cudaStream_t stream
) {
    hgemm_impl<float>(A, B, C, M, N, K, alpha, beta, layout_a, layout_b, stream);
}

void hgemm_fp16(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    float alpha, float beta,
    MatrixLayout layout_a, MatrixLayout layout_b,
    cudaStream_t stream
) {
    hgemm_impl<half>(A, B, C, M, N, K, alpha, beta, layout_a, layout_b, stream);
}

// Convenience wrappers for different transpose combinations
void gemm_nn(const float* A, const float* B, float* C, int M, int N, int K, 
             float alpha, float beta, cudaStream_t stream) {
    hgemm_fp32(A, B, C, M, N, K, alpha, beta, MatrixLayout::RowMajor, MatrixLayout::RowMajor, stream);
}

void gemm_nt(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta, cudaStream_t stream) {
    hgemm_fp32(A, B, C, M, N, K, alpha, beta, MatrixLayout::RowMajor, MatrixLayout::ColMajor, stream);
}

void gemm_tn(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta, cudaStream_t stream) {
    hgemm_fp32(A, B, C, M, N, K, alpha, beta, MatrixLayout::ColMajor, MatrixLayout::RowMajor, stream);
}

void gemm_tt(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta, cudaStream_t stream) {
    hgemm_fp32(A, B, C, M, N, K, alpha, beta, MatrixLayout::ColMajor, MatrixLayout::ColMajor, stream);
}
