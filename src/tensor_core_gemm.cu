#include "common.cuh"
#include <mma.h>

using namespace nvcuda;

// WMMA dimensions
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Basic Tensor Core GEMM kernel using WMMA
// C = A @ B where A is [M, K], B is [K, N], C is [M, N]
__global__ void tensor_core_gemm_fp16_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Warp and block indices
    int warp_id = (threadIdx.x + threadIdx.y * blockDim.x) / 32;
    int lane_id = threadIdx.x % 32;
    
    // Each warp computes a WMMA_M x WMMA_N tile
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * WMMA_M;
    int warp_col = blockIdx.x * WMMA_N;
    
    if (warp_row >= M || warp_col >= N) return;
    
    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        int a_row = warp_row;
        int a_col = k;
        int b_row = k;
        int b_col = warp_col;
        
        // Bounds check
        if (a_row < M && a_col + WMMA_K <= K && b_col < N) {
            // Load fragments
            wmma::load_matrix_sync(a_frag, A + a_row * K + a_col, K);
            wmma::load_matrix_sync(b_frag, B + b_row * N + b_col, N);
            
            // Matrix multiply-accumulate
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    // Store result
    if (warp_row < M && warp_col < N) {
        wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, N, wmma::mem_row_major);
    }
}

// Tiled Tensor Core GEMM with shared memory
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void tensor_core_gemm_tiled_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    // Shared memory for tiles
    __shared__ half smem_A[BLOCK_M][BLOCK_K];
    __shared__ half smem_B[BLOCK_K][BLOCK_N];
    
    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Number of warps per block
    constexpr int WARPS_M = BLOCK_M / WMMA_M;
    constexpr int WARPS_N = BLOCK_N / WMMA_N;
    
    int warp_row = (warp_id / WARPS_N) * WMMA_M;
    int warp_col = (warp_id % WARPS_N) * WMMA_N;
    
    // Fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Loop over K tiles
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        // Cooperative load of A tile
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += blockDim.x * blockDim.y) {
            int row = i / BLOCK_K;
            int col = i % BLOCK_K;
            int global_row = block_row + row;
            int global_col = k_tile + col;
            
            if (global_row < M && global_col < K) {
                smem_A[row][col] = A[global_row * K + global_col];
            } else {
                smem_A[row][col] = __float2half(0.0f);
            }
        }
        
        // Cooperative load of B tile
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += blockDim.x * blockDim.y) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            int global_row = k_tile + row;
            int global_col = block_col + col;
            
            if (global_row < K && global_col < N) {
                smem_B[row][col] = B[global_row * N + global_col];
            } else {
                smem_B[row][col] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        // Compute using WMMA
        for (int k = 0; k < BLOCK_K; k += WMMA_K) {
            wmma::load_matrix_sync(a_frag, &smem_A[warp_row][k], BLOCK_K);
            wmma::load_matrix_sync(b_frag, &smem_B[k][warp_col], BLOCK_N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        __syncthreads();
    }
    
    // Apply alpha scaling
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] *= alpha;
    }
    
    // Store result with beta scaling
    int c_row = block_row + warp_row;
    int c_col = block_col + warp_col;
    
    if (c_row < M && c_col < N) {
        if (beta != 0.0f) {
            // Load existing C and add
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_old;
            wmma::load_matrix_sync(c_old, C + c_row * N + c_col, N, wmma::mem_row_major);
            for (int i = 0; i < c_frag.num_elements; i++) {
                c_frag.x[i] += beta * c_old.x[i];
            }
        }
        wmma::store_matrix_sync(C + c_row * N + c_col, c_frag, N, wmma::mem_row_major);
    }
}

// INT8 Tensor Core GEMM (requires Turing+ architecture)
#if __CUDA_ARCH__ >= 720
__global__ void tensor_core_gemm_int8_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C,
    int M, int N, int K
) {
    // INT8 WMMA dimensions
    constexpr int WMMA_M_INT8 = 8;
    constexpr int WMMA_N_INT8 = 32;
    constexpr int WMMA_K_INT8 = 16;
    
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * WMMA_M_INT8;
    int warp_col = blockIdx.x * WMMA_N_INT8;
    
    if (warp_row >= M || warp_col >= N) return;
    
    wmma::fragment<wmma::matrix_a, WMMA_M_INT8, WMMA_N_INT8, WMMA_K_INT8, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M_INT8, WMMA_N_INT8, WMMA_K_INT8, int8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M_INT8, WMMA_N_INT8, WMMA_K_INT8, int32_t> c_frag;
    
    wmma::fill_fragment(c_frag, 0);
    
    for (int k = 0; k < K; k += WMMA_K_INT8) {
        if (warp_row < M && k + WMMA_K_INT8 <= K && warp_col < N) {
            wmma::load_matrix_sync(a_frag, A + warp_row * K + k, K);
            wmma::load_matrix_sync(b_frag, B + k * N + warp_col, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    if (warp_row < M && warp_col < N) {
        wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, N, wmma::mem_row_major);
    }
}
#endif

// Host wrappers
void tensor_core_gemm_fp16(
    const half* A, const half* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
) {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 32;
    
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(32, 4);  // 4 warps per block
    
    tensor_core_gemm_tiled_kernel<BLOCK_M, BLOCK_N, BLOCK_K><<<grid, block, 0, stream>>>(
        A, B, C, M, N, K, alpha, beta
    );
    CUDA_CHECK(cudaGetLastError());
}

void tensor_core_gemm_int8(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K,
    cudaStream_t stream
) {
#if __CUDA_ARCH__ >= 720
    dim3 grid((N + 31) / 32, (M + 7) / 8);
    dim3 block(32, 1);
    
    tensor_core_gemm_int8_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
#else
    throw std::runtime_error("INT8 Tensor Core requires Turing+ architecture (SM 7.2+)");
#endif
}
