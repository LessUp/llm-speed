#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>
#include <cstdint>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + \
            cudaGetErrorString(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)

// Matrix layout enumeration
enum class MatrixLayout {
    RowMajor,
    ColMajor,
    RowMajorPadded
};

// Precision enumeration
enum class Precision {
    FP32,
    FP16,
    BF16,
    INT8
};

// Kernel error codes
enum class KernelError {
    SUCCESS = 0,
    INVALID_DIMENSION,
    UNSUPPORTED_DTYPE,
    ALIGNMENT_ERROR,
    OUT_OF_MEMORY,
    CUDA_ERROR,
    INVALID_CONFIG
};

// Attention configuration
struct AttentionConfig {
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
    float scale;
    bool is_causal;
    int block_m;
    int block_n;
    Precision precision;
    
    AttentionConfig() : batch_size(1), num_heads(1), seq_len(128), head_dim(64),
                        scale(0.0f), is_causal(false), block_m(64), block_n(64),
                        precision(Precision::FP32) {
        if (scale == 0.0f) {
            scale = 1.0f / sqrtf(static_cast<float>(head_dim));
        }
    }
};

// GEMM configuration
struct GemmConfig {
    int M, N, K;
    float alpha, beta;
    MatrixLayout layout_a;
    MatrixLayout layout_b;
    int block_m, block_n, block_k;
    int warp_m, warp_n;
    int thread_m, thread_n;
    Precision precision;
    
    GemmConfig() : M(0), N(0), K(0), alpha(1.0f), beta(0.0f),
                   layout_a(MatrixLayout::RowMajor), layout_b(MatrixLayout::RowMajor),
                   block_m(128), block_n(128), block_k(32),
                   warp_m(64), warp_n(64), thread_m(8), thread_n(8),
                   precision(Precision::FP16) {}
};

// Performance metrics
struct KernelMetrics {
    float elapsed_ms;
    float tflops;
    float memory_bandwidth_gb;
    float sm_occupancy;
    float l2_hit_rate;
    
    enum class Bottleneck {
        COMPUTE_BOUND,
        MEMORY_BOUND,
        LATENCY_BOUND
    } bottleneck;
};

// Utility functions
inline int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

inline int align_up(int a, int alignment) {
    return div_ceil(a, alignment) * alignment;
}

// Check if dimension is aligned for Tensor Core
inline bool is_tensor_core_aligned(int dim, int alignment = 16) {
    return (dim % alignment) == 0;
}
