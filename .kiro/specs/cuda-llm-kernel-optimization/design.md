# Design Document: CUDA LLM Kernel Optimization

## Overview

本设计文档描述基于 CUDA 的高性能 LLM 核心算子优化实现，包括 FlashAttention 算法复现和高性能 GEMM kernel。系统采用分层架构，从 Naive 实现逐步优化到使用 Tensor Core 和流水线技术的高性能版本。

### 设计目标

1. **性能目标**: GEMM 达到 cuBLAS 90%+ 性能，FlashAttention 显存复杂度 O(N)
2. **精度支持**: FP32, FP16, BF16, INT8 混合精度
3. **可扩展性**: 模块化设计，支持不同 GPU 架构（Volta, Ampere, Hopper）
4. **易用性**: 提供 PyTorch 兼容的 Python 接口
5. **正确性**: 所有实现与 PyTorch 参考实现数值误差在容差范围内

### 设计约束

- **CUDA Compute Capability**: 最低 SM70 (Volta)，推荐 SM80+ (Ampere)
- **共享内存限制**: 每个 SM 最大 48KB (Volta) / 164KB (Ampere)
- **寄存器限制**: 每个线程最大 255 个 32-bit 寄存器
- **Tensor Core 对齐**: 矩阵维度需为 16 的倍数（或自动 padding）

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Python Interface Layer                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ flash_attention │  │   gemm_kernel   │  │    profiler     │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼─────────────────────┼─────────────────────┼──────────┘
            │                     │                     │
┌───────────┼─────────────────────┼─────────────────────┼──────────┐
│           ▼                     ▼                     ▼          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    CUDA Kernel Layer                        │ │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │ │
│  │  │ Attention     │  │ GEMM          │  │ Warp          │   │ │
│  │  │ Kernels       │  │ Kernels       │  │ Primitives    │   │ │
│  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘   │ │
│  └──────────┼──────────────────┼──────────────────┼───────────┘ │
│             │                  │                  │              │
│  ┌──────────┼──────────────────┼──────────────────┼───────────┐ │
│  │          ▼                  ▼                  ▼            │ │
│  │  ┌─────────────────────────────────────────────────────┐   │ │
│  │  │              Optimization Components                 │   │ │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │ │
│  │  │  │ Tiling      │ │ Tensor Core │ │ Pipeline    │    │   │ │
│  │  │  │ Manager     │ │ Accelerator │ │ Scheduler   │    │   │ │
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘    │   │ │
│  │  └─────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                        CUDA Runtime                              │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Naive Attention Kernel

基础实现，作为性能基准和正确性参考。

```cpp
// src/naive_attention.cu
// 每个 block 处理一个 (batch, head, row)
template<typename T>
__global__ void naive_attention_simple_kernel(
    const T* __restrict__ Q,   // [batch, heads, seq_len, head_dim]
    const T* __restrict__ K,   // [batch, heads, seq_len, head_dim]
    const T* __restrict__ V,   // [batch, heads, seq_len, head_dim]
    T* __restrict__ O,         // [batch, heads, seq_len, head_dim]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale                // 1/sqrt(head_dim)
);

// 计算流程:
// 1. S = Q @ K^T * scale    [seq_len, seq_len]   — 共享内存 scores[]
// 2. P = softmax(S, dim=-1) [seq_len, seq_len]   — block 级 max/sum 规约
// 3. O = P @ V              [seq_len, head_dim]
//
// Grid: (1, seq_len, batch_size * num_heads)
// Block: 256 threads
// 动态共享内存: (seq_len + num_warps) * sizeof(float)
```

### 2. Tiled Attention Kernel

使用共享内存分块优化的版本。

```cpp
// src/tiled_attention.cu
template<typename T, int BLOCK_M, int BLOCK_N>
__global__ void tiled_attention_kernel(
    const T* __restrict__ Q, const T* __restrict__ K,
    const T* __restrict__ V, T* __restrict__ O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale
);

// Tiling 策略:
// - BLOCK_M: Q 的行分块大小 (当前值: 32)
// - BLOCK_N: K/V 的行分块大小 (当前值: 32)
// - head_dim: 直接使用完整 head_dim，不再做 K 维度分块

// 动态共享内存布局 (带 +1 padding 消除 bank conflict):
// hd_stride = head_dim + 1
// sn_stride = BLOCK_N + 1
// smem_Q  [BLOCK_M * hd_stride]     — Q tile
// smem_K  [BLOCK_N * hd_stride]     — K tile
// smem_V  [BLOCK_N * hd_stride]     — V tile
// smem_S  [BLOCK_M * sn_stride]     — attention scores
// output  [BLOCK_M * head_dim]      — 输出累加器
//
// 使用 Online Softmax 流式更新，遍历 K/V 分块时逐步更新 row_max/row_sum
//
// Grid: (1, ceil(seq_len/BLOCK_M), batch_size * num_heads)
// Block: 256 threads
```

### 3. FlashAttention Engine

实现 Online Softmax 的核心算法。

```cpp
// src/flash_attention.cu
template<typename T, int BLOCK_M, int BLOCK_N>
__global__ void flash_attention_forward_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale,
    bool is_causal        // 是否使用因果掩码
);

// Online Softmax 状态 (include/online_softmax.cuh):
struct OnlineSoftmaxState {
    float max_val;        // 当前最大值 m_i
    float sum_exp;        // 当前指数和 l_i
};

// 共享内存布局 (带 +1 padding, K/V 双缓冲):
// smem_Q    [BLOCK_M * (head_dim+1)]      — Q tile
// smem_K[2] [2 * BLOCK_N * (head_dim+1)]  — K double buffer
// smem_V[2] [2 * BLOCK_N * (head_dim+1)]  — V double buffer
// smem_S  [BLOCK_M * (BLOCK_N+1)]     — attention scores / exp(scores)
// output  [BLOCK_M * head_dim]        — 输出累加器
// row_max [BLOCK_M]                   — 每行 running max
// row_sum [BLOCK_M]                   — 每行 running sum
// rescale [BLOCK_M]                   — 每行 rescale 因子

// Double Buffering 流程:
// Prologue: 加载第一个 K/V tile 到 buffer 0
// 主循环: 计算当前 buffer 的同时预取下一个 tile 到交替 buffer
// Causal 早退时跳过不需要的 prefetch
//
// 算法核心 (单个 tile 更新, 分两阶段):
// Phase 1: 单线程每行计算 softmax 状态 (BLOCK_N 次迭代, 开销小)
//   1. 计算 S_ij = Q_i @ K_j^T * scale
//   2. 更新 max: m_new = max(m_old, rowmax(S_ij))
//   3. 原地计算 exp(S_ij - m_new)
//   4. 更新 sum: l_new = exp(m_old - m_new) * l_old + rowsum(exp(S_ij - m_new))
//   5. 计算 rescale 因子
// Phase 2: 全线程协作更新输出 (开销大, 充分并行)
//   output[m,d] = output[m,d] * rescale[m] + (1/l_new) * sum_n(exp_S[m,n] * V[n,d])
//
// Grid: (1, ceil(seq_len/BLOCK_M), batch_size * num_heads)
// Block: 128 threads
// 当前配置: BLOCK_M = 32, BLOCK_N = 32
```

### 4. Tensor Core GEMM Kernel

使用 WMMA/MMA 指令的高性能矩阵乘法。

```cpp
// src/tensor_core_gemm.cu
#include <mma.h>
using namespace nvcuda;

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

// 基础版: 直接从全局内存加载
__global__ void tensor_core_gemm_fp16_kernel(
    const half* __restrict__ A,  // [M, K]
    const half* __restrict__ B,  // [K, N]
    float* __restrict__ C,       // [M, N]
    int M, int N, int K
);
// Fragments: a_frag(row_major), b_frag(col_major), c_frag(float)

// Tiled 版: 共享内存 + 多 warp 协作
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void tensor_core_gemm_tiled_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
);
// Tiled 版 Fragments: a_frag(row_major), b_frag(row_major), c_frag(float)
// 共享内存: smem_A[BLOCK_M][BLOCK_K+8], smem_B[BLOCK_K][BLOCK_N+8]
// 当前配置: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, 4 warps

// INT8 版 (需要 Turing+ SM≥7.2):
__global__ void tensor_core_gemm_int8_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C,
    int M, int N, int K
);
// INT8 WMMA 尺寸: 8x32x16
```

### 5. High-Performance GEMM with Register Tiling

```cpp
// src/hgemm_kernel.cu
template<
    typename T,
    int BLOCK_M,          // Thread block tile M  (128)
    int BLOCK_N,          // Thread block tile N  (128)
    int BLOCK_K,          // Thread block tile K  (32)
    int WARP_M,           // Warp tile M          (32)
    int WARP_N,           // Warp tile N          (64)
    int THREAD_M,         // Thread tile M        (8)
    int THREAD_N          // Thread tile N        (8)
>
__global__ void hgemm_register_tiled_kernel(
    const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta,
    bool trans_a, bool trans_b   // 支持转置布局
);

// Register Tiling 布局:
// 每个线程持有 THREAD_M x THREAD_N 的输出 tile
// 寄存器数组: float reg_C[THREAD_M][THREAD_N] = {0};
//            float reg_A[THREAD_M];
//            float reg_B[THREAD_N];

// Double Buffering:
// smem_A[2][BLOCK_M][BLOCK_K + 1]  — 双缓冲 + padding
// smem_B[2][BLOCK_K][BLOCK_N + 1]
// 主循环中交替使用两组缓冲区，实现计算与访存重叠
//
// Grid: (ceil(N/BLOCK_N), ceil(M/BLOCK_M))
// Block: (32, 4) = 128 threads = 4 warps
```

### 6. Pipeline Scheduler

```cpp
// include/pipeline.cuh
template<int NUM_STAGES>
class PipelineScheduler {
public:
    struct StageBuffer {
        void* smem_ptr;
        int tile_idx;
        bool valid;
    };
    
    StageBuffer buffers[NUM_STAGES];
    int current_stage;   // 加载阶段指针
    int compute_stage;   // 计算阶段指针
    
    __device__ void set_buffer(int stage, void* ptr);
    __device__ void* get_load_buffer();     // buffers[current_stage]
    __device__ void* get_compute_buffer();  // buffers[compute_stage]
    __device__ void advance_load();         // current_stage = (current_stage + 1) % NUM_STAGES
    __device__ void advance_compute();      // compute_stage = (compute_stage + 1) % NUM_STAGES
    __device__ bool is_compute_ready();     // buffers[compute_stage].valid
};

// Double Buffering 辅助结构
template<typename T, int BLOCK_SIZE>
struct DoubleBuffer {
    T* buffer[2];
    int current;
    __device__ T* get_load_buffer();     // buffer[current]
    __device__ T* get_compute_buffer();  // buffer[1 - current]
    __device__ void swap();              // current = 1 - current
};

// Async Copy Helpers (Ampere+ SM≥80, 旧架构回退到同步拷贝)
template<int BYTES>  // 4, 8, or 16
__device__ void async_copy(void* dst, const void* src);  // cp.async PTX
__device__ void async_copy_commit();     // cp.async.commit_group
template<int N>
__device__ void async_copy_wait();       // cp.async.wait_group N

// Software Pipelining 模板
template<int STAGES, typename LoadFunc, typename ComputeFunc>
__device__ void software_pipeline(int num_iterations, LoadFunc, ComputeFunc);
// Prologue: 填充 STAGES-1 个 tile
// Main loop: 计算当前 stage + 预取下一个 tile
// Epilogue: 排空流水线

// 实际使用:
// hgemm_kernel.cu 自行实现了 double buffering (宏 + 两组 smem)
// pipeline.cuh 提供通用抽象，但当前未集成到 FlashAttention
```

### 7. Warp-Level Primitives

```cpp
// include/warp_primitives.cuh

// Warp Shuffle 规约
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val);   // 求和
template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val);   // 求最大值
template<typename T>
__device__ __forceinline__ T warp_reduce_min(T val);   // 求最小值

// Block-level 规约 (使用共享内存, warp 内规约 + 跨 warp 规约)
template<typename T, int BLOCK_SIZE>
__device__ __forceinline__ T block_reduce_sum(T val, T* smem);
template<typename T, int BLOCK_SIZE>
__device__ __forceinline__ T block_reduce_max(T val, T* smem);

// Warp Shuffle 广播和交换
template<typename T>
__device__ __forceinline__ T warp_broadcast(T val, int src_lane);  // __shfl_sync
template<typename T>
__device__ __forceinline__ T warp_shuffle_xor(T val, int mask);    // __shfl_xor_sync (butterfly)

// 实现原理 (warp_reduce_sum 为例):
// #pragma unroll
// for (int offset = 16; offset > 0; offset /= 2)
//     val += __shfl_down_sync(0xffffffff, val, offset);
```

### 8. Python Interface

通过 pybind11 C++ 绑定提供统一的 `cuda_llm_ops` Python 模块，而非独立的 `.py` 文件。

```cpp
// python/bindings.cpp
// 编译为 Python 模块: cuda_llm_ops

PYBIND11_MODULE(cuda_llm_ops, m) {
    // Attention 函数
    m.def("naive_attention", &naive_attention,
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale") = 0.0f);
    m.def("tiled_attention", &tiled_attention,
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale") = 0.0f);
    m.def("flash_attention", &flash_attention,
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("scale") = 0.0f, py::arg("is_causal") = false);
    
    // GEMM 函数
    m.def("gemm", &gemm,
          py::arg("a"), py::arg("b"),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("trans_a") = false, py::arg("trans_b") = false);
    m.def("tensor_core_gemm", &tensor_core_gemm,
          py::arg("a"), py::arg("b"),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);
}
```

```python
# python/__init__.py
# 统一导出接口
from cuda_llm_ops import (
    naive_attention,     # Q,K,V -> O  (baseline)
    tiled_attention,     # Q,K,V -> O  (shared memory tiling)
    flash_attention,     # Q,K,V -> O  (online softmax, 支持 causal)
    gemm,                # A,B -> C    (register tiling, 支持转置)
    tensor_core_gemm,    # A,B -> C    (FP16 in, FP32 out)
)

# 所有函数接受 PyTorch Tensor 输入，返回 PyTorch Tensor 输出
# 支持 dtype: float32, float16
# 输入必须在 CUDA 设备上且 contiguous
```

```python
# 输入验证 (在 bindings.cpp 中实现):
# - Q/K/V 必须为 4D Tensor [batch, heads, seq_len, head_dim]
# - Q/K/V shape 必须一致
# - 必须在 CUDA 设备上
# - 必须 contiguous
# - dtype 仅支持 float32 和 float16
# - GEMM: A/B 必须为 2D Tensor，内维度匹配
```

## Data Models

### Memory Layout

```cpp
// 矩阵存储格式
enum class MatrixLayout {
    RowMajor,      // C-style, 行优先
    ColMajor,      // Fortran-style, 列优先
    RowMajorPadded // 带 padding 的行优先 (消除 bank conflict)
};

// Attention 输入输出格式
// Q, K, V, O: [batch_size, num_heads, seq_len, head_dim]
// 内存布局: batch -> head -> seq -> dim (连续)

// GEMM 输入输出格式
// A: [M, K], B: [K, N], C: [M, N]
// 支持转置: A^T, B^T
```

### Kernel Configuration

```cpp
struct AttentionConfig {
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
    float scale;
    bool is_causal;
    
    // Tiling 参数
    int block_m;      // Q tile size
    int block_n;      // K/V tile size
    
    // 精度
    enum Precision { FP32, FP16, BF16, INT8 } precision;
};

struct GemmConfig {
    int M, N, K;
    float alpha, beta;
    MatrixLayout layout_a;
    MatrixLayout layout_b;
    
    // Tiling 参数
    int block_m, block_n, block_k;
    int warp_m, warp_n;
    int thread_m, thread_n;
    
    // 精度
    enum Precision { FP32, FP16, INT8 } precision;
};
```

### Performance Metrics

```cpp
struct KernelMetrics {
    float elapsed_ms;           // 执行时间
    float tflops;               // 计算吞吐量
    float memory_bandwidth_gb;  // 内存带宽 (GB/s)
    float sm_occupancy;         // SM 占用率
    float l2_hit_rate;          // L2 缓存命中率
    
    // 瓶颈分析
    enum Bottleneck { 
        COMPUTE_BOUND,          // 计算受限
        MEMORY_BOUND,           // 访存受限
        LATENCY_BOUND           // 延迟受限
    } bottleneck;
};
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Attention 计算正确性 (Round-Trip)

*For any* 有效的 Q, K, V 输入张量（任意 batch_size, num_heads, seq_len, head_dim），Attention_Kernel 的输出与 PyTorch 参考实现 `torch.nn.functional.scaled_dot_product_attention` 的数值误差应在容差范围内（FP32: 1e-3, FP16: 1e-2）。

**Validates: Requirements 1.1, 1.3, 1.4, 1.5**

### Property 2: Softmax 数学不变量

*For any* 输入向量，Softmax 输出应满足以下不变量：
- 所有输出值在 (0, 1) 范围内
- 输出向量的和等于 1（在浮点误差范围内）
- 输出的相对顺序与输入一致（单调性保持）

**Validates: Requirements 1.2**

### Property 3: FlashAttention 与标准实现一致性

*For any* 有效的 Q, K, V 输入，FlashAttention_Engine 的输出与 Naive Attention 实现的数值误差应在容差范围内。Online Softmax 的流式计算结果应与标准 Softmax 数学等价。

**Validates: Requirements 3.1, 3.6**

### Property 4: 因果掩码正确性

*For any* 启用因果掩码的 attention 计算，对于输出位置 i，其值只依赖于输入位置 j ≤ i 的数据。等价地，attention 权重矩阵应为下三角矩阵（对角线及以下非零）。

**Validates: Requirements 3.5**

### Property 5: FP16 GEMM 正确性

*For any* 有效的 FP16 矩阵 A[M,K] 和 B[K,N]，GEMM_Kernel 和 Tensor_Core_Accelerator 的输出 C = A @ B 与 PyTorch `torch.matmul` 的数值误差应在 FP16 容差范围内（1e-2）。

**Validates: Requirements 4.2, 5.1**

### Property 6: INT8 GEMM 正确性

*For any* 有效的 INT8 矩阵 A[M,K] 和 B[K,N]，量化 GEMM 的输出与参考实现（INT32 累加后截断）应完全一致。

**Validates: Requirements 4.3, 5.2**

### Property 7: 矩阵布局等价性

*For any* 矩阵 A 和 B，以及任意布局组合（NN, NT, TN, TT），GEMM 输出应满足数学等价性：
- C_NN = A @ B
- C_NT = A @ B^T
- C_TN = A^T @ B
- C_TT = A^T @ B^T

**Validates: Requirements 5.6**

### Property 8: 维度对齐处理

*For any* 输入矩阵维度，当维度不满足 Tensor Core 对齐要求（16 的倍数）时，kernel 应正确处理（通过 padding 或回退到非 Tensor Core 路径），输出结果仍然正确。

**Validates: Requirements 4.4**

### Property 9: 流水线深度配置正确性

*For any* 支持的流水线深度配置（2, 3, 4 级），Pipeline_Scheduler 应产生与非流水线版本数值一致的输出。

**Validates: Requirements 6.4**

### Property 10: Python 接口兼容性

*For any* 有效的 PyTorch Tensor 输入，Python 绑定接口应：
- 接受 PyTorch Tensor 作为输入
- 返回 PyTorch Tensor 作为输出
- 保持 Tensor 的 device 和 dtype 属性一致

**Validates: Requirements 8.1, 8.2**

### Property 11: 批量和多头支持

*For any* batch_size ≥ 1 和 num_heads ≥ 1 的输入配置，Attention_Kernel 应正确处理并产生形状为 [batch_size, num_heads, seq_len, head_dim] 的输出。

**Validates: Requirements 8.4**

### Property 12: 任意形状矩阵支持

*For any* 满足对齐约束的矩阵形状 (M, N, K)，GEMM_Kernel 应正确计算并产生形状为 [M, N] 的输出。

**Validates: Requirements 8.5**

### Property 13: 无效输入错误处理

*For any* 无效输入（如维度不匹配、空张量、非法 dtype），kernel 应抛出明确的错误信息而非产生未定义行为或崩溃。

**Validates: Requirements 8.3**

## Error Handling

### 输入验证错误

```cpp
enum class KernelError {
    SUCCESS = 0,
    INVALID_DIMENSION,        // 维度不匹配
    UNSUPPORTED_DTYPE,        // 不支持的数据类型
    ALIGNMENT_ERROR,          // 对齐要求不满足
    OUT_OF_MEMORY,            // 显存不足
    CUDA_ERROR,               // CUDA 运行时错误
    INVALID_CONFIG            // 无效的配置参数
};

// 错误检查宏 (包含文件名和行号信息)
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + \
            cudaGetErrorString(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)
```

### Python 层错误处理

输入验证在 C++ 绑定层 (`bindings.cpp`) 中通过 `TORCH_CHECK` 宏实现，抛出 `c10::Error` 异常（Python 中表现为 `RuntimeError`）。

```cpp
// python/bindings.cpp
void validate_attention_inputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
    TORCH_CHECK(q.dim() == 4, "Q must be 4D tensor [batch, heads, seq_len, head_dim]");
    TORCH_CHECK(k.dim() == 4, "K must be 4D tensor");
    TORCH_CHECK(v.dim() == 4, "V must be 4D tensor");
    TORCH_CHECK(q.sizes() == k.sizes(), "Q and K must have same shape");
    TORCH_CHECK(k.sizes() == v.sizes(), "K and V must have same shape");
    TORCH_CHECK(q.is_cuda(), "Q must be on CUDA device");
    TORCH_CHECK(q.is_contiguous(), "Q must be contiguous");
    auto dtype = q.scalar_type();
    TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kFloat16,
                "Only float32 and float16 are supported");
}

void validate_gemm_inputs(const torch::Tensor& a, const torch::Tensor& b, bool trans_a, bool trans_b) {
    TORCH_CHECK(a.dim() == 2, "A must be 2D tensor [M, K]");
    TORCH_CHECK(b.dim() == 2, "B must be 2D tensor [K, N]");
    auto a_cols = trans_a ? a.size(0) : a.size(1);
    auto b_rows = trans_b ? b.size(1) : b.size(0);
    TORCH_CHECK(a_cols == b_rows, "Inner dimensions must match");
}
```

## Testing Strategy

### 测试框架

- **单元测试**: pytest + CUDA 测试环境
- **属性测试**: Hypothesis (Python PBT 库)
- **性能测试**: Nsight Compute + 自定义 benchmark 脚本

### 属性测试配置

每个属性测试运行至少 100 次迭代，使用 Hypothesis 生成随机输入。

```python
from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as npst

# 矩阵维度策略
dim_strategy = st.integers(min_value=16, max_value=512)
batch_strategy = st.integers(min_value=1, max_value=8)
head_strategy = st.integers(min_value=1, max_value=16)

@settings(max_examples=100)
@given(
    batch=batch_strategy,
    heads=head_strategy,
    seq_len=dim_strategy,
    head_dim=st.sampled_from([32, 64, 128])
)
def test_attention_correctness(batch, heads, seq_len, head_dim):
    """
    Feature: cuda-llm-kernel-optimization
    Property 1: Attention 计算正确性
    Validates: Requirements 1.1, 1.3, 1.4, 1.5
    """
    # 生成随机输入
    q = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
    k = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
    v = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
    
    # 计算输出
    output = flash_attention(q, k, v)
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    
    # 验证误差
    assert torch.allclose(output, reference, rtol=1e-3, atol=1e-3)
```

### 单元测试覆盖

单元测试用于验证特定示例和边界情况：

1. **边界情况测试**:
   - 最小维度 (seq_len=1, head_dim=1)
   - 最大支持维度
   - 非对齐维度

2. **精度测试**:
   - FP32 精度验证
   - FP16 精度验证
   - INT8 量化精度验证

3. **错误处理测试**:
   - 无效输入维度
   - 不支持的数据类型
   - 显存不足场景

### 性能基准测试

```python
def benchmark_attention(seq_lengths=[512, 1024, 2048, 4096]):
    """与 cuDNN/PyTorch 对比的性能基准"""
    results = []
    for seq_len in seq_lengths:
        # 测量自定义实现
        custom_time = measure_kernel_time(flash_attention, q, k, v)
        # 测量参考实现
        ref_time = measure_kernel_time(torch.nn.functional.scaled_dot_product_attention, q, k, v)
        results.append({
            'seq_len': seq_len,
            'custom_ms': custom_time,
            'reference_ms': ref_time,
            'speedup': ref_time / custom_time
        })
    return results
```

### 测试矩阵

| 测试类型 | 覆盖范围 | 工具 |
|---------|---------|------|
| 属性测试 | 正确性属性 1-13 | Hypothesis |
| 单元测试 | 边界情况、错误处理 | pytest |
| 性能测试 | 吞吐量、延迟 | Nsight Compute |
| 集成测试 | Python 接口 | pytest |

### 测试环境要求

- **硬件**: NVIDIA GPU (Volta+), 至少 8GB 显存
- **软件**: CUDA 11.0+, PyTorch 2.0+, Python 3.8+
- **依赖**: pytest, hypothesis, numpy, pybind11

### 持续集成

- 每次提交运行单元测试和属性测试
- 性能回归测试在合并前运行
- 测试覆盖率目标: 核心逻辑 90%+



## 已知限制

1. **无反向传播**: FlashAttention 仅实现了 forward kernel，未实现 backward，因此不支持 `torch.autograd` 训练场景
2. **BF16 未实现**: `Precision` 枚举声明了 BF16 但无实际 kernel 实现
3. **流水线深度固定**: FlashAttention 和 hgemm 均使用固定 2 级 double buffering，未支持 3/4 级可配置流水线
4. **Tensor Core GEMM 输出精度**: `tensor_core_gemm` 固定 FP16 输入 FP32 输出，不支持 FP16 输出

## 参考资料

1. **FlashAttention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
2. **FlashAttention-2**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023
3. **CUTLASS**: NVIDIA CUTLASS - CUDA Templates for Linear Algebra Subroutines
4. **cuBLAS**: NVIDIA cuBLAS Library Documentation
5. **CUDA Programming Guide**: NVIDIA CUDA C++ Programming Guide

## 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0 | 2026-01-01 | 初始设计，包含完整架构和正确性属性定义 |
| 1.1 | 2026-02-27 | 对齐实际代码：修正所有 kernel 签名、结构体、共享内存布局、Python 接口描述；添加已知限制章节 |
