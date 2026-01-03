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
│  │  │ Attention     │  │ GEMM          │  │ Softmax       │   │ │
│  │  │ Kernels       │  │ Kernels       │  │ Kernels       │   │ │
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
// naive_attention.cuh
template<typename T>
__global__ void naive_attention_kernel(
    const T* Q,           // [batch, heads, seq_len, head_dim]
    const T* K,           // [batch, heads, seq_len, head_dim]
    const T* V,           // [batch, heads, seq_len, head_dim]
    T* O,                 // [batch, heads, seq_len, head_dim]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale           // 1/sqrt(head_dim)
);

// 计算流程:
// 1. S = Q @ K^T * scale    [seq_len, seq_len]
// 2. P = softmax(S, dim=-1) [seq_len, seq_len]
// 3. O = P @ V              [seq_len, head_dim]
```

### 2. Tiled Attention Kernel

使用共享内存分块优化的版本。

```cpp
// tiled_attention.cuh
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void tiled_attention_kernel(
    const T* Q, const T* K, const T* V, T* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale
);

// Tiling 策略:
// - BLOCK_M: Q 的行分块大小 (典型值: 64, 128)
// - BLOCK_N: K/V 的行分块大小 (典型值: 64, 128)
// - BLOCK_K: head_dim 分块大小 (典型值: 32, 64)

// 共享内存布局 (带 padding 消除 bank conflict):
// __shared__ T smem_Q[BLOCK_M][BLOCK_K + 1];
// __shared__ T smem_K[BLOCK_N][BLOCK_K + 1];
// __shared__ T smem_V[BLOCK_N][BLOCK_K + 1];
```

### 3. FlashAttention Engine

实现 Online Softmax 的核心算法。

```cpp
// flash_attention.cuh
template<typename T, int BLOCK_M, int BLOCK_N>
__global__ void flash_attention_forward_kernel(
    const T* Q, const T* K, const T* V, T* O,
    float* L,             // logsumexp for backward
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale,
    bool is_causal        // 是否使用因果掩码
);

// Online Softmax 状态:
struct OnlineSoftmaxState {
    float max_val;        // 当前最大值 m_i
    float sum_exp;        // 当前指数和 l_i
    float* output;        // 累积输出 O_i
};

// 算法核心 (单个 tile 更新):
// 1. 计算当前 tile 的 S_ij = Q_i @ K_j^T * scale
// 2. 更新 max: m_new = max(m_old, rowmax(S_ij))
// 3. 更新 sum: l_new = exp(m_old - m_new) * l_old + rowsum(exp(S_ij - m_new))
// 4. 更新 output: O_new = exp(m_old - m_new) * O_old + exp(S_ij - m_new) @ V_j
// 5. 最终归一化: O = O / l
```

### 4. Tensor Core GEMM Kernel

使用 WMMA/MMA 指令的高性能矩阵乘法。

```cpp
// tensor_core_gemm.cuh
#include <mma.h>
using namespace nvcuda::wmma;

template<int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void tensor_core_gemm_kernel(
    const half* A,        // [M, K]
    const half* B,        // [K, N]
    float* C,             // [M, N]
    int M, int N, int K
);

// WMMA Fragment 定义:
// fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
// fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
// fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

// 典型配置 (Ampere):
// WMMA_M = 16, WMMA_N = 16, WMMA_K = 16
```

### 5. High-Performance GEMM with Register Tiling

```cpp
// hgemm_kernel.cuh
template<
    typename T,
    int BLOCK_M,          // Thread block tile M
    int BLOCK_N,          // Thread block tile N
    int BLOCK_K,          // Thread block tile K
    int WARP_M,           // Warp tile M
    int WARP_N,           // Warp tile N
    int THREAD_M,         // Thread tile M (register)
    int THREAD_N          // Thread tile N (register)
>
__global__ void hgemm_kernel(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    float alpha, float beta
);

// Register Tiling 布局:
// 每个线程持有 THREAD_M x THREAD_N 的输出 tile
// 寄存器数组: T reg_C[THREAD_M][THREAD_N];
//            T reg_A[THREAD_M];
//            T reg_B[THREAD_N];
```

### 6. Pipeline Scheduler

```cpp
// pipeline.cuh
template<int NUM_STAGES>
class PipelineScheduler {
public:
    // Double/Multi-buffering 状态
    struct StageBuffer {
        void* smem_ptr;
        bool ready;
        cudaEvent_t event;
    };
    
    StageBuffer buffers[NUM_STAGES];
    int current_stage;
    
    __device__ void prefetch_next_tile(const void* global_ptr, int tile_idx);
    __device__ void wait_for_stage(int stage);
    __device__ void* get_buffer(int stage);
};

// 流水线执行模式:
// Stage 0: Load tile[i+2] from global memory
// Stage 1: Compute on tile[i] from shared memory
// Stage 2: Store results of tile[i-1]
```

### 7. Warp-Level Primitives

```cpp
// warp_primitives.cuh

// Warp Shuffle 规约
template<typename T>
__device__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T>
__device__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level 规约 (使用共享内存)
template<typename T, int BLOCK_SIZE>
__device__ T block_reduce_sum(T val, T* smem);
```

### 8. Python Interface

```python
# python/flash_attention.py
import torch
from torch.utils.cpp_extension import load

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale=None, is_causal=False):
        """
        Args:
            q: [batch, heads, seq_len, head_dim]
            k: [batch, heads, seq_len, head_dim]
            v: [batch, heads, seq_len, head_dim]
            scale: float, default 1/sqrt(head_dim)
            is_causal: bool, whether to apply causal mask
        Returns:
            output: [batch, heads, seq_len, head_dim]
        """
        pass
    
    @staticmethod
    def backward(ctx, grad_output):
        pass

# python/gemm.py
def gemm(a: torch.Tensor, b: torch.Tensor, 
         alpha: float = 1.0, beta: float = 0.0,
         precision: str = 'fp16') -> torch.Tensor:
    """
    High-performance GEMM: C = alpha * A @ B + beta * C
    
    Args:
        a: [M, K] tensor
        b: [K, N] tensor
        precision: 'fp32', 'fp16', or 'int8'
    """
    pass
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

// 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while(0)
```

### Python 层错误处理

```python
class KernelError(Exception):
    """CUDA kernel 执行错误"""
    pass

class DimensionError(KernelError):
    """输入维度不匹配"""
    pass

class AlignmentError(KernelError):
    """对齐要求不满足"""
    pass

def validate_attention_inputs(q, k, v):
    """验证 attention 输入参数"""
    if q.dim() != 4:
        raise DimensionError(f"Expected 4D tensor, got {q.dim()}D")
    if q.shape != k.shape or k.shape != v.shape:
        raise DimensionError(f"Q, K, V shapes must match")
    if q.dtype not in [torch.float32, torch.float16]:
        raise TypeError(f"Unsupported dtype: {q.dtype}")
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
