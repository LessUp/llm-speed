---
layout: docs
title: Architecture Design
description: In-depth technical documentation covering architecture, algorithm principles, and optimization strategies
lang: en
---

# Architecture Design

In-depth technical documentation covering the architecture, algorithm principles, and optimization strategies of LLM-Speed.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Attention Kernels](#attention-kernels)
- [GEMM Kernels](#gemm-kernels)
- [Header Primitives Library](#header-primitives-library)
- [Python Bindings](#python-bindings)
- [Performance Optimization Techniques](#performance-optimization-techniques)
- [Testing Strategy](#testing-strategy)

---

## Project Overview

LLM-Speed is a high-performance CUDA operator library specifically designed for LLM inference. It employs a progressive optimization strategy:

```
Naive → Tiled → FlashAttention → Tensor Core
```

### Core Objectives

| Objective | Target |
|-----------|--------|
| GEMM Performance | ≥90% of cuBLAS |
| FlashAttention Memory | O(N) complexity |
| Pipeline Improvement | ≥20% performance gain |
| Precision Support | FP32/FP16/INT8 |

### Optimization Philosophy

We follow the principle of correct first, then fast:

1. **Correctness**: Baseline implementation verified against PyTorch reference
2. **Optimization**: Incremental improvements with measurable gains
3. **Hardware Utilization**: Leverage Tensor Cores and memory hierarchy
4. **Production Ready**: Comprehensive error handling and input validation

---

## System Architecture

### Three-Layer Architecture

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

### Optimization Roadmap

```
                    ┌─────────────────┐
                    │  Naive Kernel   │
                    │  O(N²) memory   │
                    └────────┬────────┘
                             │ Shared memory tiling
                    ┌────────▼────────┐
                    │  Tiled Kernel   │
                    │  Reduced global │
                    │  memory access  │
                    └────────┬────────┘
                             │ Online softmax
                    ┌────────▼────────┐
                    │ FlashAttention  │
                    │   O(N) memory   │
                    └────────┬────────┘
                             │ Double buffering
                    ┌────────▼────────┐
                    │ Optimized Flash │
                    │ Compute/memory  │
                    │ overlap         │
                    └─────────────────┘
```

---

## Attention Kernels

### 1. Naive Attention

Baseline implementation for correctness verification and performance comparison.

**Algorithm:**
```
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

**Computational Flow:**
1. `S = Q @ K^T * scale` → `[seq_len, seq_len]`
2. `P = softmax(S, dim=-1)` → `[seq_len, seq_len]`
3. `O = P @ V` → `[seq_len, head_dim]`

**Key Implementation Details:**

```cpp
// Each block processes one (batch, head, row)
__global__ void naive_attention_simple_kernel(
    const T* Q, const T* K, const T* V, T* O,
    int batch_size, int num_heads, int seq_len, int head_dim, float scale
) {
    // Shared memory for attention scores
    extern __shared__ float shared_mem[];
    float* scores = shared_mem;

    // Warp reduction for softmax
    float reduced_max = block_reduce_max<float, 256>(local_max, reduce_smem);
    float reduced_sum = block_reduce_sum<float, 256>(local_sum, reduce_smem);
}
```

**Complexity Analysis:**
- Time: O(N² × d)
- Memory: O(N²)

**Use Cases:**
- Correctness verification against reference
- Short sequences (N <= 64)
- Understanding baseline behavior

---

### 2. Tiled Attention

Shared memory tiling reduces global memory access.

**Tiling Configuration:**
```cpp
BLOCK_M = 32  // Q row tile size
BLOCK_N = 32  // K/V row tile size
```

**Shared Memory Layout:**
```
┌────────────────────────────────────────────┐
│ smem_Q  [BLOCK_M × (head_dim+1)]           │ ← Q tile with padding
├────────────────────────────────────────────┤
│ smem_K  [BLOCK_N × (head_dim+1)]           │ ← K tile with padding
├────────────────────────────────────────────┤
│ smem_V  [BLOCK_N × (head_dim+1)]           │ ← V tile with padding
├────────────────────────────────────────────┤
│ smem_S  [BLOCK_M × (BLOCK_N+1)]            │ ← attention scores
├────────────────────────────────────────────┤
│ output  [BLOCK_M × head_dim]               │ ← output accumulator
└────────────────────────────────────────────┘

Note: +1 padding eliminates bank conflict
```

**Performance Gains:**
- Reduced global memory traffic by ~75%
- Better cache utilization
- Suitable for sequences 128-2048

---

### 3. FlashAttention

**Core Innovation:** Avoid storing the N×N attention matrix, achieving O(N) memory complexity.

**Online Softmax Formula:**
```
For each tile t:
    S_t = Q_tile @ K_tile^T * scale
    m_t = max(m_{t-1}, row_max(S_t))

    // Rescaling
    scale_factor = exp(m_{t-1} - m_t)
    l_t = l_{t-1} * scale_factor + sum(exp(S_t - m_t))

    // Output update
    O_t = O_{t-1} * scale_factor + exp(S_t - m_t) @ V_tile

Final: O = O_T / l_T
```

**State Maintenance:**
```cpp
float row_max[BLOCK_M];   // Current row maximum m_i
float row_sum[BLOCK_M];   // Current row exponential sum l_i
float rescale[BLOCK_M];   // Per-row rescaling factor
```

**Double Buffering Implementation:**
```cpp
// Shared memory layout (K/V double buffer)
smem_Q    [BLOCK_M × (head_dim+1)]      — Q tile
smem_K[2] [2 × BLOCK_N × (head_dim+1)]  — K double buffer
smem_V[2] [2 × BLOCK_N × (head_dim+1)]  — V double buffer
smem_S    [BLOCK_M × (BLOCK_N+1)]       — attention scores
output    [BLOCK_M × head_dim]          — output accumulator

// Pipeline flow
// Prologue: Load first K/V tile to buffer 0
// Main loop: Compute current buffer, prefetch next to alternating buffer
// Causal early-exit: Skip prefetch when next tile exceeds causal window
```

**Two-Phase Computation:**
```cpp
// Phase 1: Single thread per row computes softmax state (lightweight)
if (tid < BLOCK_M) {
    // Compute rowmax(scores)
    // Update max/sum state
    // Compute rescale factor
}

// Phase 2: All threads cooperate for output update (heavyweight)
for (int i = tid; i < BLOCK_M * head_dim; i += blockDim.x) {
    // Rescale old output
    // Compute new contribution: exp_scores @ V
    // Update output
}
```

**Causal Mask:**
```cpp
if (is_causal && global_col > global_row) {
    score = -FLT_MAX;  // Mask future positions
}

// Early-exit optimization: Break when col_start >= row_start + BLOCK_M
if (is_causal && col_start >= row_start + BLOCK_M) break;
```

**Performance:**
- Memory: O(N) vs O(N²) naive
- Throughput: 2-4x faster for long sequences
- Scales to 100K+ sequence lengths

---

## GEMM Kernels

### 1. Tensor Core GEMM

Uses WMMA API to leverage Tensor Core hardware acceleration.

**WMMA Fragments:**
```cpp
#include <mma.h>
using namespace nvcuda;

// 16×16×16 matrix tiles
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

// Load → Compute → Store
wmma::load_matrix_sync(a_frag, A + offset, K);
wmma::load_matrix_sync(b_frag, B + offset, N);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(C + offset, c_frag, N, wmma::mem_row_major);
```

**Tiled Version:**
```cpp
// Shared memory tiles with padding
__shared__ half smem_A[BLOCK_M][BLOCK_K + 8];  // +8 half padding
__shared__ half smem_B[BLOCK_K][BLOCK_N + 8];

// Multi-warp cooperation
constexpr int WARPS_M = BLOCK_M / WMMA_M;  // 4 warps
constexpr int WARPS_N = BLOCK_N / WMMA_N;  // 4 warps
```

**INT8 Support (Turing+ SM≥7.2):**
```cpp
// INT8 WMMA dimensions: 8×32×16
wmma::fragment<wmma::matrix_a, 8, 32, 16, int8_t, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 8, 32, 16, int8_t, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 8, 32, 16, int32_t> c_frag;
```

---

### 2. High-Performance GEMM (Register Tiling)

**Three-Level Tiling Strategy:**
```
Block level: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
Warp level:  WARP_M=32,   WARP_N=64
Thread level: THREAD_M=8,  THREAD_N=8
```

**Register Tiling:**
```cpp
// Each thread holds THREAD_M × THREAD_N output tile
float reg_C[THREAD_M][THREAD_N] = {0};
float reg_A[THREAD_M];
float reg_B[THREAD_N];

// Outer product algorithm
for (int k = 0; k < BLOCK_K; k++) {
    // Load A/B elements to registers
    for (int m = 0; m < THREAD_M; m++)
        reg_A[m] = smem_A[warp_row + thread_row + m][k];
    for (int n = 0; n < THREAD_N; n++)
        reg_B[n] = smem_B[k][warp_col + thread_col + n];

    // Register-level matrix multiplication
    for (int m = 0; m < THREAD_M; m++)
        for (int n = 0; n < THREAD_N; n++)
            reg_C[m][n] += reg_A[m] * reg_B[n];
}
```

**Double Buffering:**
```cpp
__shared__ float smem_A[2][BLOCK_M][BLOCK_K + 1];
__shared__ float smem_B[2][BLOCK_K][BLOCK_N + 1];

// Main loop: Compute current buffer, prefetch next to alternating buffer
for (int tile = 0; tile < num_k_tiles; tile++) {
    int cur_buf = tile % 2;
    int next_buf = 1 - cur_buf;

    // Prefetch next tile
    if (tile + 1 < num_k_tiles) {
        LOAD_TILE_A(next_buf, next_k_tile);
        LOAD_TILE_B(next_buf, next_k_tile);
    }

    // Compute current tile
    COMPUTE_TILE(cur_buf);
    __syncthreads();
}
```

**Performance Target:** ≥90% of cuBLAS for matrices ≥1024×1024

---

## Header Primitives Library

### common.cuh

**Core Types:**
```cpp
enum class Precision { FP32, FP16, BF16, INT8 };
enum class MatrixLayout { RowMajor, ColMajor, RowMajorPadded };

struct AttentionConfig {
    int batch_size, num_heads, seq_len, head_dim;
    float scale;
    bool is_causal;
    int block_m, block_n;
    Precision precision;
};

struct GemmConfig {
    int M, N, K;
    float alpha, beta;
    MatrixLayout layout_a, layout_b;
    int block_m, block_n, block_k;
    int warp_m, warp_n;
    int thread_m, thread_n;
    Precision precision;
};
```

**Utility Macros:**
```cpp
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + \
            cudaGetErrorString(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)

inline int div_ceil(int a, int b) { return (a + b - 1) / b; }
inline bool is_tensor_core_aligned(int dim, int alignment = 16) { return (dim % alignment) == 0; }
```

### warp_primitives.cuh

**Warp-Level Reduction:**
```cpp
template<typename T>
__device__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template<typename T>
__device__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}
```

**Block-Level Reduction:**
```cpp
template<typename T, int BLOCK_SIZE>
__device__ T block_reduce_sum(T val, T* smem) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Warp-level reduction
    val = warp_reduce_sum(val);

    // Write to shared memory
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    // First warp does final reduction
    constexpr int num_warps = BLOCK_SIZE / 32;
    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : T(0);
        val = warp_reduce_sum(val);
    }
    return val;
}
```

### online_softmax.cuh

**Online Softmax State:**
```cpp
struct OnlineSoftmaxState {
    float max_val;  // Current maximum m_i
    float sum_exp;  // Current exponential sum l_i
};
```

**State Update:**
```cpp
__device__ void online_softmax_update(
    OnlineSoftmaxState& state, float new_val
) {
    float new_max = fmaxf(state.max_val, new_val);
    float old_scale = expf(state.max_val - new_max);
    float new_scale = expf(new_val - new_max);

    state.sum_exp = state.sum_exp * old_scale + new_scale;
    state.max_val = new_max;
}
```

---

## Python Bindings

### Interface Design

```cpp
// cuda_llm_ops/bindings.cpp
PYBIND11_MODULE(cuda_llm_ops, m) {
    m.doc() = "LLM-Speed";

    // Attention functions
    m.def("naive_attention", &naive_attention,
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale") = 0.0f,
          "Naive attention (baseline)");

    m.def("tiled_attention", &tiled_attention,
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale") = 0.0f,
          "Tiled attention with shared memory");

    m.def("flash_attention", &flash_attention,
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("scale") = 0.0f, py::arg("is_causal") = false,
          "FlashAttention with online softmax");

    // GEMM functions
    m.def("gemm", &gemm,
          py::arg("a"), py::arg("b"),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("trans_a") = false, py::arg("trans_b") = false,
          "High-performance GEMM");

    m.def("tensor_core_gemm", &tensor_core_gemm,
          py::arg("a"), py::arg("b"),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          "Tensor Core GEMM (FP16 in, FP32 out)");

    m.def("tensor_core_gemm_int8", &tensor_core_gemm_int8_wrapper,
          py::arg("a"), py::arg("b"),
          "INT8 Tensor Core GEMM (SM>=7.2 required)");
}
```

### Input Validation

```cpp
void validate_attention_inputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
    TORCH_CHECK(q.dim() == 4, "Q must be 4D tensor [batch, heads, seq_len, head_dim]");
    TORCH_CHECK(q.sizes() == k.sizes(), "Q and K must have same shape");
    TORCH_CHECK(q.is_cuda(), "Q must be on CUDA device");
    TORCH_CHECK(q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32 || q.scalar_type() == torch::kFloat16,
                "Only float32 and float16 are supported");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0 && q.size(3) > 0,
                "Tensor dimensions must be positive");
}
```

---

## Performance Optimization Techniques

### Technique Summary

| Technique | Target | Implementation |
|-----------|--------|----------------|
| Shared Memory Tiling | Reduce global memory access | tiled_attention, hgemm |
| Bank Conflict Avoidance | +1 padding | shared_memory.cuh |
| Online Softmax | O(N) memory | flash_attention |
| Warp Shuffle | Fast reduction | warp_primitives.cuh |
| Register Tiling | Data reuse | hgemm_kernel |
| Tensor Core | Hardware acceleration | tensor_core_gemm |
| Double Buffering | Hide latency | pipeline.cuh |
| Async Copy | Compute/transfer overlap | pipeline.cuh (Ampere+) |

### Bottleneck Detection

```python
compute_intensity = flops / memory_bytes  # FLOPs/byte
if compute_intensity > 100:
    bottleneck = "COMPUTE_BOUND"
else:
    bottleneck = "MEMORY_BOUND"
```

### Optimization Checklist

- [x] Aligned dimensions (multiples of 16 for Tensor Core)
- [x] Bank conflict free shared memory layout
- [x] Warp shuffle for reduction operations
- [x] Double buffering for pipeline optimization
- [x] Loop unrolling (compiler hints)
- [x] Async copy for Ampere+ (optional)

---

## Testing Strategy

### Property-Based Testing

Using Hypothesis for comprehensive correctness validation:

```python
@pytest.mark.cuda
@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    batch=st.integers(1, 4),
    heads=st.integers(1, 8),
    seq_len=st.integers(16, 256),
    head_dim=st.sampled_from([32, 64, 128])
)
def test_flash_attention_correctness(batch, heads, seq_len, head_dim, device):
    q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output = flash_attention(q, k, v)
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert_close(output, reference, rtol=1e-3, atol=1e-3)
```

### Test Coverage

| Category | Content |
|----------|---------|
| Correctness | Comparison with PyTorch reference |
| Numerical Stability | FP16/FP32 precision validation |
| Boundary Conditions | Minimum dimensions, large sequences, misaligned |
| Layout Equivalence | NN/NT/TN/TT matrix layouts |
| Error Handling | Dimension mismatch, dtype errors, empty tensors |

---

## References

1. **FlashAttention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
2. **FlashAttention-2**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023
3. **CUTLASS**: NVIDIA CUTLASS - CUDA Templates for Linear Algebra Subroutines
4. **cuBLAS**: NVIDIA cuBLAS Library Documentation
5. **CUDA Programming Guide**: NVIDIA CUDA C++ Programming Guide

[← Back to Documentation](../)
