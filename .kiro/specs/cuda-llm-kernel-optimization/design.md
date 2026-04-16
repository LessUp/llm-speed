# Design Document

## Overview

CUDA kernel optimization library for LLM inference with FlashAttention and high-performance GEMM implementations.

### Design Goals

| Goal | Target |
|------|--------|
| Performance | GEMM ≥90% cuBLAS, FlashAttention O(N) memory |
| Precision | FP32, FP16, INT8 mixed precision |
| Scalability | Volta, Ampere, Hopper architectures |
| Usability | PyTorch-compatible Python interface |
| Correctness | Numerical error within tolerance vs reference |

### Constraints

| Constraint | Value |
|------------|-------|
| CUDA Compute | SM70+ (Volta minimum) |
| Shared Memory | 48KB (Volta) / 164KB (Ampere) per SM |
| Registers | 255 per thread maximum |
| Tensor Core Alignment | Dimensions multiple of 16 |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Python Interface Layer                       │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │
│  │ flash_attention│  │  gemm_kernel   │  │   profiler     │     │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘     │
└──────────┼───────────────────┼───────────────────┼───────────────┘
           │                   │                   │
┌──────────┼───────────────────┼───────────────────┼───────────────┐
│          ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    CUDA Kernel Layer                         │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │ │
│  │  │Attention     │  │    GEMM      │  │    Warp      │       │ │
│  │  │  Kernels     │  │   Kernels    │  │ Primitives   │       │ │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │ │
│  └─────────┼─────────────────┼─────────────────┼────────────────┘ │
│            │                 │                 │                  │
│  ┌─────────┼─────────────────┼─────────────────┼────────────────┐ │
│  │         ▼                 ▼                 ▼                │ │
│  │  ┌───────────────────────────────────────────────────────┐   │ │
│  │  │               Optimization Components                  │   │ │
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐         │   │ │
│  │  │  │   Tiling   │ │Tensor Core │ │  Pipeline  │         │   │ │
│  │  │  │  Manager   │ │Accelerator │ │ Scheduler  │         │   │ │
│  │  │  └────────────┘ └────────────┘ └────────────┘         │   │ │
│  │  └───────────────────────────────────────────────────────┘   │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                         CUDA Runtime                               │
└───────────────────────────────────────────────────────────────────┘
```

---

## Kernel Specifications

### 1. Naive Attention

Baseline implementation for correctness verification.

```cpp
// src/naive_attention.cu
template<typename T>
__global__ void naive_attention_simple_kernel(
    const T* __restrict__ Q,   // [batch, heads, seq_len, head_dim]
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale
);
```

**Algorithm:**
1. `S = Q @ K^T * scale` — attention scores
2. `P = softmax(S, dim=-1)` — attention weights
3. `O = P @ V` — output

**Resources:**
- Grid: `(1, seq_len, batch * heads)`
- Block: 256 threads
- Shared memory: `(seq_len + num_warps) * sizeof(float)`

---

### 2. Tiled Attention

Shared memory tiling optimization.

```cpp
// src/tiled_attention.cu
template<typename T, int BLOCK_M, int BLOCK_N>
__global__ void tiled_attention_kernel(
    const T* __restrict__ Q, const T* __restrict__ K,
    const T* __restrict__ V, T* __restrict__ O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale
);
```

**Configuration:**
- `BLOCK_M = 32` — Q row tile size
- `BLOCK_N = 32` — K/V row tile size

**Shared Memory Layout (+1 padding for bank conflict):**
```
smem_Q  [BLOCK_M * (head_dim+1)]   — Q tile
smem_K  [BLOCK_N * (head_dim+1)]   — K tile
smem_V  [BLOCK_N * (head_dim+1)]   — V tile
smem_S  [BLOCK_M * (BLOCK_N+1)]    — scores
output  [BLOCK_M * head_dim]       — accumulator
```

---

### 3. FlashAttention

O(N) memory with online softmax and double buffering.

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
    bool is_causal
);
```

**Online Softmax State:**
```cpp
struct OnlineSoftmaxState {
    float max_val;   // running maximum m_i
    float sum_exp;   // running exp sum l_i
};
```

**Shared Memory Layout (double buffered):**
```
smem_Q    [BLOCK_M * (head_dim+1)]       — Q tile
smem_K[2] [2 * BLOCK_N * (head_dim+1)]   — K double buffer
smem_V[2] [2 * BLOCK_N * (head_dim+1)]   — V double buffer
smem_S    [BLOCK_M * (BLOCK_N+1)]        — scores
output    [BLOCK_M * head_dim]           — output accumulator
row_max   [BLOCK_M]                      — running max per row
row_sum   [BLOCK_M]                      — running sum per row
rescale   [BLOCK_M]                      — rescale factor
```

**Two-Phase Algorithm:**
- Phase 1: Single thread per row computes softmax state (light)
- Phase 2: All threads cooperate on output update (heavy, parallel)

---

### 4. Tensor Core GEMM

WMMA-based matrix multiplication.

```cpp
// src/tensor_core_gemm.cu

// Basic version
__global__ void tensor_core_gemm_fp16_kernel(
    const half* A, const half* B, float* C,
    int M, int N, int K
);

// Tiled version with shared memory
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void tensor_core_gemm_tiled_kernel(
    const half* A, const half* B, float* C,
    int M, int N, int K, float alpha, float beta
);

// INT8 version (Turing+ SM≥7.2)
__global__ void tensor_core_gemm_int8_kernel(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K
);
```

**WMMA Dimensions:**
- FP16: 16×16×16
- INT8: 8×32×16

**Configuration:**
- `BLOCK_M=64, BLOCK_N=64, BLOCK_K=32`
- 4 warps per block
- Shared memory padding: `+8` half elements

---

### 5. High-Performance GEMM

Register tiling with double buffering.

```cpp
// src/hgemm_kernel.cu
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K,
         int WARP_M, int WARP_N, int THREAD_M, int THREAD_N>
__global__ void hgemm_register_tiled_kernel(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    float alpha, float beta,
    bool trans_a, bool trans_b
);
```

**Three-Level Tiling:**
```
Block:  BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
Warp:   WARP_M=32,   WARP_N=64
Thread: THREAD_M=8,  THREAD_N=8
```

**Register Allocation:**
```cpp
float reg_C[THREAD_M][THREAD_N];  // output tile per thread
float reg_A[THREAD_M];
float reg_B[THREAD_N];
```

**Double Buffering:**
```cpp
__shared__ float smem_A[2][BLOCK_M][BLOCK_K+1];
__shared__ float smem_B[2][BLOCK_K][BLOCK_N+1];
```

---

## Header Primitives

### common.cuh

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
    Precision precision;
};

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error( \
            std::string("CUDA error: ") + cudaGetErrorString(err) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)
```

### warp_primitives.cuh

```cpp
template<typename T> __device__ T warp_reduce_sum(T val);
template<typename T> __device__ T warp_reduce_max(T val);
template<typename T> __device__ T warp_reduce_min(T val);
template<typename T, int BLOCK_SIZE> __device__ T block_reduce_sum(T val, T* smem);
template<typename T, int BLOCK_SIZE> __device__ T block_reduce_max(T val, T* smem);
template<typename T> __device__ T warp_broadcast(T val, int src_lane);
```

### pipeline.cuh

```cpp
template<typename T, int BLOCK_SIZE>
struct DoubleBuffer {
    T* buffer[2];
    int current;
    __device__ T* get_load_buffer();
    __device__ T* get_compute_buffer();
    __device__ void swap();
};

template<int BYTES> __device__ void async_copy(void* dst, const void* src);
__device__ void async_copy_commit();
template<int N> __device__ void async_copy_wait();
```

---

## Python Interface

```cpp
// python/bindings.cpp
PYBIND11_MODULE(cuda_llm_ops, m) {
    m.def("naive_attention", &naive_attention,
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale") = 0.0f);
    m.def("tiled_attention", &tiled_attention,
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale") = 0.0f);
    m.def("flash_attention", &flash_attention,
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("scale") = 0.0f, py::arg("is_causal") = false);
    m.def("gemm", &gemm,
          py::arg("a"), py::arg("b"),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("trans_a") = false, py::arg("trans_b") = false);
    m.def("tensor_core_gemm", &tensor_core_gemm,
          py::arg("a"), py::arg("b"),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);
    m.def("tensor_core_gemm_int8", &tensor_core_gemm_int8_wrapper,
          py::arg("a"), py::arg("b"));
}
```

---

## Known Limitations

| Limitation | Description |
|------------|-------------|
| No backward pass | FlashAttention forward only, no autograd support |
| BF16 unimplemented | Enum declared, no kernel implementation |
| Fixed pipeline depth | 2-stage double buffering only |
| INT8 SM requirement | Requires Turing+ (SM≥7.2) |

---

## References

1. FlashAttention: Dao et al., NeurIPS 2022
2. FlashAttention-2: Dao, 2023
3. CUTLASS: NVIDIA CUDA Templates
4. CUDA Programming Guide: NVIDIA

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-01 | Initial design |
| 1.1 | 2025-02-27 | Synced with implementation |
| 1.2 | 2026-04-16 | Restructured with tables |
