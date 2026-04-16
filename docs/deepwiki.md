# CUDA LLM 内核优化项目 - 技术深潜

本文档详细介绍 CUDA 内核优化技术栈，包括算法原理、实现细节和优化策略。

## 目录

- [项目概述](#项目概述)
- [架构设计](#架构设计)
- [Attention 内核](#attention-内核)
- [GEMM 内核](#gemm-内核)
- [头文件原语库](#头文件原语库)
- [Python 绑定](#python-绑定)
- [性能优化技术](#性能优化技术)
- [测试策略](#测试策略)

---

## 项目概述

高性能 CUDA 算子库，专为 LLM 推理优化设计。采用渐进式优化策略：

```
Naive → Tiled → FlashAttention → Tensor Core
```

### 核心目标

| 目标 | 指标 |
|------|------|
| GEMM 性能 | ≥90% cuBLAS |
| FlashAttention 显存 | O(N) 复杂度 |
| 流水线提升 | ≥20% 性能提升 |
| 精度支持 | FP32/FP16/BF16/INT8 |

---

## 架构设计

### 系统架构

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

### 优化路线

```
                    ┌─────────────────┐
                    │  Naive Kernel   │
                    │  O(N²) memory   │
                    └────────┬────────┘
                             │ 共享内存分块
                    ┌────────▼────────┐
                    │  Tiled Kernel   │
                    │  减少全局访存    │
                    └────────┬────────┘
                             │ 在线 Softmax
                    ┌────────▼────────┐
                    │ FlashAttention  │
                    │   O(N) memory   │
                    └────────┬────────┘
                             │ 双缓冲流水线
                    ┌────────▼────────┐
                    │ Optimized Flash │
                    │ 计算/访存重叠    │
                    └─────────────────┘
```

---

## Attention 内核

### 1. Naive Attention

基准实现，用于正确性验证和性能对比。

**算法：**
```
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

**计算流程：**
1. `S = Q @ K^T * scale` → `[seq_len, seq_len]`
2. `P = softmax(S, dim=-1)` → `[seq_len, seq_len]`
3. `O = P @ V` → `[seq_len, head_dim]`

**实现特点：**
```cpp
// 每个 block 处理一个 (batch, head, row)
__global__ void naive_attention_simple_kernel(
    const T* Q, const T* K, const T* V, T* O,
    int batch_size, int num_heads, int seq_len, int head_dim, float scale
) {
    // 共享内存存储注意力分数
    extern __shared__ float shared_mem[];
    float* scores = shared_mem;
    
    // Warp 归约计算 softmax
    float reduced_max = block_reduce_max<float, 256>(local_max, reduce_smem);
    float reduced_sum = block_reduce_sum<float, 256>(local_sum, reduce_smem);
}
```

**复杂度：**
- 时间：O(N² × d)
- 显存：O(N²)

---

### 2. Tiled Attention

使用共享内存分块，减少全局内存访问。

**分块配置：**
```cpp
BLOCK_M = 32  // Q 的行分块大小
BLOCK_N = 32  // K/V 的行分块大小
```

**共享内存布局：**
```
┌────────────────────────────────────────────┐
│ smem_Q  [BLOCK_M × (head_dim+1)]           │ ← Q tile
├────────────────────────────────────────────┤
│ smem_K  [BLOCK_N × (head_dim+1)]           │ ← K tile
├────────────────────────────────────────────┤
│ smem_V  [BLOCK_N × (head_dim+1)]           │ ← V tile
├────────────────────────────────────────────┤
│ smem_S  [BLOCK_M × (BLOCK_N+1)]            │ ← attention scores
├────────────────────────────────────────────┤
│ output  [BLOCK_M × head_dim]               │ ← 输出累加器
└────────────────────────────────────────────┘

注：+1 padding 用于消除 bank conflict
```

**在线 Softmax 更新：**
```cpp
// 遍历 K/V 分块，逐步更新状态
for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
    // 1. 计算当前分块的 scores
    // 2. 更新 running max: new_max = max(old_max, block_max)
    // 3. 计算 exp(scores - new_max)
    // 4. 更新 running sum: new_sum = old_sum * old_scale + block_sum
    // 5. 重缩放输出累加器
}
```

---

### 3. FlashAttention

**核心创新：** 避免存储 N×N 注意力矩阵，实现 O(N) 显存复杂度。

**在线 Softmax 公式：**
```
O_new = (l_old × exp(m_old - m_new) × O_old + exp(S - m_new) @ V) / l_new
```

**状态维护：**
```cpp
float row_max[BLOCK_M];   // 当前行最大值 m_i
float row_sum[BLOCK_M];   // 当前行指数和 l_i
float rescale[BLOCK_M];   // 每行重缩放因子
```

**双缓冲实现：**
```cpp
// 共享内存布局（K/V 双缓冲）
smem_Q    [BLOCK_M × (head_dim+1)]      — Q tile
smem_K[2] [2 × BLOCK_N × (head_dim+1)]  — K double buffer
smem_V[2] [2 × BLOCK_N × (head_dim+1)]  — V double buffer
smem_S    [BLOCK_M × (BLOCK_N+1)]       — attention scores
output    [BLOCK_M × head_dim]          — 输出累加器

// 流水线流程
// Prologue: 加载第一个 K/V tile 到 buffer 0
// 主循环: 计算当前 buffer，同时预取下一个 tile 到交替 buffer
// Causal 早退: 当下一个 tile 超出因果窗口时跳过 prefetch
```

**两阶段计算：**
```cpp
// Phase 1: 单线程每行计算 softmax 状态（轻量）
if (tid < BLOCK_M) {
    // 计算 rowmax(scores)
    // 更新 max/sum 状态
    // 计算 rescale 因子
}

// Phase 2: 全线程协作更新输出（重量，充分并行）
for (int i = tid; i < BLOCK_M * head_dim; i += blockDim.x) {
    // 重缩放旧输出
    // 计算新贡献: exp_scores @ V
    // 更新输出
}
```

**因果掩码：**
```cpp
if (is_causal && global_col > global_row) {
    score = -FLT_MAX;  // 掩码掉未来位置
}

// 早退优化：当 col_start >= row_start + BLOCK_M 时退出
if (is_causal && col_start >= row_start + BLOCK_M) break;
```

---

## GEMM 内核

### 1. Tensor Core GEMM

使用 WMMA API 调用 Tensor Core 硬件加速。

**WMMA 片段：**
```cpp
#include <mma.h>
using namespace nvcuda;

// 16×16×16 矩阵分块
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

// 加载 → 计算 → 存储
wmma::load_matrix_sync(a_frag, A + offset, K);
wmma::load_matrix_sync(b_frag, B + offset, N);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(C + offset, c_frag, N, wmma::mem_row_major);
```

**分块版本：**
```cpp
// 共享内存分块（带 padding）
__shared__ half smem_A[BLOCK_M][BLOCK_K + 8];  // +8 half padding
__shared__ half smem_B[BLOCK_K][BLOCK_N + 8];

// 多 warp 协作
constexpr int WARPS_M = BLOCK_M / WMMA_M;  // 4 warps
constexpr int WARPS_N = BLOCK_N / WMMA_N;  // 4 warps
```

**INT8 支持（Turing+ SM≥7.2）：**
```cpp
// INT8 WMMA 尺寸：8×32×16
wmma::fragment<wmma::matrix_a, 8, 32, 16, int8_t, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 8, 32, 16, int8_t, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 8, 32, 16, int32_t> c_frag;
```

---

### 2. 高性能 GEMM（寄存器分块）

**三级分块策略：**
```
Block 级: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
Warp 级:  WARP_M=32,   WARP_N=64
Thread级: THREAD_M=8,  THREAD_N=8
```

**寄存器分块：**
```cpp
// 每个线程持有 THREAD_M × THREAD_N 输出分块
float reg_C[THREAD_M][THREAD_N] = {0};
float reg_A[THREAD_M];
float reg_B[THREAD_N];

// 外积算法
for (int k = 0; k < BLOCK_K; k++) {
    // 加载 A/B 元素到寄存器
    for (int m = 0; m < THREAD_M; m++)
        reg_A[m] = smem_A[warp_row + thread_row + m][k];
    for (int n = 0; n < THREAD_N; n++)
        reg_B[n] = smem_B[k][warp_col + thread_col + n];
    
    // 寄存器内矩阵乘
    for (int m = 0; m < THREAD_M; m++)
        for (int n = 0; n < THREAD_N; n++)
            reg_C[m][n] += reg_A[m] * reg_B[n];
}
```

**双缓冲：**
```cpp
__shared__ float smem_A[2][BLOCK_M][BLOCK_K + 1];
__shared__ float smem_B[2][BLOCK_K][BLOCK_N + 1];

// 主循环：计算当前缓冲区，预取下一个到交替缓冲区
for (int tile = 0; tile < num_k_tiles; tile++) {
    int cur_buf = tile % 2;
    int next_buf = 1 - cur_buf;
    
    // 预取下一块
    if (tile + 1 < num_k_tiles) {
        LOAD_TILE_A(next_buf, next_k_tile);
        LOAD_TILE_B(next_buf, next_k_tile);
    }
    
    // 计算当前块
    COMPUTE_TILE(cur_buf);
    __syncthreads();
}
```

---

## 头文件原语库

### common.cuh

**核心类型：**
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

**工具宏：**
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

---

### warp_primitives.cuh

**Warp 级归约：**
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

**Block 级归约：**
```cpp
template<typename T, int BLOCK_SIZE>
__device__ T block_reduce_sum(T val, T* smem) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    // Warp 内归约
    val = warp_reduce_sum(val);
    
    // 写入共享内存
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();
    
    // 第一个 warp 完成最终归约
    constexpr int num_warps = BLOCK_SIZE / 32;
    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : T(0);
        val = warp_reduce_sum(val);
    }
    return val;
}
```

---

### online_softmax.cuh

**在线 Softmax 状态：**
```cpp
struct OnlineSoftmaxState {
    float max_val;  // 当前最大值 m_i
    float sum_exp;  // 当前指数和 l_i
};
```

**状态更新：**
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

**输出更新：**
```cpp
__device__ void online_softmax_update_output(
    float* output, const float* v_block, const float* exp_weights,
    int dim, int block_n,
    float old_max, float new_max, float old_sum, float new_sum
) {
    // 重缩放因子
    float rescale = expf(old_max - new_max) * old_sum / new_sum;
    float inv_new_sum = 1.0f / new_sum;
    
    // O_new = rescale * O_old + inv_new_sum * exp_weights @ V
    for (int d = 0; d < dim; d++) {
        float val = output[d] * rescale;
        float new_val = 0.0f;
        for (int j = 0; j < block_n; j++)
            new_val += exp_weights[j] * v_block[j * dim + d];
        output[d] = val + new_val * inv_new_sum;
    }
}
```

---

### pipeline.cuh

**双缓冲结构：**
```cpp
template<typename T, int BLOCK_SIZE>
struct DoubleBuffer {
    T* buffer[2];
    int current;
    
    __device__ T* get_load_buffer()    { return buffer[current]; }
    __device__ T* get_compute_buffer() { return buffer[1 - current]; }
    __device__ void swap()             { current = 1 - current; }
};
```

**异步拷贝（Ampere+ SM≥80）：**
```cpp
#if __CUDA_ARCH__ >= 800
template<int BYTES>
__device__ void async_copy(void* dst, const void* src) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2;\n"
        :: "r"(static_cast<unsigned>(reinterpret_cast<uintptr_t>(dst))),
           "l"(src), "n"(BYTES)
    );
}

__device__ void async_copy_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ void async_copy_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
#else
// 旧架构回退到同步拷贝
#endif
```

---

## Python 绑定

### 接口设计

```cpp
// python/bindings.cpp
PYBIND11_MODULE(cuda_llm_ops, m) {
    m.doc() = "CUDA LLM Kernel Optimization";
    
    // Attention 函数
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
    
    // GEMM 函数
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

### 输入验证

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

## 性能优化技术

### 技术总结

| 优化技术 | 目标 | 实现位置 |
|----------|------|----------|
| 共享内存分块 | 减少全局访存 | tiled_attention, hgemm |
| Bank 冲突避免 | +1 padding | shared_memory.cuh |
| 在线 Softmax | O(N) 显存 | flash_attention |
| Warp Shuffle | 快速归约 | warp_primitives.cuh |
| 寄存器分块 | 数据重用 | hgemm_kernel |
| Tensor Core | 硬件加速 | tensor_core_gemm |
| 双缓冲流水线 | 隐藏延迟 | pipeline.cuh |
| 异步拷贝 | 计算/传输重叠 | pipeline.cuh (Ampere+) |

### 性能分析

**FLOPs 计算：**
```python
def compute_attention_flops(batch, heads, seq_len, head_dim):
    return batch * heads * (
        4 * seq_len * seq_len * head_dim +  # Q@K^T + P@V
        5 * seq_len * seq_len               # softmax
    )

def compute_gemm_flops(M, N, K):
    return 2 * M * N * K  # multiply-add
```

**瓶颈检测：**
```python
compute_intensity = flops / memory_bytes  # FLOPs/byte
if compute_intensity > 100:
    bottleneck = "COMPUTE_BOUND"
else:
    bottleneck = "MEMORY_BOUND"
```

---

## 测试策略

### 属性测试

使用 Hypothesis 进行属性测试，验证算法正确性：

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

### 测试覆盖

| 类别 | 内容 |
|------|------|
| 正确性 | 与 PyTorch 参考实现对比 |
| 数值稳定性 | FP16/FP32 精度验证 |
| 边界条件 | 最小维度、大序列、非对齐 |
| 布局等价性 | NN/NT/TN/TT 矩阵布局 |
| 错误处理 | 维度不匹配、dtype 错误、空张量 |

---

## 参考资料

1. **FlashAttention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
2. **FlashAttention-2**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023
3. **CUTLASS**: NVIDIA CUTLASS - CUDA Templates for Linear Algebra Subroutines
4. **cuBLAS**: NVIDIA cuBLAS Library Documentation
5. **CUDA Programming Guide**: NVIDIA CUDA C++ Programming Guide
