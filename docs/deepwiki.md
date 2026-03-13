# CUDA LLM 内核优化项目 - DeepWiki

## 项目概述

高性能 CUDA 算子库，专为 LLM 推理优化设计。实现渐进式优化策略：Naive → Tiled → FlashAttention，提供完整的优化技术栈。

---

## 1. CUDA 内核实现

### 1.1 Naive Attention (`src/naive_attention.cu`)

**算法**: 标准 Scaled Dot-Product Attention
```
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

**优化技术**:
- 共享内存存储注意力分数
- Warp 归约 (`__shfl_down_sync`) 计算行最大值
- 数值稳定 softmax（减去行最大值）

**复杂度**: 时间 O(N²×d)，显存 O(N²)

---

### 1.2 Tiled Attention (`src/tiled_attention.cu`)

**算法**: 分块注意力计算，按块加载 Q/K/V 到共享内存

**分块配置**:
```cuda
BLOCK_M=32, BLOCK_N=32, BLOCK_K=64
smem_Q[BLOCK_M][BLOCK_K + 1]  // +1 填充避免 bank 冲突
```

**优化技术**:
- 共享内存填充避免 bank 冲突
- 在线 softmax 状态维护
- 寄存器累积部分结果

---

### 1.3 FlashAttention (`src/flash_attention.cu`)

**核心创新**: 避免存储 N×N 注意力矩阵，O(N) 显存复杂度

**在线 Softmax 更新公式**:
```
O_new = (l_old × exp(m_old - m_new) × O_old + exp(S - m_new) @ V) / l_new
```

**状态维护**:
```cuda
float row_max[BLOCK_M];   // 当前行最大值 m_i
float row_sum[BLOCK_M];   // 当前行指数和 l_i
```

**特性**:
- 因果掩码支持 (`is_causal`)
- 动态头维度支持
- 可选 logsumexp 输出（用于反向传播）

---

### 1.4 Tensor Core GEMM (`src/tensor_core_gemm.cu`)

**硬件加速**: 使用 WMMA API 调用 Tensor Core

```cuda
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

**支持精度**: FP16/INT8 输入，FP32 累加

---

### 1.5 高性能 GEMM (`src/hgemm_kernel.cu`)

**三级分块策略**:
```
Block 级: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
Warp 级:  WARP_M=32,   WARP_N=64
Thread级: THREAD_M=8,  THREAD_N=8
```

**外积算法**:
```cuda
float reg_C[THREAD_M][THREAD_N];  // 寄存器累积
for (m : THREAD_M)
    for (n : THREAD_N)
        reg_C[m][n] += reg_A[m] * reg_B[n];
```

---

## 2. 头文件原语

### 2.1 common.cuh

**核心类型**:
```cuda
enum class Precision { FP32, FP16, BF16, INT8 };
struct AttentionConfig { batch_size, num_heads, seq_len, head_dim, scale, is_causal, ... };
struct GemmConfig { M, N, K, alpha, beta, layout_a, layout_b, ... };
```

**工具宏**:
```cuda
#define CUDA_CHECK(call) ...  // 错误检查
inline int div_ceil(int a, int b);
inline bool is_tensor_core_aligned(int dim, int alignment = 16);
```

---

### 2.2 warp_primitives.cuh

**Warp 级归约**:
```cuda
template<typename T>
__device__ T warp_reduce_sum(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```

**块级归约**: Warp 归约 → 共享内存 → 最终归约

---

### 2.3 shared_memory.cuh

**带填充的共享内存瓦片**:
```cuda
template<typename T, int ROWS, int COLS, int PAD = 1>
struct SharedMemoryTile {
    T data[ROWS][COLS + PAD];  // 填充避免 bank 冲突
};
```

**功能**: 合并访问、边界检查、向量化加载、转置优化

---

### 2.4 online_softmax.cuh

**FlashAttention 核心**:
```cuda
struct OnlineSoftmaxState {
    float max_val;  // m_i
    float sum_exp;  // l_i
};

void online_softmax_update(state, new_val) {
    new_max = fmaxf(state.max_val, new_val);
    old_scale = expf(state.max_val - new_max);
    state.sum_exp = state.sum_exp * old_scale + expf(new_val - new_max);
    state.max_val = new_max;
}
```

---

### 2.5 pipeline.cuh

**双缓冲**:
```cuda
template<typename T, int BLOCK_SIZE>
struct DoubleBuffer {
    T* buffer[2];
    int current;
    T* get_load_buffer()    { return buffer[current]; }
    T* get_compute_buffer() { return buffer[1 - current]; }
    void swap()             { current = 1 - current; }
};
```

**异步拷贝 (Ampere+)**:
```cuda
asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src));
```

---

## 3. Python 绑定

### 3.1 bindings.cpp

**暴露接口**:
```cpp
PYBIND11_MODULE(cuda_llm_ops, m) {
    m.def("flash_attention", &flash_attention,
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("scale") = 0.0f, py::arg("is_causal") = false);

    m.def("gemm", &gemm,
          py::arg("a"), py::arg("b"),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("trans_a") = false, py::arg("trans_b") = false);
}
```

**输入验证**: 4D 张量、CUDA 设备、连续内存、dtype 检查

---

### 3.2 profiler.py

**FLOPs 计算**:
```python
def compute_attention_flops(batch, heads, seq_len, head_dim):
    return batch * heads * (
        4 * seq_len * seq_len * head_dim +  # Q@K^T + P@V
        5 * seq_len * seq_len               # softmax
    )
```

**瓶颈检测**: 计算密集型 vs 内存带宽限制

---

## 4. 测试策略

### 4.1 测试框架 (conftest.py)

**Fixtures**:
- `attention_inputs(batch, heads, seq_len, head_dim, dtype)`
- `gemm_inputs(M, N, K, dtype)`
- `compute_attention_reference()` - PyTorch 参考实现

**自定义断言**: `assert_close(actual, expected, rtol, atol)`

---

### 4.2 测试类型

**属性测试 (Hypothesis)**:
```python
@pytest.mark.property
@settings(max_examples=100)
@given(batch=st.integers(1, 4), seq_len=st.integers(16, 256), ...)
def test_attention_correctness(batch, seq_len, ...):
    ...
```

**测试覆盖**:
| 类别 | 内容 |
|------|------|
| 正确性 | 与 PyTorch 参考实现对比 |
| 数值稳定性 | FP16/FP32 精度验证 |
| 边界条件 | 最小维度、大序列 |
| 布局等价性 | NN/NT/TN/TT 矩阵布局 |
| 错误处理 | 维度不匹配、dtype 错误 |

---

## 5. 基准测试

### 5.1 Attention Benchmark

**测试配置**:
- 序列长度: 512, 1024, 2048, 4096
- 批量/头数/头维度: 1/32/128
- 预热/迭代: 10/100

**输出示例**:
```
Seq Len | PyTorch(ms) | Naive(ms) | Tiled(ms) | Flash(ms) | Speedup
   2048 |      45.123 |    62.345 |    30.456 |    25.789 |   1.75x
```

---

### 5.2 GEMM Benchmark

**测试矩阵**: 1024², 2048², 4096², 8192²

**性能目标**: 达到 cuBLAS 90%+ 性能

**输出示例**:
```
Size           | cuBLAS(ms) | Custom(ms) | TC GEMM(ms) | Custom % | TC %
4096x4096x4096 |    125.678 |    139.234 |     127.890 |    90.2% | 98.2%
```

---

## 6. 支持的 GPU 架构

| 架构 | SM 版本 | Tensor Core |
|------|---------|-------------|
| Volta | SM 7.0 | FP16 |
| Turing | SM 7.5 | FP16, INT8 |
| Ampere | SM 8.0, 8.6 | FP16, BF16, INT8, TF32 |
| Ada Lovelace | SM 8.9 | FP16, BF16, INT8, FP8 |
| Hopper | SM 9.0 | FP16, BF16, INT8, FP8 |

---

## 7. 性能优化总结

| 优化技术 | 目标 | 实现位置 |
|----------|------|----------|
| 共享内存分块 | 减少全局内存访问 | tiled_attention, hgemm |
| Bank 冲突避免 | +1 填充 | shared_memory.cuh |
| 在线 Softmax | O(N) 显存 | flash_attention |
| Warp Shuffle | 快速归约 | warp_primitives.cuh |
| 寄存器平铺 | 最大化数据重用 | hgemm_kernel |
| Tensor Core | 硬件加速 | tensor_core_gemm |
| 双缓冲流水线 | 隐藏内存延迟 | pipeline.cuh |
| 异步拷贝 | 计算/传输重叠 | pipeline.cuh (Ampere+) |
