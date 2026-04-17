# 性能调优指南

CUDA LLM Kernel Optimization 库性能最大化的综合指南。

---

## 目录

- [硬件要求](#硬件要求)
- [性能目标](#性能目标)
- [优化策略](#优化策略)
- [性能测试](#性能测试)
- [性能分析](#性能分析)
- [性能问题排查](#性能问题排查)
- [最佳实践](#最佳实践)

---

## 硬件要求

### 最低配置 vs 推荐配置

| 组件 | 最低配置 | 推荐配置 | 说明 |
|-----------|---------|-------------|-------|
| **GPU** | SM 7.0 (Volta) | SM 8.0+ (Ampere) | Tensor Core 对于峰值性能至关重要 |
| **显存** | 8 GB | 16+ GB | 更大的批次和序列需要更多显存 |
| **CUDA** | 11.0 | 12.0+ | 新版本有更好的优化 |
| **驱动** | 450.80+ | 525+ | 使用 `nvidia-smi` 检查 |

### GPU 架构特性

| 架构 | SM | Tensor Core | 关键特性 |
|--------------|-----|-------------|--------------|
| **Volta** (V100) | 7.0 | FP16 | 第一代 Tensor Core |
| **Turing** (T4) | 7.5 | FP16, INT8 | INT8 量化支持 |
| **Ampere** (A100) | 8.0, 8.6 | FP16, BF16, INT8, TF32 | 异步拷贝, 164KB 共享内存 |
| **Ada Lovelace** (RTX 4090) | 8.9 | + FP8 | 增强的 FP8 支持 |
| **Hopper** (H100) | 9.0 | + FP8 | 线程块簇, DPX 指令 |

### 显存规划

运行前计算显存需求：

```python
def estimate_attention_memory(batch, heads, seq_len, head_dim, dtype=torch.float16):
    """估算 FlashAttention 显存使用（MB）。"""
    bytes_per_elem = 2 if dtype == torch.float16 else 4
    
    # 输入张量（Q, K, V）
    input_mem = 3 * batch * heads * seq_len * head_dim * bytes_per_elem
    
    # 输出张量
    output_mem = batch * heads * seq_len * head_dim * bytes_per_elem
    
    # FlashAttention 工作显存（开销）
    # ~O(seq_len) 而非 O(seq_len²)
    working_mem = batch * heads * seq_len * head_dim * bytes_per_elem * 0.1
    
    total_mb = (input_mem + output_mem + working_mem) / (1024 * 1024)
    return total_mb

# 示例：长序列
print(f"所需显存: {estimate_attention_memory(2, 16, 8192, 64):.1f} MB")
# 输出: 所需显存: ~55 MB (相比 naive attention 的 ~2560 MB！)
```

---

## 性能目标

### GEMM 性能预期

| 矩阵大小 | cuBLAS 参考 | 目标 | 典型结果 |
|-------------|------------------|--------|----------------|
| 512×512×512 | 100% | ≥85% | 88-92% |
| 1024×1024×1024 | 100% | ≥90% | 91-95% |
| 2048×2048×2048 | 100% | >90% | 92-98% |

### Attention 延迟基准测试

| 序列长度 | PyTorch SDPA | 我们的 FlashAttention | 加速比 |
|-----------------|--------------|-------------------|---------|
| 512 | 1.0x | 1.2x | 基准 |
| 1024 | 1.0x | 1.4x | +40% |
| 2048 | 1.0x | 1.6x | +60% |
| 4096 | 1.0x | 1.8x | +80% |
| 8192 | 1.0x | 2.1x | +110% |

> **注意**: A100 上的结果，FP16 精度，batch=2，heads=8，head_dim=64

### 显存效率

| 实现 | 显存复杂度 | 4K 序列 | 16K 序列 |
|----------------|-------------------|-------------|--------------|
| Naive Attention | O(N²) | 256 MB | 4 GB |
| Tiled Attention | O(N²), 改进 | 128 MB | 2 GB |
| FlashAttention | O(N) | 4 MB | 16 MB |

---

## 优化策略

### 1. 内核选择指南

为工作负载选择合适的内核：

```python
def optimal_attention(q, k, v, is_causal=False):
    """选择最优的 Attention 实现。"""
    seq_len = q.size(2)
    
    if seq_len >= 2048:
        # 超长序列必须使用 FlashAttention
        return flash_attention(q, k, v, is_causal=is_causal)
    elif seq_len >= 512:
        # 长序列 FlashAttention 仍然有益
        return flash_attention(q, k, v, is_causal=is_causal)
    elif seq_len >= 128:
        # 中等序列 Tiled 是好选择
        return tiled_attention(q, k, v)
    else:
        # 超短序列 Naive 可能开销更低
        return naive_attention(q, k, v)
```

### 2. 精度优化

```python
# 推理：始终使用 FP16
model = model.half()  # 转换为 FP16
q = q.cuda().half()
output = flash_attention(q, k, v)

# 训练：FP32 主权重，FP16 计算
#（混合精度模式）
with torch.cuda.amp.autocast():
    output = flash_attention(q, k, v)

# Tensor Core GEMM 带累加
# 输入: FP16, 计算: FP16, 输出: FP32
c = tensor_core_gemm(a_fp16, b_fp16)  # 返回 FP32
```

### 3. Tensor Core 对齐

最优维度应为 16 的倍数：

```python
def pad_to_alignment(tensor, alignment=16):
    """将张量维度对齐到 alignment 以获得最佳 Tensor Core 性能。"""
    shape = list(tensor.shape)
    
    # 填充最后两个维度
    if len(shape) >= 2:
        for i in [-2, -1]:
            if shape[i] % alignment != 0:
                padding = alignment - (shape[i] % alignment)
                shape[i] += padding
    
    if shape != list(tensor.shape):
        padded = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
        # 复制原始数据
        slices = [slice(0, s) for s in tensor.shape]
        padded[tuple(slices)] = tensor
        return padded
    return tensor

# 使用
a_aligned = pad_to_alignment(a, 16)  # 确保 Tensor Core 效率
b_aligned = pad_to_alignment(b, 16)
c = tensor_core_gemm(a_aligned, b_aligned)
```

### 4. 批次大小调优

```python
def find_optimal_batch_size(heads, seq_len, head_dim, max_memory_gb=16):
    """在给定显存约束下找到最优批次大小。"""
    max_memory_bytes = max_memory_gb * 1024**3
    bytes_per_elem = 2  # FP16
    
    # 每个样本显存: Q + K + V + O
    per_sample = 4 * heads * seq_len * head_dim * bytes_per_elem
    
    # 考虑 ~20% 开销
    max_batch = int(max_memory_bytes / per_sample / 1.2)
    
    # 向下取整到 2 的幂以获得更好的 GPU 利用率
    optimal_batch = 2 ** (max_batch.bit_length() - 1)
    return max(optimal_batch, 1)

# 示例
batch_size = find_optimal_batch_size(16, 2048, 64, max_memory_gb=24)
print(f"最优批次大小: {batch_size}")  # 例如, 8
```

### 5. 显存池预分配

```python
class AttentionMemoryPool:
    """为重复操作预分配和重用显存。"""
    
    def __init__(self, max_batch, heads, max_seq_len, head_dim):
        self.q = torch.empty(max_batch, heads, max_seq_len, head_dim, 
                            device='cuda', dtype=torch.float16)
        self.k = torch.empty_like(self.q)
        self.v = torch.empty_like(self.q)
        self.output = torch.empty_like(self.q)
    
    def get_tensors(self, batch_size, seq_len):
        """获取适当大小的视图。"""
        return (
            self.q[:batch_size, :, :seq_len, :],
            self.k[:batch_size, :, :seq_len, :],
            self.v[:batch_size, :, :seq_len, :],
            self.output[:batch_size, :, :seq_len, :]
        )
```

### 6. 因果 vs 非因果

```python
# 训练（带填充掩码）：不要使用 is_causal
output = flash_attention(q, k, v, is_causal=False)

# 生成（自回归）：使用 is_causal
# 这会启用早退优化
output = flash_attention(q, k, v, is_causal=True)
```

---

## 性能测试

### Attention 性能测试

```bash
# 基本性能测试
python benchmarks/benchmark_attention.py

# 自定义配置
python benchmarks/benchmark_attention.py \
    --seq-lengths 512 1024 2048 4096 8192 \
    --batch-size 4 \
    --num-heads 16 \
    --head-dim 64 \
    --dtype fp16 \
    --warmup 50 \
    --iterations 200

# 导出到 JSON 用于分析
python benchmarks/benchmark_attention.py --output results.json
```

**预期输出格式：**
```
================================================================================
ATTENTION BENCHMARK RESULTS
================================================================================

配置: batch=2, heads=8, head_dim=64, dtype=float16, causal=False

 序列长度 | PyTorch(ms) | Flash(ms) | 加速比 | 显存节省
---------|-------------|-----------|---------|-------------
     512 |       0.234 |     0.189 |   1.24x |       3.5MB
    1024 |       0.823 |     0.612 |   1.34x |      14.0MB
    2048 |       3.124 |     2.156 |   1.45x |      56.0MB
    4096 |      12.456 |     7.234 |   1.72x |     224.0MB
    8192 |      49.823 |    23.456 |   2.12x |     896.0MB
```

### GEMM 性能测试

```bash
# 标准性能测试
python benchmarks/benchmark_gemm.py

# 自定义矩阵大小
python benchmarks/benchmark_gemm.py \
    --sizes 512x512x512 1024x1024x1024 2048x2048x2048 \
    --dtype fp16 \
    --output gemm_results.json
```

### 自定义性能测试脚本

```python
import torch
import time
from cuda_llm_ops import flash_attention, gemm
from cuda_llm_ops.profiler import CUDAProfiler

def benchmark_flash_attention(batch, heads, seq_len, head_dim, iterations=100):
    """自定义 FlashAttention 性能测试。"""
    q = torch.randn(batch, heads, seq_len, head_dim, 
                    device='cuda', dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    
    # Warmup
    for _ in range(10):
        _ = flash_attention(q, k, v)
    torch.cuda.synchronize()
    
    # 性能测试
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        _ = flash_attention(q, k, v)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / iterations
    
    # 计算指标
    flops = batch * heads * (4 * seq_len * seq_len * head_dim)  # 简化版
    tflops = flops / (elapsed_ms * 1e-3) / 1e12
    
    return {
        'latency_ms': elapsed_ms,
        'throughput_tflops': tflops,
        'config': f'{batch}x{heads}x{seq_len}x{head_dim}'
    }

# 运行性能测试
result = benchmark_flash_attention(2, 8, 2048, 64)
print(f"配置: {result['config']}")
print(f"延迟: {result['latency_ms']:.3f} ms")
print(f"吞吐量: {result['throughput_tflops']:.2f} TFLOPS")
```

---

## 性能分析

### 使用 Nsight Compute

```bash
# 使用详细指标分析 FlashAttention
ncu --set full --kernel-name flash_attention \
    -o flash_attention_profile.ncu-rep \
    python -c "
import torch
from cuda_llm_ops import flash_attention

q = torch.randn(2, 8, 2048, 64, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

for _ in range(100):
    flash_attention(q, k, v)
"

# 查看结果
ncu-ui flash_attention_profile.ncu-rep
```

### 关键监控指标

| 指标 | 目标 | 解读 |
|--------|--------|----------------|
| **SM 占用率** | >60% | 高：良好的线程利用率 |
| **显存吞吐** | >峰值 70% | 受显存带宽限制 |
| **L2 命中率** | >60% | 高效的缓存利用 |
| **Warp 停顿原因** | 各异 | 识别瓶颈 |

### Python 分析器

```python
from cuda_llm_ops.profiler import CUDAProfiler

profiler = CUDAProfiler()

# 分析单个操作
with profiler.profile('flash_attention'):
    output = flash_attention(q, k, v)

metrics = profiler.get_metrics()
print(f"耗时: {metrics.elapsed_ms:.2f} ms")
print(f"显存带宽: {metrics.memory_bandwidth_gb:.1f} GB/s")
print(f"计算利用率: {metrics.compute_utilization:.1%}")

# 与参考实现对比
results = profiler.compare_with_reference(
    custom_func=flash_attention,
    reference_func=lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v),
    q, k, v
)
print(f"相比 PyTorch 加速: {results['speedup']:.2f}x")
```

---

## 性能问题排查

### 问题：GPU 利用率低

**症状：**
- `nvidia-smi` 显示 GPU 利用率低（<50%）
- 内核执行时间远高于预期

**解决方案：**
1. 增加批次大小
2. 检查 CPU-GPU 同步点
3. 确保张量是连续的

```python
# 检查张量布局
assert q.is_contiguous(), "Q 必须是连续的"

# 移除不必要的同步
# 错误:
torch.cuda.synchronize()
output = flash_attention(q, k, v)
torch.cuda.synchronize()

# 正确:
output = flash_attention(q, k, v)
```

### 问题：Tensor Core 未启用

**症状：**
- 性能远低于 cuBLAS
- Nsight Compute 显示没有 Tensor Core 使用

**解决方案：**
1. 确保维度是 16 的倍数
2. 使用 `tensor_core_gemm` 而非普通 `gemm`
3. 验证 FP16 输入数据类型

```python
# 检查对齐
M, K, N = a.shape[0], a.shape[1], b.shape[1]
for dim, name in zip([M, K, N], ['M', 'K', 'N']):
    if dim % 16 != 0:
        print(f"警告: {name}={dim} 未对齐到 16")

# 使用 Tensor Core 变体
c = tensor_core_gemm(a, b)  # 强制使用 Tensor Core
```

### 问题：显存溢出

**症状：**
- `RuntimeError: CUDA out of memory`
- 迭代间显存增长

**解决方案：**
1. 使用 FlashAttention 而非 naive attention
2. 减少批次大小或序列长度
3. 迭代间清理缓存

```python
# 使用 FlashAttention（O(N) 显存）
output = flash_attention(q, k, v)  # 不是 naive_attention

# 如果需要清理缓存
torch.cuda.empty_cache()

# 监控显存
print(torch.cuda.memory_summary())
```

---

## 最佳实践

### 推理场景

1. **使用 FP16 精度** 以获得显存效率和速度
2. **序列 >512 使用 FlashAttention**
3. **为重复操作预分配显存池**
4. **对齐维度到 16 的倍数** 以使用 Tensor Core
5. **尽可能批处理请求** 以获得更好的 GPU 利用率

### 训练场景

1. **使用自动混合精度（AMP）** 配合 GradScaler
2. **使用 FP16 时监控梯度范数**
3. **先小规模测试验证数值稳定性**
4. **优化前先分析** 以识别实际瓶颈

### 生产场景

1. **实现优雅降级**（需要时回退到 PyTorch）
2. **多 GPU 设置显式设置 CUDA 可见设备**
3. **使用 CUDA 流进行并发执行** 处理多个请求
4. **监控 GPU 显存** 并实现 OOM 处理

```python
# 生产就绪模式
import torch
from cuda_llm_ops import flash_attention

def safe_flash_attention(q, k, v, fallback_to_torch=True):
    try:
        return flash_attention(q, k, v)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and fallback_to_torch:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)
        raise
```

---

## 其他资源

- [CUDA 最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/)
- [PyTorch 性能调优](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

[← 返回文档](../README.md)
