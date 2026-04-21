---
layout: docs
title: 快速入门指南
description: 5 分钟内上手 CUDA LLM Kernel Optimization
lang: zh-CN
---

# 快速入门指南

5 分钟内上手 CUDA LLM Kernel Optimization。

---

## 目录

- [安装](#安装)
- [第一步](#第一步)
- [FlashAttention 快速入门](#flashattention-快速入门)
- [GEMM 快速入门](#gemm-快速入门)
- [下一步](#下一步)

---

## 安装

### 系统要求

- CUDA 11.0 或更高版本
- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU（SM 7.0+，Volta 或更新）

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/LessUp/llm-speed.git
cd llm-speed

# 安装依赖
pip install -r requirements.txt

# 构建并安装 CUDA 扩展
pip install -e .
```

### 验证安装

```bash
python -c "import cuda_llm_ops; print(f'版本: {cuda_llm_ops.__version__}')"
```

预期输出：
```
版本: 0.3.0
```

---

## 第一步

### 检查 GPU 兼容性

```python
import torch

# 检查 CUDA 可用性
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"设备名称: {torch.cuda.get_device_name(0)}")

# 检查 GPU 计算能力（需要 SM 7.0+）
capability = torch.cuda.get_device_capability()
print(f"计算能力: {capability[0]}.{capability[1]}")

if capability[0] >= 7:
    print("✓ GPU 支持 CUDA LLM Kernel Optimization")
else:
    print("✗ GPU 太旧（需要 Volta 或更新）")
```

---

## FlashAttention 快速入门

### 基本用法

```python
import torch
from cuda_llm_ops import flash_attention

# 创建示例输入
batch = 2          # 序列数量
heads = 8          # Attention 头数
seq_len = 512      # 序列长度
head_dim = 64      # 每个头的维度

# 在 GPU 上使用 FP16 精度创建张量
q = torch.randn(batch, heads, seq_len, head_dim, 
                device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

# 计算 Attention
output = flash_attention(q, k, v)

print(f"输入形状:  {q.shape}")
print(f"输出形状: {output.shape}")
# 输入形状:  torch.Size([2, 8, 512, 64])
# 输出形状: torch.Size([2, 8, 512, 64])
```

### 因果 Attention（用于 GPT-like 模型）

```python
# 用于自回归生成的因果掩码
def generate_next_token(q, k, v_cache):
    """使用因果 Attention 生成下一个 token。"""
    # 因果掩码确保每个位置只能关注之前的的位置
    output = flash_attention(q, k, v_cache, is_causal=True)
    return output

# 示例
output_causal = flash_attention(q, k, v, is_causal=True)
```

### 显存效率演示

```python
import gc

def measure_memory(func, *args):
    """测量函数的峰值显存使用。"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    result = func(*args)
    torch.cuda.synchronize()
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    return result, peak_memory

# 使用长序列测试
long_q = torch.randn(2, 8, 4096, 64, device='cuda', dtype=torch.float16)
long_k = torch.randn_like(long_q)
long_v = torch.randn_like(long_q)

# FlashAttention
_, mem_flash = measure_memory(flash_attention, long_q, long_k, long_v)
print(f"FlashAttention 显存: {mem_flash:.1f} MB")
# ~32 MB

# 对比：标准 Attention 4K 序列会使用 ~256 MB！
print(f"节省显存: ~{256 - mem_flash:.1f} MB")
```

---

## GEMM 快速入门

### 标准 GEMM

```python
import torch
from cuda_llm_ops import gemm

# 矩阵维度
M, K, N = 1024, 512, 1024

# 在 GPU 上创建矩阵
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

# C = A @ B
c = gemm(a, b)
print(f"结果形状: {c.shape}")  # torch.Size([1024, 1024])

# C = 2.0 * A @ B（带缩放）
c_scaled = gemm(a, b, alpha=2.0)
```

### Tensor Core GEMM

```python
from cuda_llm_ops import tensor_core_gemm

# FP16 输入，FP32 输出（更高精度累加）
a = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
b = torch.randn(512, 1024, device='cuda', dtype=torch.float16)

c = tensor_core_gemm(a, b)
print(f"输入数据类型:  {a.dtype}")      # torch.float16
print(f"输出数据类型: {c.dtype}")     # torch.float32
```

### 转置操作

```python
# 处理转置矩阵而无需显式转置
a_t = torch.randn(K, M, device='cuda', dtype=torch.float16)  # 实际是 A^T
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

# C = A^T @ B（等效于 a.t() @ b，但更高效）
c = gemm(a_t, b, trans_a=True)
print(f"结果形状: {c.shape}")  # torch.Size([M, N])

# 所有组合：
# gemm(a, b, trans_a=False, trans_b=False)  # A @ B
# gemm(a, b, trans_a=False, trans_b=True)   # A @ B^T
# gemm(a, b, trans_a=True,  trans_b=False)  # A^T @ B
# gemm(a, b, trans_a=True,  trans_b=True)   # A^T @ B^T
```

---

## 下一步

### 探索示例

```bash
# 运行性能测试
python benchmarks/benchmark_attention.py
python benchmarks/benchmark_gemm.py

# 运行测试
pytest tests/ -v
```

### 阅读文档

- **[API 参考](../api/api-zh.md)** - 完整 API 文档
- **[架构指南](../architecture/architecture-zh.md)** - 实现细节
- **[性能指南](../tutorials/performance-zh.md)** - 优化建议
- **[故障排除](../tutorials/troubleshooting-zh.md)** - 常见问题

### 尝试不同配置

```python
# 尝试不同的序列长度
for seq_len in [256, 512, 1024, 2048, 4096]:
    q = torch.randn(2, 8, seq_len, 64, device='cuda', dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    
    # 计时
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    _ = flash_attention(q, k, v)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end)
    print(f"序列长度 {seq_len:4d}: {elapsed_ms:.3f} ms")
```

### 加入社区

- **GitHub Issues**: 报告 Bug 或请求功能
- **Discussions**: 提问和分享想法

---

## 快速参考卡

```python
# ========== Attention ==========
from cuda_llm_ops import flash_attention, tiled_attention, naive_attention

# 标准 Attention
out = flash_attention(q, k, v)

# 带因果掩码（GPT 风格）
out = flash_attention(q, k, v, is_causal=True)

# ========== GEMM ==========
from cuda_llm_ops import gemm, tensor_core_gemm, tensor_core_gemm_int8

# 标准 GEMM: C = alpha * A @ B + beta * C
c = gemm(a, b, alpha=1.0, beta=0.0)

# Tensor Core（FP16 输入，FP32 输出）
c = tensor_core_gemm(a_fp16, b_fp16)

# INT8 量化（仅 Turing+）
c = tensor_core_gemm_int8(a_int8, b_int8)  # 返回 INT32

# 转置
output = gemm(a_t, b, trans_a=True)  # A^T @ B
```

[← 返回文档](../)
