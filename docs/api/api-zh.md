# API 参考

CUDA LLM Kernel Optimization 库的完整 API 文档。

---

## 目录

- [安装](#安装)
- [模块概述](#模块概述)
- [Attention 函数](#attention-函数)
  - [flash_attention](#flash_attention)
  - [tiled_attention](#tiled_attention)
  - [naive_attention](#naive_attention)
- [GEMM 函数](#gemm-函数)
  - [gemm](#gemm)
  - [tensor_core_gemm](#tensor_core_gemm)
  - [tensor_core_gemm_int8](#tensor_core_gemm_int8)
- [张量要求](#张量要求)
- [错误处理](#错误处理)
- [性能建议](#性能建议)
- [版本信息](#版本信息)

---

## 安装

```bash
# 从源码安装
pip install -e .

# 验证安装
python -c "import cuda_llm_ops; print(cuda_llm_ops.__version__)"
```

## 模块概述

```python
import cuda_llm_ops

# 列出所有可用函数
dir(cuda_llm_ops)
# ['flash_attention', 'tiled_attention', 'naive_attention',
#  'gemm', 'tensor_core_gemm', 'tensor_core_gemm_int8', '__version__']
```

---

## Attention 函数

### flash_attention

采用在线 Softmax 算法实现 O(N) 显存复杂度的 FlashAttention。

```python
def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 0.0,
    is_causal: bool = False
) -> torch.Tensor
```

#### 参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `q` | `torch.Tensor` | 必需 | 查询张量，形状 `[batch, heads, seq_len, head_dim]` |
| `k` | `torch.Tensor` | 必需 | 键张量，形状 `[batch, heads, seq_len, head_dim]` |
| `v` | `torch.Tensor` | 必需 | 值张量，形状 `[batch, heads, seq_len, head_dim]` |
| `scale` | `float` | `0.0` | Attention 缩放因子。如果为 `0.0`，使用 `1/√head_dim` |
| `is_causal` | `bool` | `False` | 为自回归模型启用因果掩码 |

#### 返回值

| 类型 | 描述 |
|------|-------------|
| `torch.Tensor` | 输出张量，形状 `[batch, heads, seq_len, head_dim]` |

#### 异常

| 异常 | 条件 |
|-----------|-----------|
| `RuntimeError` | 输入张量不是 4D |
| `RuntimeError` | 输入张量不在 CUDA 设备上 |
| `RuntimeError` | 输入张量不连续 |
| `RuntimeError` | Q、K、V 形状不匹配 |
| `RuntimeError` | 不支持的数据类型（非 float32/float16） |

#### 示例

```python
import torch
from cuda_llm_ops import flash_attention

# 标准 Attention
batch, heads, seq_len, head_dim = 2, 8, 512, 64
q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

output = flash_attention(q, k, v)
print(output.shape)  # torch.Size([2, 8, 512, 64])

# 因果 Attention（用于 GPT 等自回归模型）
output_causal = flash_attention(q, k, v, is_causal=True)

# 自定义缩放因子
output_scaled = flash_attention(q, k, v, scale=0.125)  # 1/8 而非 1/√64
```

#### 显存使用对比

| 序列长度 | 标准 Attention | FlashAttention | 降低比例 |
|-----------------|-------------------|----------------|-----------|
| 1024 | 4 MB | 0.25 MB | 94% |
| 2048 | 16 MB | 0.5 MB | 97% |
| 4096 | 64 MB | 1 MB | 98% |

---

### tiled_attention

采用共享内存分块优化的分块 Attention。适用于中等长度序列。

```python
def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 0.0,
    is_causal: bool = False
) -> torch.Tensor
```

#### 参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `q` | `torch.Tensor` | 必需 | 查询张量，形状 `[batch, heads, seq_len, head_dim]` |
| `k` | `torch.Tensor` | 必需 | 键张量，形状 `[batch, heads, seq_len, head_dim]` |
| `v` | `torch.Tensor` | 必需 | 值张量，形状 `[batch, heads, seq_len, head_dim]` |
| `scale` | `float` | `0.0` | Attention 缩放因子。如果为 `0.0`，使用 `1/√head_dim` |
| `is_causal` | `bool` | `False` | 为自回归模型启用因果掩码 |

#### 返回值

与 `flash_attention` 相同。

#### 示例

```python
from cuda_llm_ops import tiled_attention

output = tiled_attention(q, k, v)

# 使用自定义缩放
output_scaled = tiled_attention(q, k, v, scale=0.1)
```

#### 说明

- 对于序列长度 ≥128，比 `naive_attention` 更高效
- 仍然在内部存储 Attention 矩阵（O(N²) 显存）
- 不建议用于序列长度 >2048 的情况

---

### naive_attention

O(N²) 显存复杂度的基准 Attention 实现。主要用于正确性验证。

```python
def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 0.0,
    is_causal: bool = False
) -> torch.Tensor
```

#### 参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `q` | `torch.Tensor` | 必需 | 查询张量，形状 `[batch, heads, seq_len, head_dim]` |
| `k` | `torch.Tensor` | 必需 | 键张量，形状 `[batch, heads, seq_len, head_dim]` |
| `v` | `torch.Tensor` | 必需 | 值张量，形状 `[batch, heads, seq_len, head_dim]` |
| `scale` | `float` | `0.0` | Attention 缩放因子。如果为 `0.0`，使用 `1/√head_dim` |
| `is_causal` | `bool` | `False` | 为自回归模型启用因果掩码 |

#### 返回值

与 `flash_attention` 相同。

#### 警告

> **显存警告**: 此实现存储完整的 N×N Attention 矩阵。对于长序列（N > 1024），可能会导致显存溢出错误。生产环境请使用 `flash_attention`。

#### 示例

```python
from cuda_llm_ops import naive_attention

# 仅建议用于短序列或测试
q = torch.randn(2, 4, 64, 32, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

output = naive_attention(q, k, v)

# 与 PyTorch 参考实现验证正确性
reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
assert torch.allclose(output, reference, rtol=1e-3, atol=1e-3)
```

---

## GEMM 函数

### gemm

采用寄存器分块的高性能通用矩阵乘法。

```python
def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    trans_a: bool = False,
    trans_b: bool = False
) -> torch.Tensor
```

#### 参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `a` | `torch.Tensor` | 必需 | 矩阵 A，形状 `[M, K]`（或 `[K, M]` 如果 `trans_a=True`） |
| `b` | `torch.Tensor` | 必需 | 矩阵 B，形状 `[K, N]`（或 `[N, K]` 如果 `trans_b=True`） |
| `alpha` | `float` | `1.0` | A @ B 的缩放因子 |
| `beta` | `float` | `0.0` | 现有 C 的缩放因子（当前未使用） |
| `trans_a` | `bool` | `False` | 转置矩阵 A |
| `trans_b` | `bool` | `False` | 转置矩阵 B |

#### 返回值

| 类型 | 描述 |
|------|-------------|
| `torch.Tensor` | 输出矩阵 C，形状 `[M, N]` |

#### 异常

| 异常 | 条件 |
|-----------|-----------|
| `RuntimeError` | 输入张量不是 2D |
| `RuntimeError` | 内维不匹配 |
| `RuntimeError` | 张量不在 CUDA 上或不连续 |
| `RuntimeError` | 不支持的数据类型 |

#### 示例

```python
import torch
from cuda_llm_ops import gemm

# 标准 C = A @ B
M, K, N = 1024, 512, 1024
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
c = gemm(a, b)

# 带缩放: C = 2.0 * A @ B
c = gemm(a, b, alpha=2.0)

# 处理转置矩阵
a_t = torch.randn(K, M, device='cuda', dtype=torch.float16)  # 实际是 A^T
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

# 计算 A^T @ B
c = gemm(a_t, b, trans_a=True)
```

#### 转置等价表

| `trans_a` | `trans_b` | 操作 |
|-----------|-----------|-----------|
| False | False | C = A @ B |
| False | True | C = A @ B^T |
| True | False | C = A^T @ B |
| True | True | C = A^T @ B^T |

---

### tensor_core_gemm

使用 WMMA API 的 Tensor Core 加速矩阵乘法。

```python
def tensor_core_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0
) -> torch.Tensor
```

#### 参数

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `a` | `torch.Tensor` | 矩阵 A，形状 `[M, K]`，必须是 `float16` |
| `b` | `torch.Tensor` | 矩阵 B，形状 `[K, N]`，必须是 `float16` |
| `alpha` | `float` | 缩放因子 |
| `beta` | `float` | 现有 C 的缩放因子 |

#### 返回值

| 类型 | 描述 |
|------|-------------|
| `torch.Tensor` | 输出矩阵 C，形状 `[M, N]`，数据类型 `float32` |

#### 硬件要求

- **GPU**: Volta 架构或更新（SM 7.0+）
- **PyTorch**: 2.0+

#### 示例

```python
import torch
from cuda_llm_ops import tensor_core_gemm

a = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
b = torch.randn(512, 1024, device='cuda', dtype=torch.float16)

# 返回 FP32 输出以获得更高精度累加
c = tensor_core_gemm(a, b)
print(c.dtype)  # torch.float32

# 缩放
c = tensor_core_gemm(a, b, alpha=0.5)
```

#### 性能建议

- M、K、N 对齐到 16 的倍数以获得最佳性能
- 输入使用 `float16` 以获得最大吞吐量
- 输出精度为 `float32` 以确保数值稳定性

---

### tensor_core_gemm_int8

使用 Tensor Core 的 INT8 量化矩阵乘法。

```python
def tensor_core_gemm_int8(
    a: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor
```

#### 参数

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `a` | `torch.Tensor` | 矩阵 A，形状 `[M, K]`，必须是 `int8` |
| `b` | `torch.Tensor` | 矩阵 B，形状 `[K, N]`，必须是 `int8` |

#### 返回值

| 类型 | 描述 |
|------|-------------|
| `torch.Tensor` | 输出矩阵 C，形状 `[M, N]`，数据类型 `int32` |

#### 硬件要求

- **GPU**: Turing 架构或更新（SM 7.2+）
- **PyTorch**: 2.0+

#### 示例

```python
import torch
from cuda_llm_ops import tensor_core_gemm_int8

# 创建 INT8 张量
a = torch.randint(-128, 127, (1024, 512), device='cuda', dtype=torch.int8)
b = torch.randint(-128, 127, (512, 1024), device='cuda', dtype=torch.int8)

# INT32 累加以确保精度
c = tensor_core_gemm_int8(a, b)
print(c.dtype)  # torch.int32

# 与参考实现验证
reference = torch.matmul(a.to(torch.int32), b.to(torch.int32))
assert torch.equal(c, reference)
```

#### 检查 GPU 兼容性

```python
import torch
 capability = torch.cuda.get_device_capability()
if capability[0] > 7 or (capability[0] == 7 and capability[1] >= 2):
    print("支持 INT8 Tensor Core")
else:
    print("需要 Turing+ GPU（SM 7.2+）")
```

---

## 张量要求

### Attention 函数

| 要求 | 规范 |
|-------------|---------------|
| 维度 | 4D: `[batch, heads, seq_len, head_dim]` |
| 设备 | 必须是 CUDA（`tensor.is_cuda == True`） |
| 布局 | 连续（`tensor.is_contiguous() == True`） |
| 数据类型 | `float32` 或 `float16` |
| 形状一致性 | Q、K、V 形状必须完全匹配 |

### GEMM 函数

| 要求 | 规范 |
|-------------|---------------|
| 维度 | 2D: `[M, K]` 和 `[K, N]` |
| 设备 | 必须是 CUDA |
| 布局 | 连续 |
| 数据类型 | `float32`、`float16`，或 `int8`（仅量化） |
| 维度对齐 | 内维必须匹配 |

---

## 错误处理

### 常见错误信息

```python
# 示例：维度错误
q = torch.randn(64, 32, device='cuda')  # 2D 而非 4D
flash_attention(q, k, v)
# RuntimeError: Q 必须是 4D 张量 [batch, heads, seq_len, head_dim]

# 示例：CPU 张量
q = torch.randn(2, 4, 64, 32)  # CPU 张量
flash_attention(q, k, v)
# RuntimeError: Q 必须在 CUDA 设备上

# 示例：非连续张量
q = torch.randn(2, 4, 64, 32, device='cuda').transpose(1, 2)
flash_attention(q, k, v)
# RuntimeError: Q 必须是连续的

# 示例：形状不匹配
q = torch.randn(2, 4, 64, 32, device='cuda')
v = torch.randn(2, 4, 128, 32, device='cuda')  # 不同的 seq_len
flash_attention(q, k, v)
# RuntimeError: K 和 V 必须具有相同形状

# 示例：不支持的数据类型
q = torch.randn(2, 4, 64, 32, device='cuda', dtype=torch.int32)
flash_attention(q, k, v)
# RuntimeError: 仅支持 float32 和 float16
```

### 错误处理模式

```python
import torch
from cuda_llm_ops import flash_attention

def safe_flash_attention(q, k, v, **kwargs):
    try:
        return flash_attention(q, k, v, **kwargs)
    except RuntimeError as e:
        error_msg = str(e)
        if "必须" in error_msg:
            print(f"错误: {error_msg}")
        else:
            print(f"错误: {error_msg}")
        raise
```

---

## 性能建议

### 显存优化

```python
# 长序列使用 FlashAttention
seq_len = 1024
if seq_len >= 512:
    output = flash_attention(q, k, v)  # O(N) 显存
else:
    output = naive_attention(q, k, v)  # 短序列可能更快
```

### 精度选择

```python
# 推理使用 FP16（推荐）
q_fp16 = q.half()
output = flash_attention(q_fp16, k_fp16, v_fp16)

# 训练或精度关键场景使用 FP32
output = flash_attention(q.float(), k.float(), v.float())

# Tensor Core GEMM 进行 FP16 输入的 FP32 累加
c = tensor_core_gemm(a.half(), b.half())  # 返回 FP32
```

### 最佳维度

```python
# 为最佳 Tensor Core 性能，使用 16 的倍数
def round_up_to_16(x):
    return ((x + 15) // 16) * 16

M = round_up_to_16(1000)  # 1008
N = round_up_to_16(500)   # 512
```

### 批处理

```python
# 一起处理多个序列以获得更好的 GPU 利用率
batch_size = 8  # 根据可用显存调整
q = torch.randn(batch_size, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
```

---

## 版本信息

```python
import cuda_llm_ops

print(cuda_llm_ops.__version__)
# 例如，"0.3.0"
```

---

## 支持

- **Issues**: https://github.com/LessUp/llm-speed/issues
- **文档**: https://lessup.github.io/llm-speed/

[← 返回文档](../README.md)
