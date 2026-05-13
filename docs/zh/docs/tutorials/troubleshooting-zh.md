---
title: 故障排除指南
description: 使用 LLM-Speed 时常见问题的解决方案
---

# 故障排除指南

使用 LLM-Speed 时常见问题的解决方案。

---

## 目录

- [安装问题](#安装问题)
- [运行时错误](#运行时错误)
- [性能问题](#性能问题)
- [数值问题](#数值问题)
- [获取帮助](#获取帮助)

---

## 安装问题

### 本地基线环境未准备完成

**现象：**
```
ImportError / ModuleNotFoundError 出现在测试收集阶段
```

**原因：**
在准备好文档要求的本地 Python 环境之前就开始执行验证。

**修复方法：**
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt pytest hypothesis ruff pre-commit
```

### CUDA 未找到

**错误：**
```
RuntimeError: CUDA not available. Please check your CUDA installation.
```

**解决方案：**

1. **验证 CUDA 安装：**
```bash
nvcc --version
nvidia-smi
```

2. **检查 PyTorch CUDA 支持：**
```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
```

3. **使用正确的 CUDA 版本重新安装 PyTorch：**
```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 构建错误

**错误：**
```
error: command 'gcc' failed with exit status 1
```

**解决方案：**

1. **检查 GCC 版本：**
```bash
gcc --version  # 需要 GCC 9.0+
```

2. **设置 CUDA 架构标志：**
```bash
# 特定 GPU 架构
CUDA_ARCHS="80" pip install -e .  # A100

# 多个架构
CUDA_ARCHS="75;80;86" pip install -e .
```

3. **常见修复：**
```bash
# 清除构建缓存
rm -rf build/
rm -rf *.egg-info

# 使用详细输出重新构建
pip install -e . --verbose
```

### 导入错误

**错误：**
```
ImportError: No module named 'cuda_llm_ops'
```

**解决方案：**

1. **验证安装：**
```bash
pip list | grep cuda
python -c "import cuda_llm_ops; print(cuda_llm_ops.__version__)"
```

2. **检查 Python 路径：**
```python
import sys
print(sys.path)
```

3. **重新安装：**
```bash
pip uninstall cuda_llm_ops
pip install -e .
```

---

## 运行时错误

### CUDA 显存不足

**错误：**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**解决方案：**

1. **使用 FlashAttention（O(N) 显存）而非 naive attention：**
```python
# 错误 - 长序列可能 OOM
from cuda_llm_ops import naive_attention
output = naive_attention(q, k, v)  # O(N²) 显存

# 正确 - 显存高效
from cuda_llm_ops import flash_attention
output = flash_attention(q, k, v)   # O(N) 显存
```

2. **减少批次大小或序列长度：**
```python
# 检查操作前的显存
print(torch.cuda.memory_summary())

# 尝试更小的批次
batch_size = 2  # 而不是 8
```

3. **清除缓存：**
```python
torch.cuda.empty_cache()
```

4. **混合精度：**
```python
# 使用 FP16 而非 FP32
q = q.half()
```

### 共享内存限制

**错误：**
```
RuntimeError: naive_attention: seq_len=4096 需要 16404 字节共享内存，
但设备最大为 49152 字节。
```

**解决方案：**
长序列使用 `flash_attention` 或 `tiled_attention`：

```python
if seq_len > 2048:
    output = flash_attention(q, k, v)  # 无共享内存限制
else:
    output = tiled_attention(q, k, v)
```

### 张量形状不匹配

**错误：**
```
RuntimeError: K and V must have same shape
```

**解决方案：**
确保 Q、K、V 具有相同的形状：

```python
print(f"Q 形状: {q.shape}")
print(f"K 形状: {k.shape}")
print(f"V 形状: {v.shape}")

# 都应该是: [batch, heads, seq_len, head_dim]
assert q.shape == k.shape == v.shape
```

### 错误的设备

**错误：**
```
RuntimeError: Q must be on CUDA device
```

**解决方案：**
将张量移动到 GPU：

```python
# 检查设备
print(f"Q 设备: {q.device}")

# 如果需要移动到 CUDA
q = q.cuda()
# 或在创建时
q = torch.randn(..., device='cuda', dtype=torch.float16)
```

### 非连续张量

**错误：**
```
RuntimeError: Q must be contiguous
```

**解决方案：**
```python
# 使其连续
q = q.contiguous()

# 或在转置时
def safe_transpose(tensor, dim0, dim1):
    """转置并使其连续。"""
    return tensor.transpose(dim0, dim1).contiguous()
```

### 不支持的数据类型

**错误：**
```
RuntimeError: Only float32 and float16 are supported
```

**解决方案：**
```python
# 转换为支持的数据类型
q = q.half()      # FP16
# 或
q = q.float()     # FP32
```

### 错误的维度

**错误：**
```
RuntimeError: Q must be 4D tensor [batch, heads, seq_len, head_dim]
```

**解决方案：**
```python
# 期望: [batch, heads, seq_len, head_dim]
print(f"当前形状: {q.shape}")
print(f"维度: {q.dim()}")

# 如果需要重塑
q = q.view(batch, heads, seq_len, head_dim)
```

### INT8 Tensor Core 不可用

**错误：**
```
RuntimeError: INT8 Tensor Core requires Turing+ architecture (SM 7.2+)
```

**解决方案：**
检查 GPU 计算能力：

```python
import torch
capability = torch.cuda.get_device_capability()
print(f"计算能力: {capability}")

if capability[0] > 7 or (capability[0] == 7 and capability[1] >= 2):
    # Turing 或更好
    from cuda_llm_ops import tensor_core_gemm_int8
    c = tensor_core_gemm_int8(a_int8, b_int8)
else:
    # 回退到 FP16
    from cuda_llm_ops import tensor_core_gemm
    c = tensor_core_gemm(a_fp16, b_fp16)
```

---

## 性能问题

### 执行缓慢

**症状：**
- 操作耗时远高于预期
- GPU 利用率低（`nvidia-smi` 中）

**解决方案：**

1. **使用适合序列长度的最优内核：**
```python
seq_len = q.size(2)

if seq_len >= 512:
    output = flash_attention(q, k, v)  # 长序列最优
elif seq_len >= 128:
    output = tiled_attention(q, k, v)  # 中等序列良好
else:
    output = naive_attention(q, k, v)  # 短序列可以
```

2. **检查对齐：**
```python
def check_alignment(M, N, K):
    for dim, name in [(M, 'M'), (N, 'N'), (K, 'K')]:
        if dim % 16 != 0:
            print(f"警告: {name}={dim} 未对齐到 16")

check_alignment(1024, 512, 1024)
```

3. **确保 warmup：**
```python
# GPU 需要 warmup 以获得一致的计时
for _ in range(10):
    _ = flash_attention(q, k, v)
torch.cuda.synchronize()
```

### GPU 利用率低

**症状：**
- `nvidia-smi` 显示利用率 < 50%
- CPU 瓶颈

**解决方案：**

1. **增加批次大小：**
```python
# 太小
batch = 1
q = torch.randn(1, heads, seq_len, head_dim, device='cuda')

# 更好
batch = 8
q = torch.randn(8, heads, seq_len, head_dim, device='cuda')
```

2. **移除 CPU-GPU 同步：**
```python
# 错误 - 强制 CPU 等待
result = flash_attention(q, k, v)
torch.cuda.synchronize()
print(result.cpu())

# 更好 - 批处理操作
results = []
for _ in range(100):
    results.append(flash_attention(q, k, v))
torch.cuda.synchronize()
```

### Tensor Core 未使用

**症状：**
- 性能远低于 cuBLAS
- Nsight Compute 显示没有 Tensor Core 使用

**解决方案：**

1. **使用 Tensor Core 变体：**
```python
# 使用常规 CUDA 核心
c = gemm(a, b)

# 使用 Tensor Core
c = tensor_core_gemm(a, b)
```

2. **确保 FP16 输入：**
```python
# 必须是 FP16 才能使用 Tensor Core
c = tensor_core_gemm(a.half(), b.half())
```

3. **检查对齐：**
```python
# 所有维度应该是 16 的倍数
M, K, N = 1024, 512, 1024  # 好：都可被 16 整除
```

---

## 数值问题

### FP16 精度损失

**症状：**
- 结果与 FP32 参考差异显著
- NaN 或 Inf 值

**解决方案：**

1. **使用 Tensor Core GEMM 进行累加：**
```python
# FP16 计算，FP32 累加
c = tensor_core_gemm(a_fp16, b_fp16)  # 返回 FP32
```

2. **为 FP16 缩放值：**
```python
# FP16 范围有限 [-65504, 65504]
# 缩放大值
scale = 1.0 / 256.0
q = q * scale
output = flash_attention(q, k, v)
output = output / scale
```

3. **梯度缩放（训练）：**
```python
from torch.cuda.amp import GradScaler

scaler = GradScaler()
with torch.cuda.amp.autocast():
    output = flash_attention(q, k, v)
scaler.scale(loss).backward()
```

### 输出与 PyTorch 不匹配

**症状：**
- 自定义内核输出与 `torch.nn.functional.scaled_dot_product_attention` 不同

**解决方案：**

1. **检查容差：**
```python
torch.testing.assert_close(
    output_custom,
    output_torch,
    rtol=1e-3,  # 相对容差
    atol=1e-3   # 绝对容差
)
```

2. **预期差异：**
```python
# FP16 有约 3-4 位十进制数精度
# 不同实现间的微小差异是正常的
```

---

## 获取帮助

### 诊断脚本

运行此脚本收集系统信息：

```python
#!/usr/bin/env python3
import sys
import torch
import cuda_llm_ops

print("=" * 60)
print("系统信息")
print("=" * 60)
print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"设备数量: {torch.cuda.device_count()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
    print(f"计算能力: {torch.cuda.get_device_capability()}")
    print(f"cuda_llm_ops 版本: {cuda_llm_ops.__version__}")

print("=" * 60)
print("快速测试")
print("=" * 60)

try:
    q = torch.randn(2, 4, 64, 32, device='cuda', dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output = cuda_llm_ops.flash_attention(q, k, v)
    print("✓ FlashAttention 测试通过")
except Exception as e:
    print(f"✗ FlashAttention 测试失败: {e}")

try:
    a = torch.randn(512, 512, device='cuda', dtype=torch.float16)
    b = torch.randn(512, 512, device='cuda', dtype=torch.float16)
    c = cuda_llm_ops.gemm(a, b)
    print("✓ GEMM 测试通过")
except Exception as e:
    print(f"✗ GEMM 测试失败: {e}")
```

### 提交 Issue

报告问题时请包含：

1. **系统信息**（来自上面的脚本）
2. **最小复现代码**
3. **预期 vs 实际行为**
4. **完整错误信息和堆栈跟踪**

示例 Issue 模板：

```markdown
## 环境
- GPU: NVIDIA A100
- CUDA: 12.1
- Python: 3.10
- PyTorch: 2.1.0
- cuda_llm_ops: 0.3.0

## 问题描述
FlashAttention 在 8K 序列长度时 OOM

## 复现代码
```python
import torch
from cuda_llm_ops import flash_attention

q = torch.randn(2, 16, 8192, 64, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
output = flash_attention(q, k, v)  # 这里 OOM
```

## 错误信息
RuntimeError: CUDA out of memory. Tried to allocate 4.00 GB
```

### 资源

- **GitHub Issues**: https://github.com/LessUp/llm-speed/issues
- **文档**: https://lessup.github.io/llm-speed/
- **Discussions**: https://github.com/LessUp/llm-speed/discussions

---

[← 返回文档](../)
