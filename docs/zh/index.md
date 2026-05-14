---
layout: home
title: LLM-Speed
titleTemplate: LLM推理CUDA内核库

hero:
  name: LLM-Speed
  text: LLM推理CUDA内核库
  tagline: 专注于LLM推理原语的CUDA内核库，实现FlashAttention前向传播、Tensor Core GEMM加速，以及无缝PyTorch集成。为现代GPU上的高效LLM推理而设计。
  image:
    src: /assets/images/logo.svg
    alt: LLM-Speed Logo
  actions:
    - theme: brand
      text: 🚀 快速开始
      link: /zh/setup/
    - theme: alt
      text: 📖 架构设计
      link: /zh/architecture/
    - theme: alt
      text: 📊 性能指南
      link: /zh/tutorials/performance

features:
  - icon: ⚡
    title: FlashAttention
    details: 使用在线Softmax算法实现O(N)内存复杂度。支持自回归模型的因果掩码。
  - icon: 🔢
    title: Tensor Core GEMM
    details: 使用WMMA API进行硬件加速矩阵乘法。FP16输入，FP32累加。
  - icon: 🐍
    title: PyTorch集成
    details: 通过pybind11与PyTorch无缝集成。原生CUDA张量支持。
  - icon: 🔄
    title: 双缓冲
    details: 流水线执行实现计算/内存重叠。Ampere+架构支持异步拷贝。
  - icon: 🏦
    title: 无Bank冲突
    details: 精心设计的共享内存布局，通过填充消除bank冲突。
  - icon: 📊
    title: 属性测试
    details: 使用Hypothesis进行全面测试，验证边界条件的正确性。
---

<div class="badges-container">
  <img src="https://github.com/LessUp/llm-speed/actions/workflows/ci.yml/badge.svg" alt="CI" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License" />
  <img src="https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white" alt="CUDA" />
  <img src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white" alt="C++" />
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white" alt="Python" />
</div>

## 内存高效设计

FlashAttention使用在线Softmax实现**O(N)内存复杂度**，而非标准注意力的O(N²)。

| 序列长度 | 标准注意力 | FlashAttention | 内存节省 |
|---------|-----------|----------------|---------|
| 1024 | 4 MB (完整注意力矩阵) | 0.25 MB (流式) | **16×** |
| 4096 | 64 MB (完整注意力矩阵) | 1 MB (流式) | **64×** |
| 8192 | 256 MB (完整注意力矩阵) | 2 MB (流式) | **128×** |

*假设8个注意力头，FP32累加，批次大小为1。*

## 快速示例

只需几行代码即可开始：

<div class="colab-container">
  <a href="https://colab.research.google.com/github/LessUp/llm-speed/blob/main/notebooks/llm-speed-demo.ipynb" target="_blank" rel="noopener">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" />
  </a>
  <span class="colab-hint">在浏览器中体验 - 无需本地GPU！</span>
</div>

::: code-group

```python [flash_attention.py]
import torch
from cuda_llm_ops import flash_attention

# 创建输入
batch, heads = 2, 8
seq_len, head_dim = 2048, 64

q = torch.randn(batch, heads, seq_len, head_dim,
                device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

# O(N)内存注意力！
output = flash_attention(q, k, v, is_causal=True)
```

```python [tensor_core_gemm.py]
import torch
from cuda_llm_ops import tensor_core_gemm

# 矩阵乘法
a = torch.randn(1024, 512, device='cuda',
                dtype=torch.float16)
b = torch.randn(512, 1024, device='cuda',
                dtype=torch.float16)

# 硬件加速GEMM
# FP16输入 → FP32输出
c = tensor_core_gemm(a, b)
print(c.dtype)  # torch.float32
```

:::

## GPU架构支持

针对**Ampere (A100, RTX 30/40)**及更新架构优化。支持Hopper及未来架构的前向兼容。

| 架构 | Tensor Core | 状态 |
|-----|-------------|------|
| **Ampere** (A100, RTX 30/40) | WMMA FP16, BF16, TF32 | ✅ 主要目标 |
| **Hopper** (H100) | WMMA FP16, BF16, FP8 | ✅ 支持 |
| **Volta** (V100) | WMMA FP16 | ⚠️ 有限 |
| **Turing** (T4, RTX 20) | WMMA FP16, INT8 | ⚠️ 有限 |

## 文档

<div class="feature-grid">

### 🚀 快速开始
5分钟内完成安装并运行基本示例。

[快速开始 →](/zh/setup/)

### 📚 API参考
完整的API文档，包含参数、示例和错误处理。

[API参考 →](/zh/api/)

### 🏗️ 架构设计
深入探讨CUDA内核、优化策略和实现细节。

[架构设计 →](/zh/architecture/)

### ⚡ 性能指南
优化技巧、基准测试工具和最佳实践。

[性能指南 →](/zh/tutorials/performance)

</div>

## 开始使用LLM-Speed

三条清晰的路径获取价值：

| 路径 | 描述 |
|------|------|
| 🚀 **快速开始** | 5分钟内安装并运行你的第一个FlashAttention或Tensor Core GEMM示例。 |
| 🏗️ **理解架构** | 探索内核设计、内存布局优化和Tensor Core利用模式。 |
| 📊 **本地基准测试** | 在你的GPU上运行性能基准测试。使用提供的工具查看内存使用和加速效果。 |

<div class="cta-buttons">
  <a href="https://github.com/LessUp/llm-speed" class="btn btn-secondary">💻 浏览源码</a>
  <a href="/zh/api/" class="btn btn-secondary">📚 API参考</a>
</div>

<style>
.badges-container {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  margin: 2rem 0;
}

.badges-container img {
  height: 20px;
}

.colab-container {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin: 1rem 0 2rem 0;
}

.colab-container img {
  height: 28px;
}

.colab-hint {
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin: 1.5rem 0;
}

.feature-grid h3 {
  margin-top: 0;
  margin-bottom: 0.5rem;
}

.feature-grid p {
  margin-bottom: 0.5rem;
}

.feature-grid a {
  font-weight: 500;
}

.cta-buttons {
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin-top: 2rem;
}

.btn {
  display: inline-flex;
  align-items: center;
  padding: 10px 20px;
  border-radius: 8px;
  font-weight: 500;
  text-decoration: none;
  transition: all 0.2s ease;
}

.btn-secondary {
  background: var(--surface-primary);
  border: 1px solid var(--vp-c-border);
  color: var(--vp-c-text-1);
}

.btn-secondary:hover {
  border-color: var(--vp-c-brand-1);
  background: var(--surface-secondary);
}
</style>
