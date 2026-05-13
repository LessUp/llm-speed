---
layout: home
title: LLM-Speed
titleTemplate: CUDA Kernels for LLM Inference

hero:
  name: LLM-Speed
  text: CUDA Kernels for LLM Inference
  tagline: A focused CUDA kernel library implementing FlashAttention forward, Tensor Core GEMM acceleration, and seamless PyTorch integration. Designed for efficient LLM inference on modern GPUs.
  image:
    src: /assets/images/logo.svg
    alt: LLM-Speed Logo
  actions:
    - theme: brand
      text: 🚀 Get Started
      link: /docs/setup/quickstart-en/
    - theme: alt
      text: 📖 Architecture
      link: /docs/architecture/architecture-en/
    - theme: alt
      text: 📊 Performance
      link: /docs/tutorials/performance-en/

features:
  - icon: ⚡
    title: FlashAttention
    details: O(N) memory complexity with online softmax algorithm. Supports causal masking for autoregressive models.
  - icon: 🔢
    title: Tensor Core GEMM
    details: Hardware-accelerated matrix multiplication using WMMA API. FP16 input with FP32 accumulation.
  - icon: 🐍
    title: PyTorch Integration
    details: Seamless integration with PyTorch via pybind11. Native CUDA tensor support.
  - icon: 🔄
    title: Double Buffering
    details: Compute/memory overlap with pipelined execution. Async copy for Ampere+ architectures.
  - icon: 🏦
    title: Bank Conflict Free
    details: Carefully designed shared memory layouts with padding to eliminate bank conflicts.
  - icon: 📊
    title: Property Testing
    details: Comprehensive tests with Hypothesis for correctness verification across edge cases.
---

<div class="badges-container">
  <img src="https://github.com/LessUp/llm-speed/actions/workflows/ci.yml/badge.svg" alt="CI" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License" />
  <img src="https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white" alt="CUDA" />
  <img src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white" alt="C++" />
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white" alt="Python" />
</div>

## Memory-Efficient Design

FlashAttention implements online softmax with **O(N) memory complexity** instead of O(N²) for standard attention.

| Sequence Length | Standard Attention | FlashAttention | Memory Savings |
|-----------------|-------------------|----------------|----------------|
| 1024 | 4 MB (full attention matrix) | 0.25 MB (streaming) | **16×** |
| 4096 | 64 MB (full attention matrix) | 1 MB (streaming) | **64×** |
| 8192 | 256 MB (full attention matrix) | 2 MB (streaming) | **128×** |

*Assumes 8 attention heads, FP32 accumulation, batch size 1.*

## Quick Example

Get started with just a few lines of code:

<div class="colab-container">
  <a href="https://colab.research.google.com/github/LessUp/llm-speed/blob/main/notebooks/llm-speed-demo.ipynb" target="_blank" rel="noopener">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" />
  </a>
  <span class="colab-hint">Try it in your browser - no GPU required!</span>
</div>

::: code-group

```python [flash_attention.py]
import torch
from cuda_llm_ops import flash_attention

# Create inputs
batch, heads = 2, 8
seq_len, head_dim = 2048, 64

q = torch.randn(batch, heads, seq_len, head_dim,
                device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

# O(N) memory attention!
output = flash_attention(q, k, v, is_causal=True)
```

```python [tensor_core_gemm.py]
import torch
from cuda_llm_ops import tensor_core_gemm

# Matrix multiplication
a = torch.randn(1024, 512, device='cuda',
                dtype=torch.float16)
b = torch.randn(512, 1024, device='cuda',
                dtype=torch.float16)

# Hardware accelerated GEMM
# FP16 input → FP32 output
c = tensor_core_gemm(a, b)
print(c.dtype)  # torch.float32
```

:::

## GPU Architecture Support

Optimized for **Ampere (A100, RTX 30/40)** and newer. Forward compatibility with Hopper and future architectures.

| Architecture | Tensor Core | Status |
|--------------|-------------|--------|
| **Ampere** (A100, RTX 30/40) | WMMA with FP16, BF16, TF32 | ✅ Primary target |
| **Hopper** (H100) | WMMA with FP16, BF16, FP8 | ✅ Supported |
| **Volta** (V100) | WMMA with FP16 | ⚠️ Limited |
| **Turing** (T4, RTX 20) | WMMA with FP16, INT8 | ⚠️ Limited |

## Documentation

<div class="feature-grid">

### 🚀 Quick Start
Get up and running in 5 minutes with installation and basic usage examples.

[Get Started →](/docs/setup/quickstart-en/)

### 📚 API Reference
Complete API documentation with parameters, examples, and error handling.

[API Reference →](/docs/api/api-en/)

### 🏗️ Architecture
Technical deep dive into CUDA kernels, optimization strategies, and implementation details.

[Architecture →](/docs/architecture/architecture-en/)

### ⚡ Performance Guide
Optimization tips, benchmarking tools, and best practices for maximum performance.

[Performance →](/docs/tutorials/performance-en/)

</div>

## Start Using LLM-Speed

Three clear paths to get value from this project:

| Path | Description |
|------|-------------|
| 🚀 **Get Started** | Install and run your first FlashAttention or Tensor Core GEMM example in 5 minutes. |
| 🏗️ **Understand Architecture** | Explore kernel design, memory layout optimization, and Tensor Core utilization patterns. |
| 📊 **Benchmark Locally** | Run performance benchmarks on your GPU. See memory usage and speedups with provided tools. |

<div class="cta-buttons">
  <a href="https://github.com/LessUp/llm-speed" class="btn btn-secondary">💻 Browse Source</a>
  <a href="/docs/api/api-en/" class="btn btn-secondary">📚 API Reference</a>
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
