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
      link: /en/setup/
    - theme: alt
      text: 📖 Architecture
      link: /en/architecture/
    - theme: alt
      text: 📊 Performance
      link: /en/tutorials/performance/

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
</style>
