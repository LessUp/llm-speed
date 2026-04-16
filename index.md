---
layout: default
title: LLM-Speed
description: High-performance CUDA kernel library for LLM inference — FlashAttention, Tensor Core GEMM, and Python bindings
---

# LLM-Speed

<div class="badges">
  <img src="https://github.com/LessUp/llm-speed/actions/workflows/ci.yml/badge.svg" alt="CI Status">
  <img src="https://github.com/LessUp/llm-speed/actions/workflows/pages.yml/badge.svg" alt="Pages Status">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white" alt="C++">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white" alt="Python">
</div>

**LLM-Speed** is a high-performance CUDA kernel library for LLM inference, featuring FlashAttention with O(N) memory complexity, Tensor Core GEMM with FP16/INT8 support, and seamless PyTorch integration via pybind11.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/LessUp/llm-speed.git
cd llm-speed

# Install dependencies and build
pip install -r requirements.txt
pip install -e .
```

### Usage

```python
import torch
from cuda_llm_ops import flash_attention, gemm

# FlashAttention
q = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
output = flash_attention(q, k, v)

# GEMM
a = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
b = torch.randn(512, 1024, device='cuda', dtype=torch.float16)
c = gemm(a, b)
```

## Documentation

| Section | Description |
|---------|-------------|
| [User Guide](docs/deepwiki) | Technical deep dive into CUDA kernels and optimization strategies |
| [API Reference](docs/api) | Complete API documentation with examples |
| [Performance Guide](docs/performance) | Hardware requirements, benchmarking, and optimization tips |
| [Contributing](https://github.com/LessUp/llm-speed/blob/master/CONTRIBUTING.md) | Development workflow and coding standards |
| [Changelog](changelog/) | Version history and release notes |

## Features

### Attention Kernels

| Kernel | Memory | Description |
|--------|--------|-------------|
| `naive_attention` | O(N²) | Baseline implementation for correctness verification |
| `tiled_attention` | O(N²) | Shared memory tiling optimization |
| `flash_attention` | O(N) | Online softmax algorithm with causal mask support |

### GEMM Kernels

| Kernel | Precision | Description |
|--------|-----------|-------------|
| `gemm` | FP16/FP32 | High-performance GEMM with register tiling |
| `tensor_core_gemm` | FP16→FP32 | Tensor Core accelerated with WMMA |
| `tensor_core_gemm_int8` | INT8→INT32 | Quantized GEMM (Turing+ SM≥7.2) |

### Key Highlights

- **O(N) Memory**: FlashAttention reduces memory from O(N²) to O(N)
- **Tensor Core Acceleration**: WMMA API for hardware-accelerated matrix operations
- **Double Buffering**: Compute/memory overlap for improved throughput
- **Bank Conflict Elimination**: Padding strategies for optimal memory access
- **Python Integration**: Native PyTorch Tensor support via pybind11

## Who Should Use This?

- **Developers** learning CUDA kernel optimization for LLM inference
- **Engineers** implementing attention mechanisms and matrix operations
- **Researchers** studying FlashAttention and Tensor Core programming
- **Maintainers** needing reliable tests and comprehensive documentation

## GPU Architecture Support

| Architecture | SM | Tensor Core | Status |
|--------------|-----|-------------|--------|
| Volta | 7.0 | FP16 | ✅ Supported |
| Turing | 7.5 | FP16, INT8 | ✅ Supported |
| Ampere | 8.0, 8.6 | FP16, BF16, INT8 | ✅ Recommended |
| Ada Lovelace | 8.9 | FP16, BF16, INT8, FP8 | ✅ Supported |
| Hopper | 9.0 | FP16, BF16, INT8, FP8 | ✅ Supported |

## Performance

### FlashAttention Memory Reduction

| Sequence Length | Standard | FlashAttention | Reduction |
|-----------------|----------|----------------|-----------|
| 1024 | 4 MB | 0.25 MB | 94% |
| 2048 | 16 MB | 0.5 MB | 97% |
| 4096 | 64 MB | 1 MB | 98% |

### GEMM Target

Target: **≥90% of cuBLAS performance** for matrices ≥1024×1024

## Resources

- [GitHub Repository](https://github.com/LessUp/llm-speed)
- [Issue Tracker](https://github.com/LessUp/llm-speed/issues)
- [Discussions](https://github.com/LessUp/llm-speed/discussions)

## License

[MIT License](https://github.com/LessUp/llm-speed/blob/master/LICENSE)

## References

1. **FlashAttention**: Dao et al., NeurIPS 2022
2. **FlashAttention-2**: Dao, 2023
3. **CUTLASS**: NVIDIA CUDA Templates for Linear Algebra
4. **CUDA Programming Guide**: NVIDIA
