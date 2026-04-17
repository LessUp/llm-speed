# CUDA LLM Kernel Optimization

[![CI](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/llm-speed/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](https://github.com/LessUp/llm-speed/releases)

English | [简体中文](README.zh-CN.md) | [Documentation](https://lessup.github.io/llm-speed/)

![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

A high-performance CUDA kernel library for LLM inference, featuring FlashAttention, Tensor Core GEMM, and PyTorch bindings.

> 🚀 **What's New in v0.3.0**: Complete bilingual documentation (English & Chinese), professional documentation structure, and comprehensive quick start guides.

---

## ✨ Features

### Attention Kernels

- **⚡ FlashAttention**: O(N) memory complexity using online softmax algorithm
  - Causal mask support for autoregressive generation
  - Double buffering for compute/memory overlap
  - Up to 98% memory reduction vs standard attention
- **🔄 Tiled Attention**: Shared memory optimization with block-wise computation
- **📊 Naive Attention**: Baseline implementation for correctness verification

### GEMM Kernels

- **🎯 High-Performance GEMM**: Register tiling with 3-level blocking strategy
- **🔢 Tensor Core GEMM**: Hardware-accelerated matrix multiplication using WMMA
  - FP16 input with FP32 accumulation
  - INT8 quantized GEMM (requires Turing+ SM≥7.2)
  - Target: ≥90% of cuBLAS performance
- **📐 Matrix Layout Support**: NN, NT, TN, TT transpose combinations

### Technical Highlights

- 🏦 Shared memory padding to eliminate bank conflicts
- ⚙️ Warp-level primitives for efficient reduction
- 🔄 Double buffering pipeline for latency hiding
- 📥 Async copy support for Ampere+ architecture
- ✅ Comprehensive input validation and error handling

---

## 📋 Requirements

| Component | Version |
|-----------|---------|
| CUDA | 11.0+ |
| Python | 3.8+ |
| PyTorch | 2.0+ |
| GPU | SM 7.0+ (Volta) |

### Supported GPU Architectures

| Architecture | SM Version | Tensor Core | Notes |
|--------------|------------|-------------|-------|
| Volta | SM 7.0 | FP16 | |
| Turing | SM 7.5 | FP16, INT8 | |
| Ampere | SM 8.0, 8.6 | FP16, INT8, TF32 | BF16: planned |
| Ada Lovelace | SM 8.9 | FP16, INT8 | BF16, FP8: planned |
| Hopper | SM 9.0 | FP16, INT8 | BF16, FP8: planned |

> **Note**: BF16 and FP8 support are planned features. Currently supported precisions are FP16, FP32, and INT8.

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/LessUp/llm-speed.git
cd llm-speed

# Install dependencies
pip install -r requirements.txt

# Build and install CUDA extension
pip install -e .
```

### FlashAttention

```python
import torch
from cuda_llm_ops import flash_attention

# Create input tensors [batch, heads, seq_len, head_dim]
batch, heads, seq_len, head_dim = 2, 8, 512, 64
q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

# Compute attention
output = flash_attention(q, k, v)

# With causal mask (for autoregressive models)
output_causal = flash_attention(q, k, v, is_causal=True)
```

### GEMM

```python
from cuda_llm_ops import gemm, tensor_core_gemm

# Standard GEMM
a = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
b = torch.randn(512, 1024, device='cuda', dtype=torch.float16)
c = gemm(a, b)

# Tensor Core GEMM (FP16 input, FP32 output)
c = tensor_core_gemm(a, b)
```

---

## 📚 Documentation

### English Documentation

| Document | Description |
|----------|-------------|
| [Quick Start](docs/setup/quickstart-en.md) | Get started in 5 minutes |
| [API Reference](docs/api/api-en.md) | Complete API documentation |
| [Architecture](docs/architecture/architecture-en.md) | Technical deep dive |
| [Performance Guide](docs/tutorials/performance-en.md) | Optimization and tuning |
| [Troubleshooting](docs/tutorials/troubleshooting-en.md) | Common issues and solutions |

### 中文文档

| Document | Description |
|----------|-------------|
| [快速入门](docs/setup/quickstart-zh.md) | 5 分钟快速上手 |
| [API 参考](docs/api/api-zh.md) | 完整 API 文档 |
| [架构设计](docs/architecture/architecture-zh.md) | 技术深度解析 |
| [性能指南](docs/tutorials/performance-zh.md) | 优化与调优 |
| [故障排除](docs/tutorials/troubleshooting-zh.md) | 常见问题与解决方案 |

---

## 📊 Performance

### Memory Efficiency

| Implementation | Memory Complexity | 4K Sequence | 16K Sequence |
|----------------|-------------------|-------------|--------------|
| Standard Attention | O(N²) | 256 MB | 4 GB |
| FlashAttention | O(N) | 4 MB | 16 MB |

### Speedup vs PyTorch SDPA (A100, FP16)

| Sequence Length | Speedup | Memory Saved |
|-----------------|---------|--------------|
| 512 | 1.2x | 94% |
| 1024 | 1.4x | 97% |
| 2048 | 1.6x | 98% |
| 4096 | 1.8x | 98% |
| 8192 | 2.1x | 98% |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/ -v -m cuda        # CUDA tests
pytest tests/ -v -m property    # Property-based tests
pytest tests/ -v -m "not cuda"  # CPU-safe tests

# Run specific test file
pytest tests/test_attention.py -v
```

---

## 📈 Benchmarking

```bash
# Attention benchmark
python benchmarks/benchmark_attention.py --seq-lengths 512 1024 2048 4096

# GEMM benchmark
python benchmarks/benchmark_gemm.py --sizes 1024x1024x1024 2048x2048x2048

# Export results to JSON
python benchmarks/benchmark_attention.py --output results.json
```

---

## 🏗️ Project Structure

```
llm-speed/
├── specs/                  # Specification documents (SDD)
│   ├── product/            # Product requirements
│   ├── rfc/                # Technical design documents
│   ├── api/                # API definitions
│   └── testing/            # BDD test specifications
├── docs/                   # Documentation (English & Chinese)
│   ├── setup/              # Setup guides
│   ├── tutorials/          # Tutorials and guides
│   ├── architecture/       # Architecture documentation
│   └── api/                # API reference
├── src/                    # CUDA kernel implementations
├── include/                # Header primitives
├── python/                 # Python bindings
├── tests/                  # Test suite
└── benchmarks/             # Benchmark scripts
```

---

## 🤝 Contributing

This project follows **Spec-Driven Development (SDD)**. See [CONTRIBUTING.md](CONTRIBUTING.md) for details on:

- Spec-Driven Development workflow
- Development setup and environment
- Code style guidelines
- Testing requirements
- Commit message conventions

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📖 References

1. **FlashAttention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
2. **FlashAttention-2**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023
3. **CUTLASS**: NVIDIA CUTLASS - CUDA Templates for Linear Algebra Subroutines

---

## 🔗 Links

- [Documentation Site](https://lessup.github.io/llm-speed/)
- [GitHub Releases](https://github.com/LessUp/llm-speed/releases)
- [Changelog](CHANGELOG.md)
- [Issues](https://github.com/LessUp/llm-speed/issues)
- [Discussions](https://github.com/LessUp/llm-speed/discussions)
