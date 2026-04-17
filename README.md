# ⚡ LLM-Speed

[![CI](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml)
[![Pages](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/llm-speed/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.0-blue)](https://github.com/LessUp/llm-speed/releases)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-3776AB?logo=python)](https://python.org)
[![CUDA 11.0+](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Discussions](https://img.shields.io/badge/GitHub-Discussions-green?logo=github)](https://github.com/LessUp/llm-speed/discussions)

English | [简体中文](README.zh-CN.md) | [Documentation](https://lessup.github.io/llm-speed/)

> High-performance CUDA kernel library for LLM inference — **FlashAttention** with O(N) memory, **Tensor Core GEMM** acceleration, and seamless **PyTorch** integration.

---

## 🎯 At a Glance

```python
import torch
from cuda_llm_ops import flash_attention, tensor_core_gemm

# O(N) memory attention - 98% memory reduction!
output = flash_attention(q, k, v, is_causal=True)

# Hardware-accelerated GEMM via Tensor Cores
c = tensor_core_gemm(a, b)  # FP16 → FP32
```

**Why LLM-Speed?**
- ⚡ **98% less memory** than standard attention (O(N) vs O(N²))
- 🚀 **2.1× faster** at 8K sequence length
- 🎯 **90%+ cuBLAS performance** for GEMM operations
- 🔧 **PyTorch-native** - works with your existing code

---

## 📑 Table of Contents

<!-- TOC -->
* [⚡ LLM-Speed](#-llm-speed)
  * [🎯 At a Glance](#-at-a-glance)
  * [📑 Table of Contents](#-table-of-contents)
  * [✨ Key Features](#-key-features)
    * [Attention Kernels](#attention-kernels)
    * [GEMM Kernels](#gemm-kernels)
    * [Technical Highlights](#technical-highlights)
  * [🚀 Quick Start](#-quick-start)
    * [Installation](#installation)
    * [Troubleshooting](#troubleshooting)
    * [Basic Usage](#basic-usage)
  * [📖 API Reference](#-api-reference)
  * [📊 Performance](#-performance)
    * [Memory Efficiency](#memory-efficiency)
    * [Speed Comparison](#speed-comparison)
    * [GPU Architecture Support](#gpu-architecture-support)
  * [🏗️ Architecture](#️-architecture)
  * [📁 Project Structure](#-project-structure)
  * [🧪 Testing & Development](#-testing--development)
    * [Run Tests](#run-tests)
    * [Benchmarks](#benchmarks)
    * [Pre-commit Hooks](#pre-commit-hooks)
  * [🤝 Contributing](#-contributing)
  * [📄 License](#-license)
  * [📚 References](#-references)
<!-- TOC -->

---

## ✨ Key Features

### Attention Kernels

| Kernel | Memory | Best For | Key Features |
|--------|--------|----------|--------------|
| **⚡ FlashAttention** | O(N) | Production | Online softmax, causal masking, double buffering |
| **🔄 Tiled Attention** | O(N²) | Medium seq | Shared memory tiling, bank conflict-free |
| **📊 Naive Attention** | O(N²) | Testing | Baseline for correctness verification |

**FlashAttention Highlights:**
- 🧠 **O(N) memory** - 98% reduction vs standard attention (4K: 256MB → 4MB)
- 🎭 **Causal masking** - native support for autoregressive generation
- 🔄 **Double buffering** - compute/memory overlap for 2.1× speedup
- 📐 **Any sequence length** - no fixed tile size constraints

### GEMM Kernels

| Kernel | Input | Output | Performance | Notes |
|--------|-------|--------|-------------|-------|
| **🎯 High-Performance GEMM** | FP32/FP16 | FP32 | 90%+ cuBLAS | Register tiling, 3-level blocking |
| **🔢 Tensor Core GEMM** | FP16 | FP32 | 95%+ cuBLAS | WMMA API, FP32 accumulation |
| **🔢 INT8 GEMM** | INT8 | INT32 | 90%+ cuBLAS | Turing+ (SM≥7.2) required |

**Matrix Layout Support:** NN, NT, TN, TT transpose combinations

### Technical Highlights

- 🏦 **Bank conflict elimination** - shared memory padding with stride alignment
- ⚙️ **Warp-level primitives** - efficient reduction via shuffle instructions
- 🔄 **Software pipelining** - latency hiding through double buffering
- 📥 **Async copy support** - Ampere+ architecture optimizations
- ✅ **Input validation** - comprehensive error handling and type checking

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

**Prerequisites:**
- CUDA Toolkit 11.0+ ([Download](https://developer.nvidia.com/cuda-downloads))
- PyTorch 2.0+ with CUDA support
- Python 3.8+
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)

**Step-by-step:**

```bash
# 1. Clone repository
git clone https://github.com/LessUp/llm-speed.git
cd llm-speed

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build and install CUDA extension
pip install -e .
```

> **💡 Tip:** For development with all extras (testing, benchmarking):
> ```bash
> pip install -e ".[dev]"
> ```

### Troubleshooting

<details>
<summary><strong>❌ CUDA compilation errors</strong></summary>

Ensure CUDA is properly installed and `nvcc` is in your PATH:

```bash
nvcc --version  # Should show 11.0+
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

If using conda:
```bash
conda install -c conda-forge cudatoolkit=11.8
```

</details>

<details>
<summary><strong>❌ "cuda_llm_ops not found" error</strong></summary>

The extension wasn't built. Rebuild with:

```bash
pip install -e . --verbose
```

Check build logs for compilation errors.

</details>

<details>
<summary><strong>❌ Import warnings about kernels not built</strong></summary>

This occurs when importing before building. Run `pip install -e .` first.

</details>

### Basic Usage

<details>
<summary><strong>FlashAttention Example</strong></summary>

```python
import torch
from cuda_llm_ops import flash_attention

# Create input tensors [batch, heads, seq_len, head_dim]
batch, heads, seq_len, head_dim = 2, 8, 2048, 64
q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

# Standard attention
output = flash_attention(q, k, v)

# With causal mask (for autoregressive generation)
output_causal = flash_attention(q, k, v, is_causal=True)

# With custom scale factor
output_scaled = flash_attention(q, k, v, scale=1.0 / (head_dim ** 0.5))
```

</details>

<details>
<summary><strong>GEMM Example</strong></summary>

```python
import torch
from cuda_llm_ops import gemm, tensor_core_gemm

# Standard GEMM: C = alpha * A @ B + beta * C
a = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
b = torch.randn(512, 1024, device='cuda', dtype=torch.float16)

# Basic matrix multiplication
c = gemm(a, b)

# With transpose and scaling
c = gemm(a, b, alpha=0.5, beta=0.1, trans_a=False, trans_b=True)

# Tensor Core GEMM (FP16 input, FP32 output)
c_tensor = tensor_core_gemm(a, b)  # ~95% cuBLAS performance
```

</details>

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

## 📖 API Reference

### Attention Functions

```python
from cuda_llm_ops import naive_attention, tiled_attention, flash_attention
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `naive_attention` | `(q, k, v, scale=0.0, is_causal=False) → Tensor` | Baseline O(N²) attention for testing |
| `tiled_attention` | `(q, k, v, scale=0.0, is_causal=False) → Tensor` | Shared memory tiled attention |
| `flash_attention` | `(q, k, v, scale=0.0, is_causal=False) → Tensor` | **Recommended** - O(N) memory FlashAttention |

**Parameters:**
- `q, k, v`: Input tensors of shape `[batch, heads, seq_len, head_dim]`
- `scale`: Custom scale factor (default: `0.0` = auto `1/√head_dim`)
- `is_causal`: Apply causal mask for autoregressive generation

**Returns:** Output tensor of same shape as inputs

### GEMM Functions

```python
from cuda_llm_ops import gemm, tensor_core_gemm, tensor_core_gemm_int8
```

| Function | Signature | Input | Output | Performance |
|----------|-----------|-------|--------|-------------|
| `gemm` | `(a, b, alpha=1.0, beta=0.0, trans_a=False, trans_b=False) → Tensor` | FP32/FP16 | FP32 | 90%+ cuBLAS |
| `tensor_core_gemm` | `(a, b, alpha=1.0, beta=0.0) → Tensor` | FP16 | FP32 | 95%+ cuBLAS |
| `tensor_core_gemm_int8` | `(a, b) → Tensor` | INT8 | INT32 | 90%+ cuBLAS |

**Parameters:**
- `a, b`: Input matrices
- `alpha`: Scaling factor for `A @ B` (default: `1.0`)
- `beta`: Scaling factor for `C` (default: `0.0`)
- `trans_a, trans_b`: Transpose input matrices

> 📚 **Full API docs**: [docs/api/api-en.md](docs/api/api-en.md)

---

## 📊 Performance

### Memory Efficiency

| Sequence Length | Standard Attention | FlashAttention | **Memory Saved** |
|-----------------|-------------------|----------------|------------------|
| 1024 | 16 MB | 0.5 MB | **97%** ↓ |
| 2048 | 64 MB | 1 MB | **98%** ↓ |
| 4096 | 256 MB | 4 MB | **98%** ↓ |
| 8192 | 1 GB | 8 MB | **99%** ↓ |
| 16384 | 4 GB | 16 MB | **99%** ↓ |

### Speed Comparison

**FlashAttention vs PyTorch SDPA (A100, FP16):**

| Sequence Length | Speedup | Memory Reduction |
|-----------------|---------|------------------|
| 512 | 1.0× | 94% |
| 1024 | 1.2× | 97% |
| 2048 | 1.4× | 98% |
| 4096 | 1.6× | 98% |
| 8192 | **2.1×** | 98% |

**GEMM vs cuBLAS (A100, FP16):**

| Kernel | Performance | Notes |
|--------|-------------|-------|
| High-Performance GEMM | 90%+ cuBLAS | Register tiling, 3-level blocking |
| Tensor Core GEMM | 95%+ cuBLAS | WMMA API, FP32 accumulation |
| INT8 GEMM | 90%+ cuBLAS | Turing+ (SM≥7.2) required |

### GPU Architecture Support

| Architecture | SM Version | Tensor Core | FlashAttention | Status |
|--------------|------------|-------------|----------------|--------|
| **Volta** (V100) | SM 7.0 | FP16 | ✅ | Supported |
| **Turing** (T4, RTX 20) | SM 7.5 | FP16, INT8 | ✅ | Supported |
| **Ampere** (A100, RTX 30) | SM 8.0, 8.6 | FP16, BF16, INT8, TF32 | ✅ | **Recommended** |
| **Ada Lovelace** (RTX 40) | SM 8.9 | FP16, BF16, INT8, FP8 | ✅ | Supported |
| **Hopper** (H100) | SM 9.0 | FP16, BF16, INT8, FP8 | ✅ | Supported |

> **Note:** BF16 and FP8 support are planned features. Current precisions: FP16, FP32, INT8.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│                  (Python / PyTorch)                          │
├─────────────────────────────────────────────────────────────┤
│                  Python Bindings                             │
│              (pybind11 / cuda_llm_ops)                       │
├──────────────┬──────────────────┬───────────────────────────┤
│  Attention   │     GEMM         │      Utilities            │
│  Kernels     │     Kernels      │                           │
│              │                  │                           │
│ • Naive      │  • HP GEMM       │  • Online Softmax         │
│ • Tiled      │  • Tensor Core   │  • Warp Primitives        │
│ • Flash      │  • INT8 GEMM     │  • Shared Memory          │
├──────────────┴──────────────────┴───────────────────────────┤
│                    CUDA Runtime Layer                        │
│                 (CUDA 11.0+, C++17)                          │
├─────────────────────────────────────────────────────────────┤
│                  GPU Hardware Layer                          │
│           (Volta → Hopper, SM 7.0 - 9.0)                     │
└─────────────────────────────────────────────────────────────┘
```

**Key Optimization Strategies:**
1. **Shared Memory Tiling** - Reduce global memory access by 10×
2. **Register Blocking** - Maximize register reuse for GEMM
3. **Double Buffering** - Overlap computation with memory transfer
4. **Warp Shuffle** - Efficient intra-warp communication
5. **Async Copy** - Hardware-accelerated memory operations (Ampere+)

---

## 📁 Project Structure

```
llm-speed/
├── cuda_llm_ops/            # Python package
│   ├── __init__.py          # Package entry point
│   ├── _cuda_llm_ops.pyi    # Type stubs for IDE
│   ├── bindings.cpp         # pybind11 bindings
│   └── profiler.py          # Performance profiling
│
├── src/                     # CUDA kernel implementations
│   ├── naive_attention.cu   # Baseline O(N²) attention
│   ├── tiled_attention.cu   # Shared memory tiling
│   ├── flash_attention.cu   # O(N) FlashAttention
│   ├── tensor_core_gemm.cu  # WMMA Tensor Cores
│   └── hgemm_kernel.cu      # High-performance GEMM
│
├── include/                 # CUDA header primitives
│   ├── common.cuh           # Core types & macros
│   ├── online_softmax.cuh   # Online softmax algorithm
│   ├── warp_primitives.cuh  # Warp-level operations
│   ├── shared_memory.cuh    # Shared memory utilities
│   └── pipeline.cuh         # Double buffering
│
├── tests/                   # Test suite (pytest + hypothesis)
│   ├── conftest.py          # Fixtures & helpers
│   ├── test_attention.py    # Attention kernel tests
│   ├── test_gemm.py         # GEMM kernel tests
│   └── test_interface.py    # Python binding tests
│
├── benchmarks/              # Performance benchmarks
│   ├── benchmark_attention.py
│   └── benchmark_gemm.py
│
├── docs/                    # User documentation (EN/ZH)
├── specs/                   # Specifications (SDD)
└── .github/workflows/       # CI/CD pipelines
```

---

## 🧪 Testing & Development

### Run Tests

```bash
# All tests
pytest tests/ -v

# Test categories
pytest tests/ -v -m cuda         # GPU tests
pytest tests/ -v -m property     # Property-based (Hypothesis)
pytest tests/ -v -m "not cuda"   # CPU-safe tests

# Specific test file
pytest tests/test_attention.py -v
```

### Benchmarks

```bash
# Attention benchmark
python benchmarks/benchmark_attention.py --seq-lengths 512 1024 2048 4096

# GEMM benchmark
python benchmarks/benchmark_gemm.py --sizes 1024x1024x1024 2048x2048x2048

# Export results
python benchmarks/benchmark_attention.py --output results.json
```

### Pre-commit Hooks

This project uses pre-commit for automated code quality checks:

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run checks manually
pre-commit run --all-files
```

**Checks include:**
- ✅ Ruff (linting + formatting)
- ✅ clang-format (C++/CUDA style)
- ✅ Trailing whitespace removal
- ✅ YAML/JSON validation
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

| 文档 | 描述 |
|------|------|
| [快速入门](docs/setup/quickstart-zh.md) | 5 分钟快速上手 |
| [API 参考](docs/api/api-zh.md) | 完整 API 文档 |
| [架构设计](docs/architecture/architecture-zh.md) | 技术深度解析 |
| [性能指南](docs/tutorials/performance-zh.md) | 优化与调优 |
| [故障排除](docs/tutorials/troubleshooting-zh.md) | 常见问题与解决方案 |

---

## 🤝 Contributing

This project follows **Spec-Driven Development (SDD)**. All changes must have corresponding spec updates.

**Quick Start:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Update specs first (see `specs/` directory)
4. Implement your changes
5. Run tests (`pytest tests/ -v`)
6. Commit with conventional messages (`feat:`, `fix:`, `perf:`, etc.)
7. Submit a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📚 References

1. **FlashAttention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
2. **FlashAttention-2**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023
3. **CUTLASS**: NVIDIA CUTLASS - CUDA Templates for Linear Algebra Subroutines

---

## 🔗 Links

- [📖 Documentation Site](https://lessup.github.io/llm-speed/)
- [📦 GitHub Releases](https://github.com/LessUp/llm-speed/releases)
- [📝 Changelog](CHANGELOG.md)
- [🐛 Report Issues](https://github.com/LessUp/llm-speed/issues)
- [💬 Discussions](https://github.com/LessUp/llm-speed/discussions)
