# CUDA LLM Kernel Optimization

[![CI](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/llm-speed/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

English | [简体中文](README.zh-CN.md) | [Docs](https://lessup.github.io/llm-speed/)

![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

A high-performance CUDA kernel library for LLM inference, featuring FlashAttention, Tensor Core GEMM, and PyTorch bindings.

## Features

### Attention Kernels
- **Naive Attention**: Baseline implementation with O(N²) memory complexity
- **Tiled Attention**: Shared memory optimization with block-wise computation
- **FlashAttention**: O(N) memory complexity using online softmax algorithm
  - Causal mask support for autoregressive generation
  - Double buffering for compute/memory overlap

### GEMM Kernels
- **High-Performance GEMM**: Register tiling with 3-level blocking strategy
- **Tensor Core GEMM**: Hardware-accelerated matrix multiplication using WMMA
  - FP16 input with FP32 accumulation
  - INT8 quantized GEMM (requires Turing+ SM≥7.2)
- **Matrix Layout Support**: NN, NT, TN, TT transpose combinations

### Technical Highlights
- Shared memory padding to eliminate bank conflicts
- Warp-level primitives for efficient reduction
- Double buffering pipeline for latency hiding
- Async copy support for Ampere+ architecture
- Comprehensive input validation and error handling

## Requirements

| Component | Version |
|-----------|---------|
| CUDA | 11.0+ |
| Python | 3.8+ |
| PyTorch | 2.0+ |
| GPU | SM 7.0+ (Volta) |

### Supported GPU Architectures

| Architecture | SM Version | Tensor Core |
|--------------|------------|-------------|
| Volta | SM 7.0 | FP16 |
| Turing | SM 7.5 | FP16, INT8 |
| Ampere | SM 8.0, 8.6 | FP16, BF16, INT8, TF32 |
| Ada Lovelace | SM 8.9 | FP16, BF16, INT8, FP8 |
| Hopper | SM 9.0 | FP16, BF16, INT8, FP8 |

## Installation

### Quick Install

```bash
# Clone repository
git clone https://github.com/LessUp/llm-speed.git
cd llm-speed

# Install dependencies
pip install -r requirements.txt

# Build and install CUDA extension
pip install -e .
```

### Build with Specific CUDA Architectures

```bash
# Build for specific GPU (e.g., A100 = SM 8.0)
CUDA_ARCHS="80" pip install -e .

# Build for multiple architectures
CUDA_ARCHS="80;86;89" pip install -e .
```

### Alternative: CMake Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Quick Start

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
import torch
from cuda_llm_ops import gemm, tensor_core_gemm

# Standard GEMM
a = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
b = torch.randn(512, 1024, device='cuda', dtype=torch.float16)
c = gemm(a, b)

# With scaling
c = gemm(a, b, alpha=2.0, beta=0.5)

# Tensor Core GEMM (FP16 input, FP32 output)
c = tensor_core_gemm(a, b)
```

## API Reference

### Attention Functions

| Function | Description | Memory |
|----------|-------------|--------|
| `naive_attention(q, k, v, scale=0.0)` | Baseline implementation | O(N²) |
| `tiled_attention(q, k, v, scale=0.0)` | Shared memory tiling | O(N²) |
| `flash_attention(q, k, v, scale=0.0, is_causal=False)` | Online softmax | O(N) |

**Parameters:**
- `q, k, v`: Input tensors `[batch, heads, seq_len, head_dim]`
- `scale`: Attention scale factor (default: `1/√head_dim`)
- `is_causal`: Enable causal mask (FlashAttention only)

### GEMM Functions

| Function | Description | Precision |
|----------|-------------|-----------|
| `gemm(a, b, alpha=1.0, beta=0.0, trans_a=False, trans_b=False)` | High-performance GEMM | FP16/FP32 |
| `tensor_core_gemm(a, b, alpha=1.0, beta=0.0)` | Tensor Core accelerated | FP16→FP32 |
| `tensor_core_gemm_int8(a, b)` | INT8 quantized GEMM | INT8→INT32 |

## Testing

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

## Benchmarking

```bash
# Attention benchmark
python benchmarks/benchmark_attention.py --seq-lengths 512 1024 2048 4096

# GEMM benchmark
python benchmarks/benchmark_gemm.py --sizes 1024x1024x1024 2048x2048x2048

# Export results to JSON
python benchmarks/benchmark_attention.py --output results.json
```

## Project Structure

```
llm-speed/
├── src/                    # CUDA kernel implementations
│   ├── naive_attention.cu  # Baseline attention
│   ├── tiled_attention.cu  # Tiled optimization
│   ├── flash_attention.cu  # FlashAttention (O(N) memory)
│   ├── tensor_core_gemm.cu # Tensor Core GEMM
│   └── hgemm_kernel.cu     # High-performance GEMM
├── include/                # Header primitives
│   ├── common.cuh          # Core types and utilities
│   ├── online_softmax.cuh  # Online softmax algorithm
│   ├── warp_primitives.cuh # Warp-level operations
│   ├── shared_memory.cuh   # Shared memory management
│   └── pipeline.cuh        # Pipeline utilities
├── python/                 # Python bindings
│   ├── bindings.cpp        # pybind11 bindings
│   ├── __init__.py         # Module interface
│   └── profiler.py         # Performance profiler
├── tests/                  # Test suite
│   ├── conftest.py         # Test fixtures
│   ├── test_attention.py   # Attention tests
│   ├── test_gemm.py        # GEMM tests
│   └── test_interface.py   # Interface tests
├── benchmarks/             # Benchmark scripts
├── docs/                   # Documentation
└── changelog/              # Change history
```

## Documentation

- [API Reference](docs/api.md) - Detailed API documentation
- [DeepWiki](docs/deepwiki.md) - Technical deep dive
- [Performance Guide](docs/performance.md) - Optimization tips
- [Contributing](CONTRIBUTING.md) - Contribution guidelines

## Performance

### FlashAttention Memory Usage

| Sequence Length | Standard Attention | FlashAttention | Reduction |
|-----------------|-------------------|----------------|-----------|
| 1024 | 4 MB | 0.25 MB | 94% |
| 2048 | 16 MB | 0.5 MB | 97% |
| 4096 | 64 MB | 1 MB | 98% |

### GEMM Performance Target

Target: ≥90% of cuBLAS performance for matrices ≥1024×1024

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on:
- Development workflow
- Code style guidelines
- Testing requirements
- Commit message conventions

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

1. **FlashAttention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
2. **FlashAttention-2**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023
3. **CUTLASS**: NVIDIA CUTLASS - CUDA Templates for Linear Algebra Subroutines
