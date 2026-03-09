# CUDA LLM Kernel Optimization

[![CI](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)

[简体中文](README.md) | English

High-performance CUDA operator library for LLM inference optimization, including FlashAttention and high-performance GEMM kernels.

## Features

- **FlashAttention**: Online Softmax, O(N) memory, causal mask support
- **High-Performance GEMM**: FP32/FP16/INT8 mixed precision, Tensor Core (WMMA)
- **Progressive Optimization**: Naive → Tiled → FlashAttention (double-buffered)
- **Register Tiling GEMM**: 128×128 blocks + 8×8 register accumulation + double buffer pipeline
- **PyTorch Integration**: pybind11 Python bindings, direct PyTorch Tensor I/O
- **Property Testing**: Hypothesis-driven property-based tests

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### CMake Build

```bash
cmake --preset release
cmake --build --preset release
```

## Usage

```python
from cuda_llm_ops import flash_attention, gemm, tensor_core_gemm

# FlashAttention (causal mask)
output = flash_attention(q, k, v, is_causal=True)

# High-performance GEMM
c = gemm(a, b, alpha=1.0, beta=0.0)

# Tensor Core GEMM (FP16 → FP32)
c_fp32 = tensor_core_gemm(a, b)
```

## Testing

```bash
pytest tests/ -v                         # All tests
pytest tests/ -v -m property             # Property tests
python benchmarks/benchmark_attention.py # Benchmarks
```

## GPU Architecture Support

| Arch | SM | Features |
|------|-----|----------|
| Volta | 7.0 | FP16 Tensor Core |
| Turing | 7.5 | FP16 + INT8 |
| Ampere | 8.0, 8.6 | TF32 + async copy |
| Ada | 8.9 | FP8 |
| Hopper | 9.0 | TMA + Warp Group MMA |

## License

MIT License
