# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Build and install CUDA extension (development mode)
pip install -e .

# Build with CMake (alternative)
mkdir build && cd build
cmake .. && make

# Set specific CUDA architectures (optional)
CUDA_ARCHS="80;86" pip install -e .
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run property-based tests
pytest tests/ -v -m property

# Run single test file
pytest tests/test_attention.py -v

# Run single test function
pytest tests/test_attention.py::test_function_name -v
```

## Benchmarks

```bash
python benchmarks/benchmark_attention.py
python benchmarks/benchmark_gemm.py
```

## Architecture

This is a CUDA kernel optimization library for LLM inference, providing FlashAttention and high-performance GEMM implementations with PyTorch bindings.

### Core Components

**CUDA Kernels (`src/`):**
- `naive_attention.cu` → `tiled_attention.cu` → `flash_attention.cu`: Progressive attention optimization (baseline → tiled → O(N) memory FlashAttention with online softmax)
- `tensor_core_gemm.cu`, `hgemm_kernel.cu`: High-performance GEMM using Tensor Cores, supporting FP16/INT8

**Header Primitives (`include/`):**
- `common.cuh`: Core types (`AttentionConfig`, `GemmConfig`, `KernelMetrics`), CUDA_CHECK macro, utility functions
- `online_softmax.cuh`: Online softmax algorithm for FlashAttention
- `warp_primitives.cuh`: Warp-level operations
- `shared_memory.cuh`: Shared memory management
- `pipeline.cuh`: Memory prefetch pipeline utilities

**Python Bindings (`python/`):**
- `bindings.cpp`: pybind11 bindings exposing `flash_attention()` and `gemm()` functions
- Module name: `cuda_llm_ops`

### Key Patterns

- Tensor Core alignment requirement: dimensions should be multiples of 16
- Supported GPU architectures: SM 70 (Volta) through SM 90 (Hopper)
- FP16 is the primary precision for Tensor Core operations
- Tests use `assert_close()` from `conftest.py` for numerical validation with configurable tolerances
