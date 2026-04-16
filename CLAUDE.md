# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Build CUDA extension | `pip install -e .` |
| Run all tests | `pytest tests/ -v` |
| Run property tests | `pytest tests/ -v -m property` |
| Run CPU-safe tests | `pytest tests/ -v -m "not cuda"` |
| Lint check | `ruff check python/ tests/ benchmarks/` |
| Format code | `ruff format python/ tests/ benchmarks/` |
| Benchmark attention | `python benchmarks/benchmark_attention.py` |
| Benchmark GEMM | `python benchmarks/benchmark_gemm.py` |

---

## Build Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Build and install CUDA extension (development mode)
pip install -e .

# Build with verbose output (for debugging)
pip install -e . --verbose

# Build with CMake (alternative)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Set specific CUDA architectures (optional)
CUDA_ARCHS="80;86" pip install -e .
CUDA_ARCHS="80;86;89" pip install -e .  # A100 + A6000 + 4090
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run property-based tests (Hypothesis)
pytest tests/ -v -m property

# Run CUDA tests only
pytest tests/ -v -m cuda

# Run CPU-safe tests (no GPU required)
pytest tests/ -v -m "not cuda"

# Run single test file
pytest tests/test_attention.py -v

# Run single test class
pytest tests/test_attention.py::TestFlashAttention -v

# Run single test function
pytest tests/test_attention.py::TestFlashAttention::test_correctness -v

# Run with coverage
pytest tests/ --cov=python --cov-report=html
```

---

## Benchmarks

```bash
# Attention benchmark
python benchmarks/benchmark_attention.py \
    --seq-lengths 512 1024 2048 4096 \
    --batch-size 2 \
    --num-heads 16 \
    --dtype fp16

# GEMM benchmark
python benchmarks/benchmark_gemm.py \
    --sizes 1024x1024x1024 2048x2048x2048 \
    --dtype fp16

# Export results to JSON
python benchmarks/benchmark_attention.py --output results.json
```

---

## Architecture

This is a CUDA kernel optimization library for LLM inference, providing FlashAttention and high-performance GEMM implementations with PyTorch bindings.

### Optimization Roadmap

```
Naive → Tiled → FlashAttention → Tensor Core
  │        │          │              │
  │        │          │              └─ Hardware acceleration
  │        │          └─ O(N) memory, online softmax
  │        └─ Shared memory tiling
  └─ Baseline (O(N²) memory)
```

### Core Components

**CUDA Kernels (`src/`):**

| File | Description | Key Features |
|------|-------------|--------------|
| `naive_attention.cu` | Baseline attention | O(N²) memory, correctness reference |
| `tiled_attention.cu` | Tiled optimization | Shared memory, bank conflict padding |
| `flash_attention.cu` | FlashAttention | O(N) memory, online softmax, double buffering |
| `tensor_core_gemm.cu` | Tensor Core GEMM | WMMA API, FP16/INT8, tiled version |
| `hgemm_kernel.cu` | High-perf GEMM | Register tiling, double buffering, layout support |

**Header Primitives (`include/`):**

| File | Description |
|------|-------------|
| `common.cuh` | Core types (`AttentionConfig`, `GemmConfig`, `KernelMetrics`), CUDA_CHECK macro |
| `online_softmax.cuh` | Online softmax algorithm for FlashAttention |
| `warp_primitives.cuh` | Warp-level operations (reduce_sum, reduce_max, broadcast) |
| `shared_memory.cuh` | Shared memory management, padding utilities |
| `pipeline.cuh` | Double buffering, async copy (Ampere+), software pipeline |

**Python Bindings (`python/`):**

| File | Description |
|------|-------------|
| `bindings.cpp` | pybind11 bindings exposing all kernel functions |
| `__init__.py` | Module interface, exports all functions |
| `profiler.py` | Performance profiling utilities |

**Module name:** `cuda_llm_ops`

---

## API Reference

### Attention Functions

```python
from cuda_llm_ops import naive_attention, tiled_attention, flash_attention

# All functions share the same signature:
output = flash_attention(q, k, v, scale=0.0, is_causal=False)

# Input shape: [batch, heads, seq_len, head_dim]
# Output shape: [batch, heads, seq_len, head_dim]
# dtype: float32 or float16
# device: CUDA, contiguous
```

### GEMM Functions

```python
from cuda_llm_ops import gemm, tensor_core_gemm, tensor_core_gemm_int8

# Standard GEMM: C = alpha * A @ B + beta * C
c = gemm(a, b, alpha=1.0, beta=0.0, trans_a=False, trans_b=False)

# Tensor Core GEMM: FP16 input, FP32 output
c = tensor_core_gemm(a, b, alpha=1.0, beta=0.0)

# INT8 GEMM: INT8 input, INT32 output (requires Turing+ SM≥7.2)
c = tensor_core_gemm_int8(a, b)
```

---

## Key Patterns

### Tensor Core Alignment

Dimensions should be multiples of 16 for optimal Tensor Core performance:

```python
# Good: Aligned
M, N, K = 1024, 1024, 1024

# Acceptable: Non-aligned (correct but may be slower)
M, N, K = 100, 100, 100
```

### GPU Architecture Support

| Architecture | SM | Tensor Core |
|--------------|-----|-------------|
| Volta | 7.0 | FP16 |
| Turing | 7.5 | FP16, INT8 |
| Ampere | 8.0, 8.6 | FP16, BF16, INT8, TF32 |
| Ada Lovelace | 8.9 | FP16, BF16, INT8, FP8 |
| Hopper | 9.0 | FP16, BF16, INT8, FP8 |

### Precision Guidelines

- **FP16**: Primary precision for inference, best memory efficiency
- **FP32**: Training or when precision is critical
- **INT8**: Quantized inference (requires Turing+)

### Testing Patterns

```python
# Use assert_close from conftest.py for numerical validation
from .conftest import assert_close

assert_close(output, reference, rtol=1e-3, atol=1e-3)

# Property tests use Hypothesis
from hypothesis import given, settings, strategies as st

@pytest.mark.cuda
@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(batch=st.integers(1, 4), seq_len=st.integers(16, 256))
def test_property(batch, seq_len, device):
    ...
```

---

## Common Tasks

### Adding a New Kernel

1. Create kernel file in `src/`
2. Add header primitives in `include/` if needed
3. Add Python binding in `python/bindings.cpp`
4. Export in `python/__init__.py`
5. Add tests in `tests/`
6. Update documentation

### Debugging CUDA Kernels

```bash
# Profile with Nsight Compute
ncu --set full -o profile_output python -c "
import torch
from cuda_llm_ops import flash_attention
q = torch.randn(1, 8, 1024, 64, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
for _ in range(10): flash_attention(q, k, v)
"

# Check CUDA errors
CUDA_LAUNCH_BLOCKING=1 pytest tests/test_attention.py -v
```

### Memory Debugging

```python
import torch

# Check memory usage
print(torch.cuda.memory_summary())

# Clear cache
torch.cuda.empty_cache()

# Monitor memory
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
```

---

## File Structure

```
llm-speed/
├── src/                    # CUDA kernels
├── include/                # Header primitives
├── python/                 # Python bindings
├── tests/                  # Test suite
├── benchmarks/             # Performance benchmarks
├── docs/                   # Documentation
├── changelog/              # Change history
├── .kiro/specs/            # Design specifications
└── .github/workflows/      # CI/CD workflows
```

---

## Code Style

- **C++/CUDA**: Follow `.clang-format`, use `snake_case` for functions
- **Python**: PEP 8, 4 spaces, max 100 chars, use f-strings
- **Commits**: Conventional Commits (`feat:`, `fix:`, `perf:`, etc.)

---

## Related Documentation

- [API Reference](docs/api.md) - Detailed API documentation
- [Technical Deep Dive](docs/deepwiki.md) - Implementation details
- [Performance Guide](docs/performance.md) - Optimization strategies
- [Contributing](CONTRIBUTING.md) - Development workflow
