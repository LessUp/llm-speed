# QWEN.md — LLM-Speed Project Context

## Project Overview

**LLM-Speed** is a high-performance CUDA kernel library for LLM inference, providing optimized attention mechanisms and GEMM (General Matrix Multiply) operations. The project achieves **98% memory reduction** compared to standard attention implementations and delivers **2.1× speedup** at 8K sequence lengths on NVIDIA GPUs.

### Core Features

| Feature | Description | Performance |
|---------|-------------|-------------|
| **FlashAttention** | O(N) memory attention with online softmax | 98% memory reduction |
| **Tensor Core GEMM** | Hardware-accelerated matrix multiplication via WMMA API | 95%+ cuBLAS performance |
| **High-Performance GEMM** | Register tiling with 3-level blocking | 90%+ cuBLAS performance |
| **INT8 GEMM** | Quantized matrix operations (Turing+ required) | 90%+ cuBLAS performance |

### Key Technologies

- **CUDA** 11.0+ with C++17
- **PyTorch** 2.0+ integration via pybind11
- **Python** 3.8+ bindings
- **GPU Support**: Volta (SM 7.0) through Hopper (SM 9.0)

### Architecture

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

---

## Project Structure

```
llm-speed/
├── cuda_llm_ops/            # Python package
│   ├── __init__.py          # Package entry point with fallback
│   ├── _cuda_llm_ops.pyi    # Type stubs for IDE autocomplete
│   ├── bindings.cpp         # pybind11 bindings to CUDA kernels
│   └── profiler.py          # Performance profiling utilities
│
├── src/                     # CUDA kernel implementations
│   ├── naive_attention.cu   # Baseline O(N²) attention
│   ├── tiled_attention.cu   # Shared memory tiling optimization
│   ├── flash_attention.cu   # O(N) FlashAttention implementation
│   ├── tensor_core_gemm.cu  # WMMA Tensor Core acceleration
│   └── hgemm_kernel.cu      # High-performance GEMM kernel
│
├── include/                 # CUDA header primitives
│   ├── common.cuh           # Core types (AttentionConfig, GemmConfig)
│   ├── online_softmax.cuh   # Online softmax algorithm
│   ├── warp_primitives.cuh  # Warp-level operations (reduce, broadcast)
│   ├── shared_memory.cuh    # Shared memory utilities with padding
│   └── pipeline.cuh         # Double buffering, async copy
│
├── tests/                   # Test suite (pytest + hypothesis)
│   ├── conftest.py          # Fixtures, helpers, reference implementations
│   ├── test_attention.py    # Attention kernel correctness tests
│   ├── test_gemm.py         # GEMM kernel tests
│   ├── test_interface.py    # Python binding integration tests
│   └── test_profiler.py     # Profiler unit tests
│
├── benchmarks/              # Performance benchmarking scripts
│   ├── benchmark_attention.py
│   └── benchmark_gemm.py
│
├── docs/                    # User documentation (English & Chinese)
│   ├── setup/               # Installation and quickstart guides
│   ├── api/                 # API reference documentation
│   ├── architecture/        # Technical architecture deep dives
│   ├── tutorials/           # Performance guides, troubleshooting
│   └── changelog/           # Version history (EN/ZH)
│
├── specs/                   # Specifications (Spec-Driven Development)
│   ├── product/             # Product requirements, user stories
│   ├── rfc/                 # Technical design documents
│   ├── api/                 # API interface specifications
│   └── testing/             # BDD test specifications
│
├── _layouts/                # Jekyll layouts for GitHub Pages
├── assets/                  # GitHub Pages assets (CSS, JS, images)
└── .github/workflows/       # CI/CD pipelines
```

---

## Building and Running

### Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| CUDA Toolkit | 11.0+ | Required for GPU compilation |
| Python | 3.8+ | 3.10+ recommended |
| PyTorch | 2.0+ | Must have CUDA support |
| NVIDIA GPU | SM 7.0+ | Volta or newer |
| CMake | 3.18+ | Optional (alternative build method) |

### Installation

```bash
# 1. Clone repository
git clone https://github.com/LessUp/llm-speed.git
cd llm-speed

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build and install CUDA extension (editable mode)
pip install -e .
```

**For development with all extras:**
```bash
pip install -e ".[dev]"
```

### Build System

The project supports **two build methods**:

#### Method 1: setuptools (Primary)

```bash
pip install -e .
```

- Reads version from `pyproject.toml`
- Compiles CUDA sources via `torch.utils.cpp_extension`
- Targets: SM 7.0, 7.5, 8.0, 8.6, 8.9, 9.0
- Installs as `cuda_llm_ops` Python package

#### Method 2: CMake (Alternative)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

- Uses FetchContent for pybind11 (v2.11.1)
- Builds both shared library and Python module
- Same CUDA architectures

### Running

```python
import torch
from cuda_llm_ops import flash_attention, tensor_core_gemm

# FlashAttention - O(N) memory
q = torch.randn(2, 8, 2048, 64, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

output = flash_attention(q, k, v, is_causal=True)

# Tensor Core GEMM
a = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
b = torch.randn(512, 1024, device='cuda', dtype=torch.float16)
c = tensor_core_gemm(a, b)  # FP16 input, FP32 output
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Test categories
pytest tests/ -v -m cuda         # GPU-requiring tests
pytest tests/ -v -m property     # Property-based tests (Hypothesis)
pytest tests/ -v -m "not cuda"   # CPU-safe tests

# Specific test files
pytest tests/test_attention.py -v
pytest tests/test_gemm.py -v
```

### Benchmarking

```bash
# Attention benchmark
python benchmarks/benchmark_attention.py --seq-lengths 512 1024 2048 4096

# GEMM benchmark
python benchmarks/benchmark_gemm.py --sizes 1024x1024x1024

# Export results
python benchmarks/benchmark_attention.py --output results.json
```

---

## Development Conventions

### Spec-Driven Development (SDD)

This project follows **Spec-Driven Development** paradigm. **Specs first, code second.**

**Workflow:**
1. Review specs in `/specs/` before writing code
2. Update specs first when adding features or changing interfaces
3. Implement code that 100% complies with specs
4. Test against acceptance criteria in specs

**Spec directories:**
- `/specs/product/` - Product requirements, user stories
- `/specs/rfc/` - Technical design documents, architecture decisions
- `/specs/api/` - API interface specifications
- `/specs/testing/` - BDD test specifications

### Code Style

**Python:**
- PEP 8, 4 spaces, max 100 chars
- Use f-strings for formatting
- Ruff for linting and formatting (v0.15.10)
- Type hints encouraged (see `_cuda_llm_ops.pyi`)

**C++/CUDA:**
- C++17 standard enforced
- `snake_case` for function names
- clang-format for code style
- Use `--use_fast_math` for CUDA optimizations

**Commit Messages:**
- Conventional Commits format: `type: description`
- Types: `feat:`, `fix:`, `perf:`, `docs:`, `refactor:`, `test:`, `chore:`
- Reference specs in commits: `feat: implement REQ-3 FlashAttention`

### Pre-commit Hooks

The project uses pre-commit for automated quality checks:

```bash
# Install hooks
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

**Checks include:**
- ✅ Ruff (linting + formatting) for Python files
- ✅ clang-format for C++/CUDA files
- ✅ Trailing whitespace removal
- ✅ YAML/JSON validation
- ✅ Merge conflict detection
- ✅ Large file prevention (>1MB)

### Ruff Configuration

**Ignored rules (acceptable exceptions):**
- `E501` - Line length (handled by formatter)
- `N803`, `N806` - Uppercase M, N, K (standard GEMM conventions)
- `B006` - Mutable defaults (acceptable in benchmarks)
- `B017` - Blind exceptions (acceptable in tests)
- `B028` - No stacklevel (acceptable in warnings)

### Testing Practices

- **pytest markers**: `cuda`, `property`, `slow`
- **Hypothesis** for property-based testing
- **Reference implementations** in `conftest.py` for correctness verification
- **CPU-safe tests** for CI environments without GPUs
- **Test categories**: correctness, properties, edge cases, performance

---

## CI/CD Pipelines

### CI Workflow (`.github/workflows/ci.yml`)

**Triggers:** Push/PR to main/master, manual dispatch

**Jobs:**
1. **Lint** - Ruff check + format verification
2. **CPU Tests** - Python syntax validation + CPU-safe tests
3. **Docs** - YAML validation + Markdown link check

### Pages Workflow (`.github/workflows/pages.yml`)

**Triggers:** Markdown/docs changes, manual dispatch

**Jobs:**
1. **Build Search Index** - Ruby script generates JSON search index
2. **Build Jekyll Site** - Full documentation site generation
3. **Deploy** - Deploy to GitHub Pages

**Site URL:** https://lessup.github.io/llm-speed/

---

## API Reference

### Attention Functions

```python
from cuda_llm_ops import naive_attention, tiled_attention, flash_attention
```

All functions share signature:
```python
output = attention_fn(q, k, v, scale=0.0, is_causal=False)
```

**Parameters:**
- `q, k, v`: Tensors of shape `[batch, heads, seq_len, head_dim]`
- `scale`: Custom scale factor (0.0 = auto `1/√head_dim`)
- `is_causal`: Apply causal mask for autoregressive generation

**Returns:** Output tensor of same shape

### GEMM Functions

```python
from cuda_llm_ops import gemm, tensor_core_gemm, tensor_core_gemm_int8
```

| Function | Input | Output | Notes |
|----------|-------|--------|-------|
| `gemm` | FP32/FP16 | FP32 | Standard GEMM with transpose support |
| `tensor_core_gemm` | FP16 | FP32 | WMMA API, FP32 accumulation |
| `tensor_core_gemm_int8` | INT8 | INT32 | Requires Turing+ (SM≥7.2) |

**Signature:**
```python
c = gemm(a, b, alpha=1.0, beta=0.0, trans_a=False, trans_b=False)
```

---

## Performance Targets

| Metric | Target | Current Status |
|--------|--------|----------------|
| Memory (FlashAttention) | O(N) | ✅ Achieved (98% reduction) |
| Speed (FlashAttention) | 2× PyTorch SDPA | ✅ 2.1× at 8K |
| GEMM Performance | 90%+ cuBLAS | ✅ 90-95% achieved |
| GPU Support | Volta → Hopper | ✅ SM 7.0-9.0 |

---

## Common Development Tasks

### Add a New CUDA Kernel

1. Create spec in `/specs/product/` or update RFC
2. Implement kernel in `src/new_kernel.cu`
3. Add header declarations in `include/`
4. Create pybind11 bindings in `cuda_llm_ops/bindings.cpp`
5. Write tests in `tests/test_new_kernel.py`
6. Update `setup.py` and `CMakeLists.txt` source lists
7. Run tests: `pytest tests/test_new_kernel.py -v`

### Update Python API

1. Update API spec in `/specs/api/`
2. Modify bindings in `cuda_llm_ops/bindings.cpp`
3. Update type stubs in `cuda_llm_ops/_cuda_llm_ops.pyi`
4. Update `cuda_llm_ops/__init__.py` exports
5. Update documentation in `docs/api/`

### Run Code Quality Checks

```bash
# Python linting and formatting
ruff check cuda_llm_ops/ tests/ benchmarks/
ruff format --check cuda_llm_ops/ tests/ benchmarks/

# C++/CUDA formatting
clang-format -i src/*.cu include/*.cuh cuda_llm_ops/bindings.cpp

# Pre-commit (all checks)
pre-commit run --all-files
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `pyproject.toml` | Python project metadata, dependencies, tool configs |
| `setup.py` | setuptools build script with CUDA extension |
| `CMakeLists.txt` | Alternative CMake build system |
| `cuda_llm_ops/__init__.py` | Python package entry with fallback |
| `cuda_llm_ops/bindings.cpp` | pybind11 bindings exposing all kernels |
| `include/common.cuh` | Core types and CUDA_CHECK macro |
| `tests/conftest.py` | Pytest fixtures, reference implementations |
| `.pre-commit-config.yaml` | Pre-commit hooks configuration |
| `_config.yml` | Jekyll configuration for GitHub Pages |
| `AGENTS.md` | AI agent workflow instructions |

---

## Troubleshooting

### CUDA Compilation Errors

```bash
# Verify CUDA installation
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"

# If using conda:
conda install -c conda-forge cudatoolkit=11.8
```

### Module Not Found

```bash
# Rebuild extension
pip install -e . --verbose

# Check build logs for errors
```

### Import Warnings

If you see "CUDA kernels not built" warning:
```bash
pip install -e .
```

---

## Related Resources

- **Documentation:** https://lessup.github.io/llm-speed/
- **GitHub Releases:** https://github.com/LessUp/llm-speed/releases
- **Issues:** https://github.com/LessUp/llm-speed/issues
- **Discussions:** https://github.com/LessUp/llm-speed/discussions

---

## References

1. **FlashAttention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
2. **FlashAttention-2**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023
3. **CUTLASS**: NVIDIA CUTLASS - CUDA Templates for Linear Algebra Subroutines
