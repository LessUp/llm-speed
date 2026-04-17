# AGENTS.md — AI Agent Workflow Instructions

This file provides instructions for AI coding assistants (Claude Code, Cursor, GitHub Copilot, etc.) working on this repository.

---

## Project Philosophy: Spec-Driven Development (SDD)

This project strictly follows the **Spec-Driven Development (SDD)** paradigm. All code implementations must use the specification documents in the `/specs` directory as the **Single Source of Truth (SSOT)**.

**Core Principle**: Specs first, code second. Never write code without a corresponding spec.

---

## Directory Context

| Directory | Purpose |
|-----------|---------|
| `/specs/product/` | Product feature definitions, user stories, and acceptance criteria |
| `/specs/rfc/` | Technical design documents, architecture decisions, and implementation plans |
| `/specs/api/` | API interface definitions (OpenAPI, GraphQL schemas, etc.) |
| `/specs/db/` | Database and schema specifications |
| `/specs/testing/` | BDD test case specifications (Gherkin `.feature` files) |
| `/docs/` | User guides, tutorials, setup guides, and developer documentation |

---

## AI Agent Workflow Instructions

When you (AI) are asked to develop a new feature, modify an existing one, or fix a bug, **you must strictly follow this workflow without skipping any steps**:

### Step 1: Review Specs (Review First)

- **Before writing any code**, read the relevant spec documents in `/specs`:
  - Product specs: `/specs/product/*.md`
  - RFC/Architecture: `/specs/rfc/*.md`
  - API definitions: `/specs/api/*.yaml` or `/specs/api/*.md`
- If the user's request **conflicts with existing specs**:
  - **Stop coding immediately**
  - Point out the conflict clearly
  - Ask the user whether to update the spec first

### Step 2: Spec-First Update

- If this is a **new feature**, or if it requires changes to existing interfaces/database structures:
  - **You must first propose modifications** to the corresponding spec documents
  - Examples: update `openapi.yaml`, create a new RFC, or modify product requirements
- **Wait for user confirmation** of the spec changes before entering the code-writing phase
- Never assume spec changes are approved without explicit user acknowledgment

### Step 3: Code Implementation

- When writing code, **100% comply with the spec definitions**:
  - Variable naming conventions
  - API paths and HTTP methods
  - Data types and validation rules
  - HTTP status codes and error responses
- **Do not add features not defined in the spec** (No Gold-Plating)
- If you need to make a technical decision not covered by the spec, document it and ask the user whether to add it to the spec

### Step 4: Test Against Spec

- Write unit and integration tests based on the **acceptance criteria** in `/specs`
- Ensure test cases cover all **boundary conditions** described in the specs
- For property-based tests, ensure properties align with the correctness criteria defined in `/specs/product/`
- If a test fails, reference the specific spec requirement that is not met

---

## Code Generation Rules

| Rule | Description |
|------|-------------|
| **API Changes** | Any externally exposed API changes must update `/specs/api/` |
| **Architecture Decisions** | Consult `/specs/rfc/` for conventions; do not invent design patterns |
| **No Spec, No Code** | Never write code without a corresponding spec or spec update proposal |
| **No Gold-Plating** | Do not implement features beyond what the spec defines |
| **Traceability** | Reference spec requirements in commit messages (e.g., `feat: implement REQ-3 FlashAttention`) |

---

## Why This Matters

1. **Prevent AI Hallucination**: Forcing the AI to read `/specs` first anchors its thinking to the project's actual requirements and constraints.

2. **Document-Code Synchronization**: "Modify spec first, then code" ensures documentation and code are always in sync.

3. **PR Quality**: When the AI generates Pull Requests, the implementation will be highly aligned with business logic because it was developed against the acceptance criteria you defined.

---

## Quick Reference: Common Commands

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

## Current Specifications

| Spec | Location | Description |
|------|----------|-------------|
| Product Requirements | [`/specs/product/cuda-llm-kernel-optimization.md`](specs/product/cuda-llm-kernel-optimization.md) | Feature requirements and acceptance criteria |
| Core Architecture RFC | [`/specs/rfc/0001-core-architecture.md`](specs/rfc/0001-core-architecture.md) | Technical design and architecture |
| Implementation Tasks RFC | [`/specs/rfc/0002-implementation-tasks.md`](specs/rfc/0002-implementation-tasks.md) | Implementation plan and task breakdown |

---

## Project Overview

This is a **CUDA kernel optimization library for LLM inference**, providing:

- **FlashAttention**: O(N) memory complexity with online softmax algorithm
- **Tensor Core GEMM**: Hardware-accelerated matrix multiplication (FP16/INT8)
- **High-Performance GEMM**: Register tiling and double buffering
- **PyTorch Integration**: Python bindings via pybind11

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

## Testing

### Test Categories

| Marker | Purpose | Command |
|--------|---------|---------|
| `cuda` | Requires GPU | `pytest -m cuda` |
| `property` | Hypothesis tests | `pytest -m property` |
| `slow` | Long-running | `pytest -m "not slow"` |

### Test Structure

```python
import pytest
from hypothesis import given, settings, strategies as st

class TestFlashAttention:
    @pytest.mark.cuda
    def test_correctness(self, device):
        """Verify output matches PyTorch reference."""
        q = torch.randn(2, 4, 64, 32, device=device)
        output = flash_attention(q, k, v)
        reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        assert_close(output, reference)

    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(batch=st.integers(1, 4), seq_len=st.integers(16, 256))
    def test_property(self, batch, seq_len, device):
        """Property-based testing with Hypothesis."""
        pass
```

---

## Code Style

- **C++/CUDA**: Follow `.clang-format`, use `snake_case` for functions
- **Python**: PEP 8, 4 spaces, max 100 chars, use f-strings
- **Commits**: Conventional Commits (`feat:`, `fix:`, `perf:`, etc.)

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/api/api-en.md) | Detailed API documentation |
| [Architecture](docs/architecture/architecture-en.md) | Technical deep dive |
| [Performance Guide](docs/tutorials/performance-en.md) | Optimization strategies |
| [Contributing](CONTRIBUTING.md) | Development workflow |
