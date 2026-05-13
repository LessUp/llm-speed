---
title: Quick Start Guide
description: Get up and running with LLM-Speed in 5 minutes
---

# Quick Start Guide

Get up and running with LLM-Speed in 5 minutes.

---

## Table of Contents

- [Installation](#installation)
- [First Steps](#first-steps)
- [FlashAttention Quickstart](#flashattention-quickstart)
- [GEMM Quickstart](#gemm-quickstart)
- [Next Steps](#next-steps)

---

## Installation

### Prerequisites

- CUDA 11.0 or higher
- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU (SM 7.0+, Volta or newer)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/LessUp/llm-speed.git
cd llm-speed

# Create an isolated environment
python3 -m venv .venv
. .venv/bin/activate

# Install dependencies and local validation tools
pip install -U pip setuptools wheel
pip install -r requirements.txt pytest hypothesis ruff pre-commit

# Build and install CUDA extension
pip install -e .
```

### Verify Installation

```bash
python -c "import cuda_llm_ops; print(f'Version: {cuda_llm_ops.__version__}')"
```

Expected output:
```
Version: 0.3.0
```

---

## First Steps

### Check GPU Compatibility

```python
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

# Check GPU compute capability (need SM 7.0+)
capability = torch.cuda.get_device_capability()
print(f"Compute capability: {capability[0]}.{capability[1]}")

if capability[0] >= 7:
    print("✓ GPU supports LLM-Speed")
else:
    print("✗ GPU too old (need Volta or newer)")
```

---

## FlashAttention Quickstart

### Basic Usage

```python
import torch
from cuda_llm_ops import flash_attention

# Create sample inputs
batch = 2          # Number of sequences
heads = 8          # Number of attention heads
seq_len = 512      # Sequence length
head_dim = 64      # Dimension per head

# Create tensors on GPU with FP16 precision
q = torch.randn(batch, heads, seq_len, head_dim,
                device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

# Compute attention
output = flash_attention(q, k, v)

print(f"Input shape:  {q.shape}")
print(f"Output shape: {output.shape}")
# Input shape:  torch.Size([2, 8, 512, 64])
# Output shape: torch.Size([2, 8, 512, 64])
```

### Causal Attention (for GPT-like models)

```python
# Use causal mask for autoregressive generation
def generate_next_token(q, k, v_cache):
    """Generate next token with causal attention."""
    # Causal mask ensures each position only attends to previous positions
    output = flash_attention(q, k, v_cache, is_causal=True)
    return output

# Example
output_causal = flash_attention(q, k, v, is_causal=True)
```

### Memory Efficiency Demo

```python
import gc

def measure_memory(func, *args):
    """Measure peak memory usage of a function."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    result = func(*args)
    torch.cuda.synchronize()

    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    return result, peak_memory

# Test with long sequence
long_q = torch.randn(2, 8, 4096, 64, device='cuda', dtype=torch.float16)
long_k = torch.randn_like(long_q)
long_v = torch.randn_like(long_q)

# FlashAttention
_, mem_flash = measure_memory(flash_attention, long_q, long_k, long_v)
print(f"FlashAttention memory: {mem_flash:.1f} MB")
# ~32 MB

# Compare: Standard attention would use ~256 MB for 4K sequence!
print(f"Memory saved: ~{256 - mem_flash:.1f} MB")
```

---

## GEMM Quickstart

### Standard GEMM

```python
import torch
from cuda_llm_ops import gemm

# Matrix dimensions
M, K, N = 1024, 512, 1024

# Create matrices on GPU
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

# C = A @ B
c = gemm(a, b)
print(f"Result shape: {c.shape}")  # torch.Size([1024, 1024])

# C = 2.0 * A @ B (scaled)
c_scaled = gemm(a, b, alpha=2.0)
```

### Tensor Core GEMM

```python
from cuda_llm_ops import tensor_core_gemm

# FP16 input, FP32 output (higher precision accumulation)
a = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
b = torch.randn(512, 1024, device='cuda', dtype=torch.float16)

c = tensor_core_gemm(a, b)
print(f"Input dtype:  {a.dtype}")      # torch.float16
print(f"Output dtype: {c.dtype}")     # torch.float32
```

### Transposed Operations

```python
# Handle transposed matrices without explicit transpose
a_t = torch.randn(K, M, device='cuda', dtype=torch.float16)  # Actually A^T
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

# C = A^T @ B (equivalent to a.t() @ b, but more efficient)
c = gemm(a_t, b, trans_a=True)
print(f"Result shape: {c.shape}")  # torch.Size([M, N])

# All combinations:
# gemm(a, b, trans_a=False, trans_b=False)  # A @ B
# gemm(a, b, trans_a=False, trans_b=True)   # A @ B^T
# gemm(a, b, trans_a=True,  trans_b=False)  # A^T @ B
# gemm(a, b, trans_a=True,  trans_b=True)   # A^T @ B^T
```

---

## Next Steps

### Explore Examples

```bash
# Run benchmarks
python benchmarks/benchmark_attention.py
python benchmarks/benchmark_gemm.py

# Run tests
pytest tests/ -v
```

### Read the Documentation

- **[API Reference](../api/api-en.md)** - Complete API documentation
- **[Architecture Guide](../architecture/architecture-en.md)** - Implementation details
- **[Performance Guide](../tutorials/performance-en.md)** - Optimization tips
- **[Troubleshooting](../tutorials/troubleshooting-en.md)** - Common issues

### Try Different Configurations

```python
# Experiment with different sequence lengths
for seq_len in [256, 512, 1024, 2048, 4096]:
    q = torch.randn(2, 8, seq_len, 64, device='cuda', dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Time it
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    _ = flash_attention(q, k, v)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    print(f"Seq len {seq_len:4d}: {elapsed_ms:.3f} ms")
```

### Join the Community

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions and share ideas

---

## Quick Reference Card

```python
# ========== Attention ==========
from cuda_llm_ops import flash_attention, tiled_attention, naive_attention

# Standard attention
out = flash_attention(q, k, v)

# With causal mask (GPT-style)
out = flash_attention(q, k, v, is_causal=True)

# ========== GEMM ==========
from cuda_llm_ops import gemm, tensor_core_gemm, tensor_core_gemm_int8

# Standard GEMM: C = alpha * A @ B + beta * C
c = gemm(a, b, alpha=1.0, beta=0.0)

# Tensor Core (FP16 in, FP32 out)
c = tensor_core_gemm(a_fp16, b_fp16)

# INT8 quantization (Turing+ only)
c = tensor_core_gemm_int8(a_int8, b_int8)  # Returns INT32

# Transposed
output = gemm(a_t, b, trans_a=True)  # A^T @ B
```

[← Back to Documentation](../)
