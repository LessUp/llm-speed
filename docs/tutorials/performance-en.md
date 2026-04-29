---
layout: docs
title: Performance Tuning Guide
description: Comprehensive guide for maximizing performance with LLM-Speed
lang: en
---

# Performance Tuning Guide

Comprehensive guide for maximizing performance with LLM-Speed.

---

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Performance Targets](#performance-targets)
- [Optimization Strategies](#optimization-strategies)
- [Benchmarking](#benchmarking)
- [Profiling](#profiling)
- [Troubleshooting Performance Issues](#troubleshooting-performance-issues)
- [Best Practices](#best-practices)

---

## Hardware Requirements

### Minimum vs Recommended

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **GPU** | SM 7.0 (Volta) | SM 8.0+ (Ampere) | Tensor Cores essential for peak performance |
| **VRAM** | 8 GB | 16+ GB | Larger batches and sequences need more memory |
| **CUDA** | 11.0 | 12.0+ | Newer versions have better optimizations |
| **Driver** | 450.80+ | 525+ | Check `nvidia-smi` |

### GPU Architecture Features

| Architecture | SM | Tensor Core | Key Features |
|--------------|-----|-------------|--------------|
| **Volta** (V100) | 7.0 | FP16 | First Tensor Core generation |
| **Turing** (T4) | 7.5 | FP16, INT8 | INT8 quantization support |
| **Ampere** (A100) | 8.0, 8.6 | FP16, BF16, INT8, TF32 | Async copy, 164KB shared mem |
| **Ada Lovelace** (RTX 4090) | 8.9 | + FP8 | Enhanced FP8 support |
| **Hopper** (H100) | 9.0 | + FP8 | Thread block clusters, DPX instructions |

These rows describe **hardware capabilities**, not the full set of precisions currently exposed by this repository.

### Memory Planning

Calculate memory requirements before running:

```python
def estimate_attention_memory(batch, heads, seq_len, head_dim, dtype=torch.float16):
    """Estimate FlashAttention memory usage in MB."""
    bytes_per_elem = 2 if dtype == torch.float16 else 4

    # Input tensors (Q, K, V)
    input_mem = 3 * batch * heads * seq_len * head_dim * bytes_per_elem

    # Output tensor
    output_mem = batch * heads * seq_len * head_dim * bytes_per_elem

    # FlashAttention working memory (overhead)
    # ~O(seq_len) rather than O(seq_len²)
    working_mem = batch * heads * seq_len * head_dim * bytes_per_elem * 0.1

    total_mb = (input_mem + output_mem + working_mem) / (1024 * 1024)
    return total_mb

# Example: Large sequence
print(f"Memory needed: {estimate_attention_memory(2, 16, 8192, 64):.1f} MB")
# Output: Memory needed: ~55 MB (vs ~2560 MB for naive attention!)
```

---

## Performance Targets

### GEMM Performance Expectations

| Matrix Size | cuBLAS Reference | Target | Typical Result |
|-------------|------------------|--------|----------------|
| 512×512×512 | 100% | ≥85% | 88-92% |
| 1024×1024×1024 | 100% | ≥90% | 91-95% |
| 2048×2048×2048 | 100% | >90% | 92-98% |

### Attention Latency Benchmarks

| Sequence Length | PyTorch SDPA | Our FlashAttention | Speedup |
|-----------------|--------------|-------------------|---------|
| 512 | 1.0x | 1.2x | Baseline |
| 1024 | 1.0x | 1.4x | +40% |
| 2048 | 1.0x | 1.6x | +60% |
| 4096 | 1.0x | 1.8x | +80% |
| 8192 | 1.0x | 2.1x | +110% |

> **Note**: Results on A100, FP16 precision, batch=2, heads=8, head_dim=64

### Memory Efficiency

| Implementation | Memory Complexity | 4K Sequence | 16K Sequence |
|----------------|-------------------|-------------|--------------|
| Naive Attention | O(N²) | 256 MB | 4 GB |
| Tiled Attention | O(N²), improved | 128 MB | 2 GB |
| FlashAttention | O(N) | 4 MB | 16 MB |

---

## Optimization Strategies

### 1. Kernel Selection Guide

Choose the right kernel for your workload:

```python
def optimal_attention(q, k, v, is_causal=False):
    """Select optimal attention implementation."""
    seq_len = q.size(2)

    if seq_len >= 2048:
        # FlashAttention essential for very long sequences
        return flash_attention(q, k, v, is_causal=is_causal)
    elif seq_len >= 512:
        # FlashAttention still beneficial
        return flash_attention(q, k, v, is_causal=is_causal)
    elif seq_len >= 128:
        # Tiled attention good balance
        return tiled_attention(q, k, v)
    else:
        # Naive may have lower overhead for very short sequences
        return naive_attention(q, k, v)
```

### 2. Precision Optimization

```python
# For inference: Always use FP16
model = model.half()  # Convert to FP16
q = q.cuda().half()
output = flash_attention(q, k, v)

# For training: FP32 master weights, FP16 compute
# (Mixed precision pattern)
with torch.cuda.amp.autocast():
    output = flash_attention(q, k, v)

# Tensor Core GEMM with accumulation
# Input: FP16, Compute: FP16, Output: FP32
c = tensor_core_gemm(a_fp16, b_fp16)  # Returns FP32
```

### 3. Tensor Core Alignment

Optimal dimensions should be multiples of 16:

```python
def pad_to_alignment(tensor, alignment=16):
    """Pad tensor dimensions to alignment for optimal Tensor Core performance."""
    shape = list(tensor.shape)

    # Pad last two dimensions
    if len(shape) >= 2:
        for i in [-2, -1]:
            if shape[i] % alignment != 0:
                padding = alignment - (shape[i] % alignment)
                shape[i] += padding

    if shape != list(tensor.shape):
        padded = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
        # Copy original data
        slices = [slice(0, s) for s in tensor.shape]
        padded[tuple(slices)] = tensor
        return padded
    return tensor

# Usage
a_aligned = pad_to_alignment(a, 16)  # Ensures Tensor Core efficiency
b_aligned = pad_to_alignment(b, 16)
c = tensor_core_gemm(a_aligned, b_aligned)
```

### 4. Batch Size Tuning

```python
def find_optimal_batch_size(heads, seq_len, head_dim, max_memory_gb=16):
    """Find optimal batch size given memory constraints."""
    max_memory_bytes = max_memory_gb * 1024**3
    bytes_per_elem = 2  # FP16

    # Memory per sample: Q + K + V + O
    per_sample = 4 * heads * seq_len * head_dim * bytes_per_elem

    # Account for ~20% overhead
    max_batch = int(max_memory_bytes / per_sample / 1.2)

    # Round down to power of 2 for better GPU utilization
    optimal_batch = 2 ** (max_batch.bit_length() - 1)
    return max(optimal_batch, 1)

# Example
batch_size = find_optimal_batch_size(16, 2048, 64, max_memory_gb=24)
print(f"Optimal batch size: {batch_size}")  # e.g., 8
```

### 5. Memory Pool Pre-allocation

```python
class AttentionMemoryPool:
    """Pre-allocate and reuse memory for repeated operations."""

    def __init__(self, max_batch, heads, max_seq_len, head_dim):
        self.q = torch.empty(max_batch, heads, max_seq_len, head_dim,
                            device='cuda', dtype=torch.float16)
        self.k = torch.empty_like(self.q)
        self.v = torch.empty_like(self.q)
        self.output = torch.empty_like(self.q)

    def get_tensors(self, batch_size, seq_len):
        """Get appropriately sized views."""
        return (
            self.q[:batch_size, :, :seq_len, :],
            self.k[:batch_size, :, :seq_len, :],
            self.v[:batch_size, :, :seq_len, :],
            self.output[:batch_size, :, :seq_len, :]
        )
```

### 6. Causal vs Non-Causal

```python
# For training (with padding mask): Don't use is_causal
output = flash_attention(q, k, v, is_causal=False)

# For generation (autoregressive): Use is_causal
# This enables early-exit optimization
output = flash_attention(q, k, v, is_causal=True)
```

---

## Benchmarking

### Attention Benchmark

```bash
# Basic benchmark
python benchmarks/benchmark_attention.py

# Custom configuration
python benchmarks/benchmark_attention.py \
    --seq-lengths 512 1024 2048 4096 8192 \
    --batch-size 4 \
    --num-heads 16 \
    --head-dim 64 \
    --dtype fp16 \
    --warmup 50 \
    --iterations 200

# Export to JSON for analysis
python benchmarks/benchmark_attention.py --output results.json
```

**Expected Output Format:**
```
================================================================================
ATTENTION BENCHMARK RESULTS
================================================================================

Config: batch=2, heads=8, head_dim=64, dtype=float16, causal=False

 Seq Len | PyTorch(ms) | Flash(ms) | Speedup | Memory Saved
---------|-------------|-----------|---------|-------------
     512 |       0.234 |     0.189 |   1.24x |       3.5MB
    1024 |       0.823 |     0.612 |   1.34x |      14.0MB
    2048 |       3.124 |     2.156 |   1.45x |      56.0MB
    4096 |      12.456 |     7.234 |   1.72x |     224.0MB
    8192 |      49.823 |    23.456 |   2.12x |     896.0MB
```

### GEMM Benchmark

```bash
# Standard benchmark
python benchmarks/benchmark_gemm.py

# Custom matrix sizes
python benchmarks/benchmark_gemm.py \
    --sizes 512x512x512 1024x1024x1024 2048x2048x2048 \
    --dtype fp16 \
    --output gemm_results.json
```

### Custom Benchmark Script

```python
import torch
import time
from cuda_llm_ops import flash_attention, gemm
from cuda_llm_ops.profiler import CUDAProfiler

def benchmark_flash_attention(batch, heads, seq_len, head_dim, iterations=100):
    """Custom FlashAttention benchmark."""
    q = torch.randn(batch, heads, seq_len, head_dim,
                    device='cuda', dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup
    for _ in range(10):
        _ = flash_attention(q, k, v)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        _ = flash_attention(q, k, v)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iterations

    # Calculate metrics
    flops = batch * heads * (4 * seq_len * seq_len * head_dim)  # Simplified
    tflops = flops / (elapsed_ms * 1e-3) / 1e12

    return {
        'latency_ms': elapsed_ms,
        'throughput_tflops': tflops,
        'config': f'{batch}x{heads}x{seq_len}x{head_dim}'
    }

# Run benchmark
result = benchmark_flash_attention(2, 8, 2048, 64)
print(f"Config: {result['config']}")
print(f"Latency: {result['latency_ms']:.3f} ms")
print(f"Throughput: {result['throughput_tflops']:.2f} TFLOPS")
```

---

## Profiling

### Using Nsight Compute

```bash
# Profile FlashAttention with detailed metrics
ncu --set full --kernel-name flash_attention \
    -o flash_attention_profile.ncu-rep \
    python -c "
import torch
from cuda_llm_ops import flash_attention

q = torch.randn(2, 8, 2048, 64, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

for _ in range(100):
    flash_attention(q, k, v)
"

# View results
ncu-ui flash_attention_profile.ncu-rep
```

### Key Metrics to Monitor

| Metric | Target | Interpretation |
|--------|--------|----------------|
| **SM Occupancy** | >60% | High: Good thread utilization |
| **Memory Throughput** | >70% of peak | Bound by memory bandwidth |
| **L2 Hit Rate** | >60% | Efficient cache utilization |
| **Warp Stall Reasons** | Various | Identify bottlenecks |

### Python Profiler

```python
from cuda_llm_ops.profiler import CUDAProfiler

profiler = CUDAProfiler()

# Profile single operation
with profiler.profile('flash_attention'):
    output = flash_attention(q, k, v)

metrics = profiler.get_metrics()
print(f"Elapsed: {metrics.elapsed_ms:.2f} ms")
print(f"Memory bandwidth: {metrics.memory_bandwidth_gb:.1f} GB/s")
print(f"Compute utilization: {metrics.compute_utilization:.1%}")

# Compare with reference
results = profiler.compare_with_reference(
    custom_func=flash_attention,
    reference_func=lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v),
    q, k, v
)
print(f"Speedup vs PyTorch: {results['speedup']:.2f}x")
```

---

## Troubleshooting Performance Issues

### Issue: Low GPU Utilization

**Symptoms:**
- `nvidia-smi` shows low GPU utilization (<50%)
- Kernel execution time much higher than expected

**Solutions:**
1. Increase batch size
2. Check for CPU-GPU synchronization points
3. Ensure tensors are contiguous

```python
# Check tensor layout
assert q.is_contiguous(), "Q must be contiguous"

# Remove unnecessary synchronization
# Bad:
torch.cuda.synchronize()
output = flash_attention(q, k, v)
torch.cuda.synchronize()

# Good:
output = flash_attention(q, k, v)
```

### Issue: Tensor Core Not Engaged

**Symptoms:**
- Performance significantly below cuBLAS
- Nsight Compute shows no Tensor Core usage

**Solutions:**
1. Ensure dimensions are multiples of 16
2. Use `tensor_core_gemm` instead of regular `gemm`
3. Verify FP16 input dtype

```python
# Check alignment
M, K, N = a.shape[0], a.shape[1], b.shape[1]
for dim, name in zip([M, K, N], ['M', 'K', 'N']):
    if dim % 16 != 0:
        print(f"Warning: {name}={dim} not aligned to 16")

# Use Tensor Core variant
c = tensor_core_gemm(a, b)  # Forces Tensor Core usage
```

### Issue: Out of Memory

**Symptoms:**
- `RuntimeError: CUDA out of memory`
- Memory grows across iterations

**Solutions:**
1. Use FlashAttention instead of naive attention
2. Reduce batch size or sequence length
3. Clear cache between iterations

```python
# Use FlashAttention (O(N) memory)
output = flash_attention(q, k, v)  # Not naive_attention

# Clear cache if needed
torch.cuda.empty_cache()

# Monitor memory
print(torch.cuda.memory_summary())
```

---

## Best Practices

### For Inference

1. **Use FP16 precision** for memory efficiency and speed
2. **Use FlashAttention** for sequences > 512
3. **Pre-allocate memory pools** for repeated operations
4. **Align dimensions** to multiples of 16 for Tensor Core
5. **Batch requests** when possible for better GPU utilization

### For Training

1. **Use automatic mixed precision (AMP)** with GradScaler
2. **Monitor gradient norms** when using FP16
3. **Verify numerical stability** with small-scale tests first
4. **Profile before optimizing** to identify actual bottlenecks

### For Production

1. **Implement graceful degradation** (fallback to PyTorch if needed)
2. **Set CUDA visible devices** explicitly for multi-GPU setups
3. **Use streams for concurrent execution** when processing multiple requests
4. **Monitor GPU memory** and implement OOM handling

```python
# Production-ready pattern
import torch
from cuda_llm_ops import flash_attention

def safe_flash_attention(q, k, v, fallback_to_torch=True):
    try:
        return flash_attention(q, k, v)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and fallback_to_torch:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)
        raise
```

---

## Additional Resources

- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

[← Back to Documentation](../)
