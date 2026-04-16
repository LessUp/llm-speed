# Performance Tuning Guide

This guide covers hardware requirements, performance optimization strategies, and benchmarking tools for the CUDA LLM Kernel Optimization library.

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Performance Targets](#performance-targets)
- [Optimization Strategies](#optimization-strategies)
- [Benchmarking](#benchmarking)
- [Profiling](#profiling)
- [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | SM 7.0 (Volta) | SM 8.0+ (Ampere) |
| VRAM | 8 GB | 16+ GB |
| CUDA | 11.0 | 12.0+ |
| Driver | 450.80+ | 525+ |

### GPU Architecture Features

| Architecture | SM | Tensor Core | Key Features |
|--------------|-----|-------------|--------------|
| **Volta** | 7.0 | FP16 | First Tensor Core support |
| **Turing** | 7.5 | FP16, INT8 | INT8 Tensor Core |
| **Ampere** | 8.0, 8.6 | FP16, BF16, INT8, TF32 | Async copy, shared memory 164KB |
| **Ada Lovelace** | 8.9 | FP16, BF16, INT8, FP8 | FP8 support |
| **Hopper** | 9.0 | FP16, BF16, INT8, FP8 | Thread block clusters |

### Memory Requirements by Sequence Length

| Sequence Length | Batch=1, Heads=8 | Batch=4, Heads=16 |
|-----------------|------------------|-------------------|
| 512 | ~50 MB | ~400 MB |
| 1024 | ~200 MB | ~1.6 GB |
| 2048 | ~800 MB | ~6.4 GB |
| 4096 | ~3.2 GB | ~25 GB |

---

## Performance Targets

### GEMM Performance

Target: **≥90% of cuBLAS performance** for matrices ≥1024×1024

```python
# Performance measurement
def measure_gemm_performance(M, N, K, dtype=torch.float16):
    a = torch.randn(M, K, device='cuda', dtype=dtype)
    b = torch.randn(K, N, device='cuda', dtype=dtype)
    
    # Warmup
    for _ in range(10):
        torch.matmul(a, b)
        gemm(a, b)
    
    # Measure cuBLAS
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    cublas_time = start.elapsed_time(end) / 100
    
    # Measure custom GEMM
    start.record()
    for _ in range(100):
        gemm(a, b)
    end.record()
    torch.cuda.synchronize()
    custom_time = start.elapsed_time(end) / 100
    
    return {
        'cublas_ms': cublas_time,
        'custom_ms': custom_time,
        'ratio': custom_time / cublas_time
    }
```

### FlashAttention Performance

| Metric | Target |
|--------|--------|
| Memory Reduction | 90%+ vs standard attention |
| Throughput | Comparable to cuDNN SDPA |

### Memory Complexity

| Implementation | Memory Complexity |
|----------------|-------------------|
| Naive Attention | O(N²) |
| Tiled Attention | O(N²) reduced constant |
| FlashAttention | O(N) |

---

## Optimization Strategies

### 1. Choose the Right Kernel

```python
def optimal_attention(q, k, v, is_causal=False):
    seq_len = q.size(2)
    
    if seq_len >= 512:
        # FlashAttention for long sequences (O(N) memory)
        return flash_attention(q, k, v, is_causal=is_causal)
    elif seq_len >= 128:
        # Tiled for medium sequences
        return tiled_attention(q, k, v)
    else:
        # Naive may be faster for very short sequences
        return naive_attention(q, k, v)
```

### 2. Precision Selection

```python
# FP16 for memory efficiency (recommended for inference)
q = q.half()
k = k.half()
v = v.half()
output = flash_attention(q, k, v)

# FP32 for training or when precision is critical
q = q.float()
output = flash_attention(q, k, v)

# Tensor Core GEMM with FP32 accumulation
a = a.half()
b = b.half()
c = tensor_core_gemm(a, b)  # FP32 output
```

### 3. Tensor Core Alignment

For optimal Tensor Core performance, ensure dimensions are multiples of 16:

```python
# Good: Aligned dimensions
M, K, N = 1024, 512, 1024  # All multiples of 16

# Acceptable: Non-aligned (still correct, may be slower)
M, K, N = 100, 50, 75

# Best practice: Pad to alignment
def pad_to_alignment(size, alignment=16):
    return ((size + alignment - 1) // alignment) * alignment
```

### 4. Batch Size Optimization

```python
# Larger batches are more efficient
# But watch out for memory limits

# For attention
def compute_attention_memory(batch, heads, seq_len, head_dim, dtype=torch.float16):
    bytes_per_elem = 2 if dtype == torch.float16 else 4
    params = batch * heads * seq_len * head_dim * bytes_per_elem
    activation = batch * heads * seq_len * seq_len * bytes_per_elem  # Naive only
    return params, activation

# Find optimal batch size
def find_optimal_batch(max_memory_gb, heads, seq_len, head_dim):
    for batch in [1, 2, 4, 8, 16, 32]:
        memory = compute_attention_memory(batch, heads, seq_len, head_dim)
        if memory[0] / 1e9 > max_memory_gb:
            return batch // 2
    return batch
```

### 5. Memory Planning

```python
def plan_memory_allocation(config):
    """Plan memory allocation for inference."""
    batch = config['batch_size']
    heads = config['num_heads']
    seq_len = config['seq_len']
    head_dim = config['head_dim']
    
    # Input tensors (Q, K, V)
    input_memory = 3 * batch * heads * seq_len * head_dim * 2  # FP16
    
    # Output tensor (O)
    output_memory = batch * heads * seq_len * head_dim * 2
    
    # FlashAttention working memory
    working_memory = batch * heads * (
        32 * (head_dim + 1) +        # Q tile
        4 * 32 * (head_dim + 1) +    # K/V double buffer
        32 * 33 +                     # Scores
        32 * head_dim +               # Output accumulator
        3 * 32                        # State
    ) * 4  # FP32
    
    total_mb = (input_memory + output_memory + working_memory) / 1e6
    return total_mb
```

---

## Benchmarking

### Attention Benchmark

```bash
# Basic benchmark
python benchmarks/benchmark_attention.py

# Custom configuration
python benchmarks/benchmark_attention.py \
    --seq-lengths 512 1024 2048 4096 \
    --batch-size 2 \
    --num-heads 16 \
    --head-dim 64 \
    --dtype fp16 \
    --warmup 20 \
    --iterations 200

# Export results
python benchmarks/benchmark_attention.py --output results.json
```

**Output Example:**
```
================================================================================
ATTENTION BENCHMARK RESULTS
================================================================================

 Seq Len | PyTorch(ms) | Naive(ms) | Tiled(ms) | Flash(ms) | Best Speedup
         |             |           |           |           |
--------------------------------------------------------------------------------
      512 |       12.34 |     18.56 |     10.23 |      8.45 |        1.46x
     1024 |       45.67 |     72.34 |     38.45 |     32.12 |        1.42x
     2048 |      178.90 |    289.45 |    152.34 |    125.67 |        1.42x
     4096 |      712.34 |   1156.78 |    612.45 |    498.12 |        1.43x
```

### GEMM Benchmark

```bash
# Basic benchmark
python benchmarks/benchmark_gemm.py

# Custom sizes
python benchmarks/benchmark_gemm.py \
    --sizes 1024x1024x1024 2048x2048x2048 4096x4096x4096 \
    --dtype fp16 \
    --output gemm_results.json
```

**Output Example:**
```
================================================================================
GEMM BENCHMARK RESULTS
================================================================================

 Size              | cuBLAS(ms) | Custom(ms) | TC GEMM(ms) | Custom % | TC %
 (M x N x K)       |            |            |             | of cuBLAS| of cuBLAS
--------------------------------------------------------------------------------
 1024x1024x1024    |      8.45 |      9.23 |       8.67 |    91.5% | 97.4%
 2048x2048x2048    |     62.34 |     68.45 |      63.12 |    91.1% | 98.7%
 4096x4096x4096    |    485.67 |    523.45 |     492.34 |    92.8% | 98.7%
```

### Custom Benchmark Script

```python
import torch
from cuda_llm_ops import flash_attention, gemm
from cuda_llm_ops.profiler import CUDAProfiler

def custom_benchmark():
    profiler = CUDAProfiler()
    
    # Benchmark attention
    attention_metrics = profiler.profile_attention(
        flash_attention,
        batch_size=2,
        num_heads=8,
        seq_len=1024,
        head_dim=64,
        dtype=torch.float16
    )
    
    print(f"Attention: {attention_metrics.elapsed_ms:.2f} ms")
    print(f"Throughput: {attention_metrics.tflops:.2f} TFLOPS")
    print(f"Bandwidth: {attention_metrics.memory_bandwidth_gb:.2f} GB/s")
    print(f"Bottleneck: {attention_metrics.bottleneck.value}")
    
    # Benchmark GEMM
    gemm_metrics = profiler.profile_gemm(
        gemm,
        M=1024, N=1024, K=1024,
        dtype=torch.float16
    )
    
    print(f"GEMM: {gemm_metrics.elapsed_ms:.2f} ms")
    print(f"Throughput: {gemm_metrics.tflops:.2f} TFLOPS")
```

---

## Profiling

### Using Nsight Compute

```bash
# Profile FlashAttention kernel
ncu --set full -o flash_attention_profile \
    python -c "
import torch
from cuda_llm_ops import flash_attention
q = torch.randn(1, 8, 1024, 64, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
for _ in range(10): flash_attention(q, k, v)
"

# Profile GEMM kernel
ncu --set full -o gemm_profile \
    python -c "
import torch
from cuda_llm_ops import gemm
a = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
b = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
for _ in range(10): gemm(a, b)
"
```

### Key Metrics to Monitor

| Metric | Description | Target |
|--------|-------------|--------|
| SM Occupancy | Utilization of streaming multiprocessors | >50% |
| Memory Throughput | Memory bandwidth utilization | >70% |
| Compute Throughput | FLOPS utilization | Depends on workload |
| L2 Hit Rate | Cache efficiency | >50% |
| Warp Execution Efficiency | Active threads per warp | >80% |

### Python Profiler

```python
from cuda_llm_ops.profiler import CUDAProfiler

profiler = CUDAProfiler()

# Compare with reference
results = profiler.compare_with_reference(
    custom_func=flash_attention,
    reference_func=lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v),
    q, k, v  # arguments
)

print(f"Speedup: {results['speedup']:.2f}x")
print(f"Custom time: {results['custom_ms']:.2f} ms")
print(f"Reference time: {results['reference_ms']:.2f} ms")
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Use `flash_attention` instead of `naive_attention`
- Reduce batch size
- Use FP16 instead of FP32
- Check for memory leaks with `torch.cuda.memory_summary()`

```python
# Monitor memory
print(torch.cuda.memory_summary())

# Clear cache
torch.cuda.empty_cache()
```

#### 2. Shared Memory Limit

```
RuntimeError: naive_attention: seq_len=4096 requires 16404 bytes shared memory, 
but device max is 49152 bytes.
```

**Solution:** Use `flash_attention` or `tiled_attention` for long sequences.

#### 3. INT8 Tensor Core Not Available

```
RuntimeError: INT8 Tensor Core requires Turing+ architecture (SM 7.2+)
```

**Solution:** INT8 GEMM requires Turing or newer GPU. Check your GPU:

```python
import torch
print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
# (7, 5) = Turing, (8, 0) = Ampere, etc.
```

#### 4. Slow Performance

**Check alignment:**
```python
# Ensure dimensions are multiples of 16 for Tensor Core
def check_alignment(M, N, K):
    for dim, name in [(M, 'M'), (N, 'N'), (K, 'K')]:
        if dim % 16 != 0:
            print(f"Warning: {name}={dim} is not aligned to 16")
```

**Check memory bandwidth:**
```python
# Compare with theoretical peak
def get_memory_bandwidth_peak():
    # Example: A100 = 1555 GB/s, V100 = 900 GB/s
    props = torch.cuda.get_device_properties(0)
    return props.memory_bandwidth  # May not be available
```

### Performance Debugging Checklist

- [ ] Using FP16 for inference
- [ ] Dimensions aligned to 16 for Tensor Core
- [ ] Using FlashAttention for sequences > 512
- [ ] Adequate warmup before timing
- [ ] `torch.cuda.synchronize()` before/after timing
- [ ] Not running other GPU processes
- [ ] Latest CUDA driver installed

---

## Best Practices Summary

### For Inference

1. Use FP16 precision
2. Use FlashAttention for sequences > 512
3. Align dimensions to multiples of 16
4. Batch requests when possible

### For Development

1. Profile before optimizing
2. Compare against cuBLAS/cuDNN baselines
3. Monitor memory usage
4. Test on target GPU architecture

### For Production

1. Pre-allocate memory pools
2. Use CUDA streams for async execution
3. Implement graceful degradation for memory limits
4. Monitor GPU utilization metrics
