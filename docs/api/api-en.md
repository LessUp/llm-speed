---
title: API Reference
description: Complete API documentation for LLM-Speed
---

# API Reference

Complete API documentation for LLM-Speed.

---

## Table of Contents

- [Installation](#installation)
- [Module Overview](#module-overview)
- [Attention Functions](#attention-functions)
  - [flash_attention](#flash_attention)
  - [tiled_attention](#tiled_attention)
  - [naive_attention](#naive_attention)
- [GEMM Functions](#gemm-functions)
  - [gemm](#gemm)
  - [tensor_core_gemm](#tensor_core_gemm)
  - [tensor_core_gemm_int8](#tensor_core_gemm_int8)
- [Tensor Requirements](#tensor-requirements)
- [Error Handling](#error-handling)
- [Performance Tips](#performance-tips)
- [Version Information](#version-information)

---

## Installation

```bash
# Install from source
pip install -e .

# Verify installation
python -c "import cuda_llm_ops; print(cuda_llm_ops.__version__)"
```

## Module Overview

```python
import cuda_llm_ops

# List all available functions
dir(cuda_llm_ops)
# ['flash_attention', 'tiled_attention', 'naive_attention',
#  'gemm', 'tensor_core_gemm', 'tensor_core_gemm_int8', '__version__']
```

---

## Attention Functions

### flash_attention

FlashAttention with O(N) memory complexity using online softmax algorithm.

```python
def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 0.0,
    is_causal: bool = False
) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | `torch.Tensor` | Required | Query tensor, shape `[batch, heads, seq_len, head_dim]` |
| `k` | `torch.Tensor` | Required | Key tensor, shape `[batch, heads, seq_len, head_dim]` |
| `v` | `torch.Tensor` | Required | Value tensor, shape `[batch, heads, seq_len, head_dim]` |
| `scale` | `float` | `0.0` | Attention scale factor. If `0.0`, uses `1/√head_dim` |
| `is_causal` | `bool` | `False` | Enable causal mask for autoregressive models |

#### Returns

| Type | Description |
|------|-------------|
| `torch.Tensor` | Output tensor, shape `[batch, heads, seq_len, head_dim]` |

#### Raises

| Exception | Condition |
|-----------|-----------|
| `RuntimeError` | Input tensors not 4D |
| `RuntimeError` | Input tensors not on CUDA device |
| `RuntimeError` | Input tensors not contiguous |
| `RuntimeError` | Shape mismatch between Q, K, V |
| `RuntimeError` | Unsupported dtype (not float32/float16) |

#### Examples

```python
import torch
from cuda_llm_ops import flash_attention

# Standard attention
batch, heads, seq_len, head_dim = 2, 8, 512, 64
q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

output = flash_attention(q, k, v)
print(output.shape)  # torch.Size([2, 8, 512, 64])

# Causal attention (for autoregressive models like GPT)
output_causal = flash_attention(q, k, v, is_causal=True)

# Custom scale factor
output_scaled = flash_attention(q, k, v, scale=0.125)  # 1/8 instead of 1/√64
```

#### Memory Usage Comparison

| Sequence Length | Standard Attention | FlashAttention | Reduction |
|-----------------|-------------------|----------------|-----------|
| 1024 | 4 MB | 0.25 MB | 94% |
| 2048 | 16 MB | 0.5 MB | 97% |
| 4096 | 64 MB | 1 MB | 98% |

---

### tiled_attention

Tiled attention with shared memory optimization. Suitable for medium-length sequences.

```python
def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 0.0,
    is_causal: bool = False
) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | `torch.Tensor` | Required | Query tensor, shape `[batch, heads, seq_len, head_dim]` |
| `k` | `torch.Tensor` | Required | Key tensor, shape `[batch, heads, seq_len, head_dim]` |
| `v` | `torch.Tensor` | Required | Value tensor, shape `[batch, heads, seq_len, head_dim]` |
| `scale` | `float` | `0.0` | Attention scale factor. If `0.0`, uses `1/√head_dim` |
| `is_causal` | `bool` | `False` | Enable causal mask for autoregressive models |

#### Returns

Same as `flash_attention`.

#### Examples

```python
from cuda_llm_ops import tiled_attention

output = tiled_attention(q, k, v)

# With custom scale
output_scaled = tiled_attention(q, k, v, scale=0.1)
```

#### Notes

- More efficient than `naive_attention` for sequences ≥128
- Still stores attention matrix internally (O(N²) memory)
- Not recommended for sequences >2048

---

### naive_attention

Baseline attention implementation with O(N²) memory complexity. Used primarily for correctness verification.

```python
def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 0.0,
    is_causal: bool = False
) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | `torch.Tensor` | Required | Query tensor, shape `[batch, heads, seq_len, head_dim]` |
| `k` | `torch.Tensor` | Required | Key tensor, shape `[batch, heads, seq_len, head_dim]` |
| `v` | `torch.Tensor` | Required | Value tensor, shape `[batch, heads, seq_len, head_dim]` |
| `scale` | `float` | `0.0` | Attention scale factor. If `0.0`, uses `1/√head_dim` |
| `is_causal` | `bool` | `False` | Enable causal mask for autoregressive models |

#### Returns

Same as `flash_attention`.

#### Warning

> **Memory Alert**: This implementation stores the full N×N attention matrix. For long sequences (N > 1024), this may cause out-of-memory errors. Use `flash_attention` for production workloads.

#### Examples

```python
from cuda_llm_ops import naive_attention

# Only recommended for short sequences or testing
q = torch.randn(2, 4, 64, 32, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

output = naive_attention(q, k, v)

# Verify correctness against PyTorch reference
reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
assert torch.allclose(output, reference, rtol=1e-3, atol=1e-3)
```

---

## GEMM Functions

### gemm

High-performance general matrix multiplication with register tiling.

```python
def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    trans_a: bool = False,
    trans_b: bool = False
) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `a` | `torch.Tensor` | Required | Matrix A, shape `[M, K]` (or `[K, M]` if `trans_a=True`) |
| `b` | `torch.Tensor` | Required | Matrix B, shape `[K, N]` (or `[N, K]` if `trans_b=True`) |
| `alpha` | `float` | `1.0` | Scaling factor for A @ B |
| `beta` | `float` | `0.0` | Scaling factor for existing C (currently unused) |
| `trans_a` | `bool` | `False` | Transpose matrix A |
| `trans_b` | `bool` | `False` | Transpose matrix B |

#### Returns

| Type | Description |
|------|-------------|
| `torch.Tensor` | Output matrix C, shape `[M, N]` |

#### Raises

| Exception | Condition |
|-----------|-----------|
| `RuntimeError` | Input tensors not 2D |
| `RuntimeError` | Inner dimensions don't match |
| `RuntimeError` | Tensors not on CUDA or not contiguous |
| `RuntimeError` | Unsupported dtype |

#### Examples

```python
import torch
from cuda_llm_ops import gemm

# Standard C = A @ B
M, K, N = 1024, 512, 1024
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
c = gemm(a, b)

# With scaling: C = 2.0 * A @ B
c = gemm(a, b, alpha=2.0)

# Handle transposed matrices
a_t = torch.randn(K, M, device='cuda', dtype=torch.float16)  # Actually A^T
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

# Compute A^T @ B
c = gemm(a_t, b, trans_a=True)
```

#### Layout Equivalence Table

| `trans_a` | `trans_b` | Operation |
|-----------|-----------|-----------|
| False | False | C = A @ B |
| False | True | C = A @ B^T |
| True | False | C = A^T @ B |
| True | True | C = A^T @ B^T |

---

### tensor_core_gemm

Tensor Core accelerated matrix multiplication using WMMA API.

```python
def tensor_core_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0
) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `torch.Tensor` | Matrix A, shape `[M, K]`, must be `float16` |
| `b` | `torch.Tensor` | Matrix B, shape `[K, N]`, must be `float16` |
| `alpha` | `float` | Scaling factor |
| `beta` | `float` | Scaling factor for existing C |

#### Returns

| Type | Description |
|------|-------------|
| `torch.Tensor` | Output matrix C, shape `[M, N]`, dtype `float32` |

#### Hardware Requirements

- **GPU**: Volta architecture or newer (SM 7.0+)
- **PyTorch**: 2.0+

#### Examples

```python
import torch
from cuda_llm_ops import tensor_core_gemm

a = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
b = torch.randn(512, 1024, device='cuda', dtype=torch.float16)

# Returns FP32 output for higher precision accumulation
c = tensor_core_gemm(a, b)
print(c.dtype)  # torch.float32

# Scaling
c = tensor_core_gemm(a, b, alpha=0.5)
```

#### Performance Tips

- Align M, K, N to multiples of 16 for optimal performance
- Use `float16` input for maximum throughput
- Output precision is `float32` for numerical stability

---

### tensor_core_gemm_int8

INT8 quantized matrix multiplication using Tensor Cores.

```python
def tensor_core_gemm_int8(
    a: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `torch.Tensor` | Matrix A, shape `[M, K]`, must be `int8` |
| `b` | `torch.Tensor` | Matrix B, shape `[K, N]`, must be `int8` |

#### Returns

| Type | Description |
|------|-------------|
| `torch.Tensor` | Output matrix C, shape `[M, N]`, dtype `int32` |

#### Hardware Requirements

- **GPU**: Turing architecture or newer (SM 7.2+)
- **PyTorch**: 2.0+

#### Examples

```python
import torch
from cuda_llm_ops import tensor_core_gemm_int8

# Create INT8 tensors
a = torch.randint(-128, 127, (1024, 512), device='cuda', dtype=torch.int8)
b = torch.randint(-128, 127, (512, 1024), device='cuda', dtype=torch.int8)

# INT32 accumulation for precision
c = tensor_core_gemm_int8(a, b)
print(c.dtype)  # torch.int32

# Verify against reference
reference = torch.matmul(a.to(torch.int32), b.to(torch.int32))
assert torch.equal(c, reference)
```

#### Checking GPU Compatibility

```python
import torch
 capability = torch.cuda.get_device_capability()
if capability[0] > 7 or (capability[0] == 7 and capability[1] >= 2):
    print("INT8 Tensor Core is supported")
else:
    print("Requires Turing+ GPU (SM 7.2+)")
```

---

## Tensor Requirements

### Attention Functions

| Requirement | Specification |
|-------------|---------------|
| Dimensions | 4D: `[batch, heads, seq_len, head_dim]` |
| Device | Must be CUDA (`tensor.is_cuda == True`) |
| Layout | Contiguous (`tensor.is_contiguous() == True`) |
| Dtype | `float32` or `float16` |
| Shape Consistency | Q, K, V shapes must match exactly |

### GEMM Functions

| Requirement | Specification |
|-------------|---------------|
| Dimensions | 2D: `[M, K]` and `[K, N]` |
| Device | Must be CUDA |
| Layout | Contiguous |
| Dtype | `float32`, `float16`, or `int8` |
| Dimension Alignment | Inner dimensions must match |

---

## Error Handling

### Common Error Messages

```python
# Example: Wrong dimensions
q = torch.randn(64, 32, device='cuda')  # 2D instead of 4D
flash_attention(q, k, v)
# RuntimeError: Q must be 4D tensor [batch, heads, seq_len, head_dim]

# Example: CPU tensor
q = torch.randn(2, 4, 64, 32)  # CPU tensor
flash_attention(q, k, v)
# RuntimeError: Q must be on CUDA device

# Example: Non-contiguous tensor
q = torch.randn(2, 4, 64, 32, device='cuda').transpose(1, 2)
flash_attention(q, k, v)
# RuntimeError: Q must be contiguous

# Example: Shape mismatch
q = torch.randn(2, 4, 64, 32, device='cuda')
v = torch.randn(2, 4, 128, 32, device='cuda')  # Different seq_len
flash_attention(q, k, v)
# RuntimeError: K and V must have same shape

# Example: Unsupported dtype
q = torch.randn(2, 4, 64, 32, device='cuda', dtype=torch.int32)
flash_attention(q, k, v)
# RuntimeError: Only float32 and float16 are supported
```

### Error Handling Pattern

```python
import torch
from cuda_llm_ops import flash_attention

def safe_flash_attention(q, k, v, **kwargs):
    try:
        return flash_attention(q, k, v, **kwargs)
    except RuntimeError as e:
        error_msg = str(e)
        if "must be on CUDA device" in error_msg:
            print("Error: Please move tensors to CUDA using .cuda()")
        elif "must be 4D tensor" in error_msg:
            print("Error: Input shapes must be [batch, heads, seq_len, head_dim]")
        elif "must have same shape" in error_msg:
            print("Error: Q, K, V tensors must have identical shapes")
        else:
            print(f"Error: {error_msg}")
        raise
```

---

## Performance Tips

### Memory Optimization

```python
# Use FlashAttention for long sequences
seq_len = 1024
if seq_len >= 512:
    output = flash_attention(q, k, v)  # O(N) memory
else:
    output = naive_attention(q, k, v)  # May be faster for short sequences
```

### Precision Selection

```python
# FP16 for inference (recommended)
q_fp16 = q.half()
output = flash_attention(q_fp16, k_fp16, v_fp16)

# FP32 for training or when precision is critical
output = flash_attention(q.float(), k.float(), v.float())

# Tensor Core GEMM for FP16 inputs with FP32 accumulation
c = tensor_core_gemm(a.half(), b.half())  # Returns FP32
```

### Optimal Dimensions

```python
# For best Tensor Core performance, use multiples of 16
def round_up_to_16(x):
    return ((x + 15) // 16) * 16

M = round_up_to_16(1000)  # 1008
N = round_up_to_16(500)   # 512
```

### Batch Processing

```python
# Process multiple sequences together for better GPU utilization
batch_size = 8  # Adjust based on available memory
q = torch.randn(batch_size, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
```

---

## Version Information

```python
import cuda_llm_ops

print(cuda_llm_ops.__version__)
# e.g., "0.3.0"
```

---

## Support

- **Issues**: https://github.com/LessUp/llm-speed/issues
- **Documentation**: https://lessup.github.io/llm-speed/

[← Back to Documentation](../)
