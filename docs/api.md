# API Reference

Complete API documentation for the CUDA LLM Kernel Optimization library.

## Installation

```bash
pip install -e .
```

## Module: `cuda_llm_ops`

```python
import cuda_llm_ops
# or
from cuda_llm_ops import flash_attention, gemm
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

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | `torch.Tensor` | Query tensor, shape `[batch, heads, seq_len, head_dim]` |
| `k` | `torch.Tensor` | Key tensor, shape `[batch, heads, seq_len, head_dim]` |
| `v` | `torch.Tensor` | Value tensor, shape `[batch, heads, seq_len, head_dim]` |
| `scale` | `float` | Attention scale factor. Default: `1/sqrt(head_dim)` if 0.0 |
| `is_causal` | `bool` | Enable causal mask for autoregressive models. Default: `False` |

**Returns:**

| Type | Description |
|------|-------------|
| `torch.Tensor` | Output tensor, shape `[batch, heads, seq_len, head_dim]` |

**Raises:**

- `RuntimeError`: If inputs are not 4D, not on CUDA, not contiguous, or have mismatched shapes
- `RuntimeError`: If dtype is not `float32` or `float16`

**Example:**

```python
import torch
from cuda_llm_ops import flash_attention

# Standard attention
batch, heads, seq_len, head_dim = 2, 8, 512, 64
q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

output = flash_attention(q, k, v)
print(output.shape)  # [2, 8, 512, 64]

# Causal attention (for autoregressive models)
output_causal = flash_attention(q, k, v, is_causal=True)

# Custom scale
output_scaled = flash_attention(q, k, v, scale=0.1)
```

**Memory Usage:**

| Sequence Length | Standard Attention | FlashAttention |
|-----------------|-------------------|----------------|
| 1024 | 4 MB | 0.25 MB |
| 2048 | 16 MB | 0.5 MB |
| 4096 | 64 MB | 1 MB |

---

### tiled_attention

Tiled attention with shared memory optimization.

```python
def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 0.0
) -> torch.Tensor
```

**Parameters:**

Same as `flash_attention`, except `is_causal` is not supported.

**Returns:**

Same as `flash_attention`.

**Example:**

```python
from cuda_llm_ops import tiled_attention

output = tiled_attention(q, k, v)
```

---

### naive_attention

Baseline attention implementation with O(N²) memory complexity.

```python
def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 0.0
) -> torch.Tensor
```

**Parameters:**

Same as `tiled_attention`.

**Returns:**

Same as `flash_attention`.

**Note:** This implementation stores the full N×N attention matrix, so it may run out of memory for long sequences. Use `flash_attention` for long sequences.

**Example:**

```python
from cuda_llm_ops import naive_attention

output = naive_attention(q, k, v)
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

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `torch.Tensor` | Matrix A, shape `[M, K]` (or `[K, M]` if `trans_a=True`) |
| `b` | `torch.Tensor` | Matrix B, shape `[K, N]` (or `[N, K]` if `trans_b=True`) |
| `alpha` | `float` | Scaling factor for A @ B. Default: `1.0` |
| `beta` | `float` | Scaling factor for existing C (currently not used). Default: `0.0` |
| `trans_a` | `bool` | Transpose matrix A. Default: `False` |
| `trans_b` | `bool` | Transpose matrix B. Default: `False` |

**Returns:**

| Type | Description |
|------|-------------|
| `torch.Tensor` | Output matrix C = alpha × A @ B + beta × C, shape `[M, N]` |

**Raises:**

- `RuntimeError`: If inputs are not 2D, not on CUDA, not contiguous, or inner dimensions don't match
- `RuntimeError`: If dtype is not `float32` or `float16`

**Example:**

```python
import torch
from cuda_llm_ops import gemm

# Standard matrix multiplication
M, K, N = 1024, 512, 1024
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

c = gemm(a, b)
print(c.shape)  # [1024, 1024]

# With scaling
c = gemm(a, b, alpha=2.0)

# Transposed operations
a_t = torch.randn(K, M, device='cuda', dtype=torch.float16)  # A^T
b_t = torch.randn(N, K, device='cuda', dtype=torch.float16)  # B^T

# C = A^T @ B
c = gemm(a_t, b, trans_a=True)

# C = A @ B^T
c = gemm(a, b_t, trans_b=True)

# C = A^T @ B^T
c = gemm(a_t, b_t, trans_a=True, trans_b=True)
```

**Layout Equivalence:**

| trans_a | trans_b | Operation |
|---------|---------|-----------|
| False | False | C = A @ B |
| False | True | C = A @ B^T |
| True | False | C = A^T @ B |
| True | True | C = A^T @ B^T |

---

### tensor_core_gemm

Tensor Core accelerated matrix multiplication (FP16 input, FP32 output).

```python
def tensor_core_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0
) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `torch.Tensor` | Matrix A, shape `[M, K]`, must be `float16` |
| `b` | `torch.Tensor` | Matrix B, shape `[K, N]`, must be `float16` |
| `alpha` | `float` | Scaling factor. Default: `1.0` |
| `beta` | `float` | Scaling factor for existing C. Default: `0.0` |

**Returns:**

| Type | Description |
|------|-------------|
| `torch.Tensor` | Output matrix C, shape `[M, N]`, dtype `float32` |

**Raises:**

- `RuntimeError`: If inputs are not `float16`
- Same as `gemm` for other validations

**Example:**

```python
import torch
from cuda_llm_ops import tensor_core_gemm

a = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
b = torch.randn(512, 1024, device='cuda', dtype=torch.float16)

# Returns FP32 output for higher precision
c = tensor_core_gemm(a, b)
print(c.dtype)  # torch.float32
```

**Performance:** Utilizes Tensor Core hardware for accelerated computation on Volta+ GPUs.

---

### tensor_core_gemm_int8

INT8 quantized matrix multiplication (requires Turing+ SM≥7.2).

```python
def tensor_core_gemm_int8(
    a: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `torch.Tensor` | Matrix A, shape `[M, K]`, must be `int8` |
| `b` | `torch.Tensor` | Matrix B, shape `[K, N]`, must be `int8` |

**Returns:**

| Type | Description |
|------|-------------|
| `torch.Tensor` | Output matrix C, shape `[M, N]`, dtype `int32` |

**Raises:**

- `RuntimeError`: If inputs are not `int8`
- `RuntimeError`: If GPU architecture is below SM 7.2 (Turing)

**Example:**

```python
import torch
from cuda_llm_ops import tensor_core_gemm_int8

a = torch.randint(-128, 127, (1024, 512), device='cuda', dtype=torch.int8)
b = torch.randint(-128, 127, (512, 1024), device='cuda', dtype=torch.int8)

# INT32 accumulation for precision
c = tensor_core_gemm_int8(a, b)
print(c.dtype)  # torch.int32

# Verify correctness
reference = torch.matmul(a.to(torch.int32), b.to(torch.int32))
assert torch.equal(c, reference)
```

**Note:** This function requires a Turing or newer GPU (SM 7.2+).

---

## Tensor Requirements

### Attention Functions

| Requirement | Description |
|-------------|-------------|
| Dimensions | 4D tensor: `[batch, heads, seq_len, head_dim]` |
| Device | Must be on CUDA |
| Memory Layout | Must be contiguous |
| Dtype | `float32` or `float16` |
| Shape Matching | Q, K, V must have identical shapes |

### GEMM Functions

| Requirement | Description |
|-------------|-------------|
| Dimensions | 2D tensor |
| Device | Must be on CUDA |
| Memory Layout | Must be contiguous |
| Dtype | `float32`, `float16`, or `int8` (quantized only) |
| Dimension Alignment | Inner dimensions must match (K dimension) |

---

## Error Messages

### Common Errors

```python
# Wrong dimensions
q = torch.randn(64, 32, device='cuda')  # 2D instead of 4D
flash_attention(q, k, v)
# RuntimeError: Q must be 4D tensor [batch, heads, seq_len, head_dim]

# CPU tensor
q = torch.randn(2, 4, 64, 32)  # CPU tensor
flash_attention(q, k, v)
# RuntimeError: Q must be on CUDA device

# Non-contiguous tensor
q = torch.randn(2, 4, 64, 32, device='cuda').transpose(1, 2)
flash_attention(q, k, v)
# RuntimeError: Q must be contiguous

# Shape mismatch
q = torch.randn(2, 4, 64, 32, device='cuda')
v = torch.randn(2, 4, 128, 32, device='cuda')  # Different seq_len
flash_attention(q, k, v)
# RuntimeError: K and V must have same shape

# Unsupported dtype
q = torch.randn(2, 4, 64, 32, device='cuda', dtype=torch.int32)
flash_attention(q, k, v)
# RuntimeError: Only float32 and float16 are supported

# Empty tensor
q = torch.randn(0, 4, 64, 32, device='cuda')
flash_attention(q, k, v)
# RuntimeError: Tensor dimensions must be positive (non-zero)
```

---

## Performance Tips

### Memory Optimization

```python
# Use FlashAttention for long sequences
if seq_len > 512:
    output = flash_attention(q, k, v)  # O(N) memory
else:
    output = naive_attention(q, k, v)  # May be faster for short sequences
```

### Precision Selection

```python
# Use FP16 for memory efficiency and speed
q_fp16 = q.half()
output = flash_attention(q_fp16, k_fp16, v_fp16)

# Use Tensor Core GEMM for FP16 inputs with FP32 precision
c = tensor_core_gemm(a_fp16, b_fp16)  # FP32 output
```

### Matrix Alignment

```python
# For best Tensor Core performance, align dimensions to 16
M, K, N = 1024, 512, 1024  # All multiples of 16

# Non-aligned dimensions still work but may be slower
M, K, N = 100, 50, 75
```

---

## Version

```python
import cuda_llm_ops
print(cuda_llm_ops.__version__)  # e.g., "0.1.0"
```
