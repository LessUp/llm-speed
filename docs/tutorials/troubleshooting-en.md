---
layout: docs
title: Troubleshooting Guide
description: Solutions to common issues when using CUDA LLM Kernel Optimization
lang: en
---

# Troubleshooting Guide

Solutions to common issues when using CUDA LLM Kernel Optimization.

---

## Table of Contents

- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Numerical Issues](#numerical-issues)
- [Getting Help](#getting-help)

---

## Installation Issues

### Baseline environment not prepared

**Symptom:**
```
ImportError / ModuleNotFoundError during test collection
```

**Reason:**
Validation started before the documented local Python environment was prepared.

**Fix:**
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt pytest hypothesis ruff pre-commit
```

### CUDA Not Found

**Error:**
```
RuntimeError: CUDA not available. Please check your CUDA installation.
```

**Solutions:**

1. **Verify CUDA installation:**
```bash
nvcc --version
nvidia-smi
```

2. **Check PyTorch CUDA support:**
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

3. **Reinstall PyTorch with correct CUDA version:**
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Build Errors

**Error:**
```
error: command 'gcc' failed with exit status 1
```

**Solutions:**

1. **Check GCC version:**
```bash
gcc --version  # Need GCC 9.0+
```

2. **Set CUDA architecture flags:**
```bash
# For specific GPU architecture
CUDA_ARCHS="80" pip install -e .  # A100

# For multiple architectures
CUDA_ARCHS="75;80;86" pip install -e .
```

3. **Common fixes:**
```bash
# Clear build cache
rm -rf build/
rm -rf *.egg-info

# Rebuild with verbose output
pip install -e . --verbose
```

### Import Errors

**Error:**
```
ImportError: No module named 'cuda_llm_ops'
```

**Solutions:**

1. **Verify installation:**
```bash
pip list | grep cuda
python -c "import cuda_llm_ops; print(cuda_llm_ops.__version__)"
```

2. **Check Python path:**
```python
import sys
print(sys.path)
```

3. **Reinstall:**
```bash
pip uninstall cuda_llm_ops
pip install -e .
```

---

## Runtime Errors

### CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**

1. **Use FlashAttention (O(N) memory) instead of naive attention:**
```python
# Bad - may OOM for long sequences
from cuda_llm_ops import naive_attention
output = naive_attention(q, k, v)  # O(N²) memory

# Good - memory efficient
from cuda_llm_ops import flash_attention
output = flash_attention(q, k, v)   # O(N) memory
```

2. **Reduce batch size or sequence length:**
```python
# Check memory before operation
print(torch.cuda.memory_summary())

# Try smaller batch
batch_size = 2  # Instead of 8
```

3. **Clear cache:**
```python
torch.cuda.empty_cache()
```

4. **Mixed precision:**
```python
# Use FP16 instead of FP32
q = q.half()
```

### Shared Memory Limit

**Error:**
```
RuntimeError: naive_attention: seq_len=4096 requires 16404 bytes shared memory,
but device max is 49152 bytes.
```

**Solution:**
Use `flash_attention` or `tiled_attention` for long sequences:

```python
if seq_len > 2048:
    output = flash_attention(q, k, v)  # No shared memory limit
else:
    output = tiled_attention(q, k, v)
```

### Tensor Shape Mismatch

**Error:**
```
RuntimeError: K and V must have same shape
```

**Solution:**
Ensure Q, K, V have identical shapes:

```python
print(f"Q shape: {q.shape}")
print(f"K shape: {k.shape}")
print(f"V shape: {v.shape}")

# All should be: [batch, heads, seq_len, head_dim]
assert q.shape == k.shape == v.shape
```

### Wrong Device

**Error:**
```
RuntimeError: Q must be on CUDA device
```

**Solution:**
Move tensors to GPU:

```python
# Check device
print(f"Q device: {q.device}")

# Move to CUDA if needed
q = q.cuda()
# Or during creation
q = torch.randn(..., device='cuda', dtype=torch.float16)
```

### Non-Contiguous Tensors

**Error:**
```
RuntimeError: Q must be contiguous
```

**Solution:**
```python
# Make contiguous
q = q.contiguous()

# Or during transpose
def safe_transpose(tensor, dim0, dim1):
    """Transpose and make contiguous."""
    return tensor.transpose(dim0, dim1).contiguous()
```

### Unsupported Data Type

**Error:**
```
RuntimeError: Only float32 and float16 are supported
```

**Solution:**
```python
# Convert to supported dtype
q = q.half()      # FP16
# or
q = q.float()     # FP32
```

### Wrong Dimensions

**Error:**
```
RuntimeError: Q must be 4D tensor [batch, heads, seq_len, head_dim]
```

**Solution:**
```python
# Expected: [batch, heads, seq_len, head_dim]
print(f"Current shape: {q.shape}")
print(f"Dimensions: {q.dim()}")

# Reshape if needed
q = q.view(batch, heads, seq_len, head_dim)
```

### INT8 Tensor Core Not Available

**Error:**
```
RuntimeError: INT8 Tensor Core requires Turing+ architecture (SM 7.2+)
```

**Solution:**
Check GPU compute capability:

```python
import torch
capability = torch.cuda.get_device_capability()
print(f"Compute capability: {capability}")

if capability[0] > 7 or (capability[0] == 7 and capability[1] >= 2):
    # Turing or better
    from cuda_llm_ops import tensor_core_gemm_int8
    c = tensor_core_gemm_int8(a_int8, b_int8)
else:
    # Fallback to FP16
    from cuda_llm_ops import tensor_core_gemm
    c = tensor_core_gemm(a_fp16, b_fp16)
```

---

## Performance Issues

### Slow Execution

**Symptoms:**
- Operations take much longer than expected
- GPU utilization is low in `nvidia-smi`

**Solutions:**

1. **Use optimal kernel for sequence length:**
```python
seq_len = q.size(2)

if seq_len >= 512:
    output = flash_attention(q, k, v)  # Best for long sequences
elif seq_len >= 128:
    output = tiled_attention(q, k, v)  # Good for medium sequences
else:
    output = naive_attention(q, k, v)  # Okay for short sequences
```

2. **Check alignment:**
```python
def check_alignment(M, N, K):
    for dim, name in [(M, 'M'), (N, 'N'), (K, 'K')]:
        if dim % 16 != 0:
            print(f"Warning: {name}={dim} not aligned to 16")

check_alignment(1024, 512, 1024)
```

3. **Ensure warmup:**
```python
# GPU needs warmup for consistent timing
for _ in range(10):
    _ = flash_attention(q, k, v)
torch.cuda.synchronize()
```

### Low GPU Utilization

**Symptoms:**
- `nvidia-smi` shows utilization < 50%
- CPU bottleneck

**Solutions:**

1. **Increase batch size:**
```python
# Too small
batch = 1
q = torch.randn(1, heads, seq_len, head_dim, device='cuda')

# Better
batch = 8
q = torch.randn(8, heads, seq_len, head_dim, device='cuda')
```

2. **Remove CPU-GPU synchronization:**
```python
# Bad - forces CPU wait
result = flash_attention(q, k, v)
torch.cuda.synchronize()
print(result.cpu())

# Better - batch operations
results = []
for _ in range(100):
    results.append(flash_attention(q, k, v))
torch.cuda.synchronize()
```

### Tensor Core Not Used

**Symptoms:**
- Performance significantly below cuBLAS
- Nsight Compute shows no Tensor Core usage

**Solutions:**

1. **Use Tensor Core variant:**
```python
# Uses regular CUDA cores
c = gemm(a, b)

# Uses Tensor Cores
c = tensor_core_gemm(a, b)
```

2. **Ensure FP16 input:**
```python
# Must be FP16 for Tensor Core
c = tensor_core_gemm(a.half(), b.half())
```

3. **Check alignment:**
```python
# All dimensions should be multiples of 16
M, K, N = 1024, 512, 1024  # Good: all divisible by 16
```

---

## Numerical Issues

### Precision Loss in FP16

**Symptoms:**
- Results differ significantly from FP32 reference
- NaN or Inf values

**Solutions:**

1. **Use Tensor Core GEMM for accumulation:**
```python
# FP16 computation with FP32 accumulation
c = tensor_core_gemm(a_fp16, b_fp16)  # Returns FP32
```

2. **Scale values for FP16:**
```python
# FP16 has limited range [-65504, 65504]
# Scale down large values
scale = 1.0 / 256.0
q = q * scale
output = flash_attention(q, k, v)
output = output / scale
```

3. **Gradient scaling (for training):**
```python
from torch.cuda.amp import GradScaler

scaler = GradScaler()
with torch.cuda.amp.autocast():
    output = flash_attention(q, k, v)
scaler.scale(loss).backward()
```

### Output Does Not Match PyTorch

**Symptoms:**
- Custom kernel output differs from `torch.nn.functional.scaled_dot_product_attention`

**Solutions:**

1. **Check tolerances:**
```python
torch.testing.assert_close(
    output_custom,
    output_torch,
    rtol=1e-3,  # Relative tolerance
    atol=1e-3   # Absolute tolerance
)
```

2. **Expected differences:**
```python
# FP16 has ~3-4 decimal digits of precision
# Small differences are normal for different implementations
```

---

## Getting Help

### Diagnostic Script

Run this to collect system information:

```python
#!/usr/bin/env python3
import sys
import torch
import cuda_llm_ops

print("=" * 60)
print("System Information")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability()}")
    print(f"cuda_llm_ops version: {cuda_llm_ops.__version__}")

print("=" * 60)
print("Quick Test")
print("=" * 60)

try:
    q = torch.randn(2, 4, 64, 32, device='cuda', dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output = cuda_llm_ops.flash_attention(q, k, v)
    print("✓ FlashAttention test passed")
except Exception as e:
    print(f"✗ FlashAttention test failed: {e}")

try:
    a = torch.randn(512, 512, device='cuda', dtype=torch.float16)
    b = torch.randn(512, 512, device='cuda', dtype=torch.float16)
    c = cuda_llm_ops.gemm(a, b)
    print("✓ GEMM test passed")
except Exception as e:
    print(f"✗ GEMM test failed: {e}")
```

### Submit an Issue

When reporting issues, please include:

1. **System information** (from script above)
2. **Minimal reproduction code**
3. **Expected vs actual behavior**
4. **Full error message** with stack trace

Example issue template:

```markdown
## Environment
- GPU: NVIDIA A100
- CUDA: 12.1
- Python: 3.10
- PyTorch: 2.1.0
- cuda_llm_ops: 0.3.0

## Issue Description
FlashAttention fails with OOM on 8K sequence length

## Reproduction Code
```python
import torch
from cuda_llm_ops import flash_attention

q = torch.randn(2, 16, 8192, 64, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
output = flash_attention(q, k, v)  # OOM here
```

## Error Message
RuntimeError: CUDA out of memory. Tried to allocate 4.00 GB
```

### Resources

- **GitHub Issues**: https://github.com/LessUp/llm-speed/issues
- **Documentation**: https://lessup.github.io/llm-speed/
- **Discussions**: https://github.com/LessUp/llm-speed/discussions

---

[← Back to Documentation](../)
