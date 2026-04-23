# Design: FlashAttention Backward Pass

## Algorithm Overview

The backward pass computes gradients for Q, K, V given upstream gradient dO.

### Mathematical Foundation

Given:
- Forward: O = softmax(QK^T/√d)V
- Loss gradient: dO

Compute:
- dV = softmax^T · dO
- dS = dO · V^T
- dQ = dS · K / √d
- dK = dS^T · Q / √d

### Memory-Efficient Backward

The backward pass can be implemented with O(N) memory using:
- Recomputation of attention scores (instead of storing from forward)
- Chunked processing similar to forward pass

## Implementation

### CUDA Kernel

```cpp
// src/flash_attention_backward.cu
template<typename T, int BLOCK_M, int BLOCK_N>
__global__ void flash_attention_backward_kernel(
    const T* Q, const T* K, const T* V,
    const T* O, const T* dO,
    T* dQ, T* dK, T* dV,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool is_causal
);
```

### Shared Memory Strategy

Re-use the same double-buffered layout from forward pass.
Additional buffers needed for intermediate dS computation.

### Autograd Integration

```python
# cuda_llm_ops/flash_attention.py
import torch

class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, is_causal):
        output = flash_attention_forward(q, k, v, scale, is_causal)
        ctx.save_for_backward(q, k, v, output)
        ctx.scale = scale
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, output = ctx.saved_tensors
        dq, dk, dv = flash_attention_backward(
            q, k, v, output, grad_output,
            ctx.scale, ctx.is_causal
        )
        return dq, dk, dv, None, None
```

### Files to Create/Modify

| File | Action |
|------|--------|
| `src/flash_attention_backward.cu` | Create |
| `cuda_llm_ops/flash_attention.py` | Create - autograd wrapper |
| `cuda_llm_ops/bindings.cpp` | Modify - add backward binding |
| `tests/test_flash_attention_backward.py` | Create |

## Causal Masking in Backward

For causal attention, the backward pass must respect the causal mask:
- dQ[i] only depends on K[j≤i]
- dK[i] only depends on Q[j≥i]

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Numerical precision in gradients | FP32 accumulation, gradient check tests |
| Memory overhead | Recomputation strategy |
| Performance | Profile and optimize tile sizes |
