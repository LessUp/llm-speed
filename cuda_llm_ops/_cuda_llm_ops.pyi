"""
Type stubs for CUDA LLM Kernel Optimization extension module.

This module provides high-performance CUDA kernels for LLM inference,
including FlashAttention and Tensor Core GEMM implementations.
"""

import torch

# ==============================================================================
# Attention Functions
# ==============================================================================

def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = ...,
    is_causal: bool = ...,
) -> torch.Tensor:
    """
    Naive attention implementation (baseline).

    Args:
        q: Query tensor of shape [batch, heads, seq_len, head_dim].
           Must be contiguous and on CUDA device.
           Supported dtypes: float32, float16.
        k: Key tensor of shape [batch, heads, seq_len, head_dim].
           Must have same shape and dtype as q.
        v: Value tensor of shape [batch, heads, seq_len, head_dim].
           Must have same shape and dtype as q.
        scale: Softmax scale factor. If 0.0, uses 1/sqrt(head_dim).
        is_causal: If True, applies causal mask for autoregressive models.

    Returns:
        Output tensor of shape [batch, heads, seq_len, head_dim] with same dtype as input.
    """
    ...

def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = ...,
    is_causal: bool = ...,
) -> torch.Tensor:
    """
    Tiled attention with shared memory optimization.

    Args:
        q: Query tensor of shape [batch, heads, seq_len, head_dim].
           Must be contiguous and on CUDA device.
           Supported dtypes: float32, float16.
        k: Key tensor of shape [batch, heads, seq_len, head_dim].
           Must have same shape and dtype as q.
        v: Value tensor of shape [batch, heads, seq_len, head_dim].
           Must have same shape and dtype as q.
        scale: Softmax scale factor. If 0.0, uses 1/sqrt(head_dim).
        is_causal: If True, applies causal mask for autoregressive models.

    Returns:
        Output tensor of shape [batch, heads, seq_len, head_dim] with same dtype as input.
    """
    ...

def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = ...,
    is_causal: bool = ...,
) -> torch.Tensor:
    """
    FlashAttention with online softmax for O(N) memory complexity.

    Args:
        q: Query tensor of shape [batch, heads, seq_len, head_dim].
           Must be contiguous and on CUDA device.
           Supported dtypes: float32, float16.
        k: Key tensor of shape [batch, heads, seq_len, head_dim].
           Must have same shape and dtype as q.
        v: Value tensor of shape [batch, heads, seq_len, head_dim].
           Must have same shape and dtype as q.
        scale: Softmax scale factor. If 0.0, uses 1/sqrt(head_dim).
        is_causal: If True, applies causal mask for autoregressive models.

    Returns:
        Output tensor of shape [batch, heads, seq_len, head_dim] with same dtype as input.
    """
    ...

# ==============================================================================
# GEMM Functions
# ==============================================================================

def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = ...,
    beta: float = ...,
    trans_a: bool = ...,
    trans_b: bool = ...,
) -> torch.Tensor:
    """
    High-performance GEMM with register tiling.

    Computes: output = alpha * (A @ B) + beta * output

    Args:
        a: Left operand tensor of shape [M, K] (or [K, M] if trans_a=True).
           Must be contiguous and on CUDA device.
           Supported dtypes: float32, float16.
        b: Right operand tensor of shape [K, N] (or [N, K] if trans_b=True).
           Must be contiguous and on CUDA device.
           Must have same dtype as a.
        alpha: Scaling factor for the product. Default: 1.0.
        beta: Scaling factor for the output initialization. Default: 0.0.
        trans_a: If True, transpose A before multiplication.
        trans_b: If True, transpose B before multiplication.

    Returns:
        Output tensor of shape [M, N] with same dtype as input.
    """
    ...

def tensor_core_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = ...,
    beta: float = ...,
) -> torch.Tensor:
    """
    Tensor Core GEMM using FP16 input with FP32 accumulation.

    Computes: output = alpha * (A @ B) + beta * output

    Args:
        a: Left operand tensor of shape [M, K].
           Must be contiguous, on CUDA device, and float16 dtype.
        b: Right operand tensor of shape [K, N].
           Must be contiguous, on CUDA device, and float16 dtype.
        alpha: Scaling factor for the product. Default: 1.0.
        beta: Scaling factor for the output initialization. Default: 0.0.

    Returns:
        Output tensor of shape [M, N] with float32 dtype.
    """
    ...

def tensor_core_gemm_int8(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Tensor Core GEMM using INT8 input with INT32 accumulation.

    Computes: output = A @ B (no scaling factors)

    Requires Turing+ architecture (SM >= 7.2).

    Args:
        a: Left operand tensor of shape [M, K].
           Must be contiguous, on CUDA device, and int8 dtype.
        b: Right operand tensor of shape [K, N].
           Must be contiguous, on CUDA device, and int8 dtype.

    Returns:
        Output tensor of shape [M, N] with int32 dtype.
    """
    ...
