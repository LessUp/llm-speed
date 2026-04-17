"""
Type stubs for CUDA LLM Kernel Optimization extension module.
"""

from typing import Any

def naive_attention(
    q: Any,
    k: Any,
    v: Any,
    scale: float = ...,
    is_causal: bool = ...,
) -> Any: ...
def tiled_attention(
    q: Any,
    k: Any,
    v: Any,
    scale: float = ...,
    is_causal: bool = ...,
) -> Any: ...
def flash_attention(
    q: Any,
    k: Any,
    v: Any,
    scale: float = ...,
    is_causal: bool = ...,
) -> Any: ...
def gemm(
    a: Any,
    b: Any,
    alpha: float = ...,
    beta: float = ...,
    trans_a: bool = ...,
    trans_b: bool = ...,
) -> Any: ...
def tensor_core_gemm(
    a: Any,
    b: Any,
    alpha: float = ...,
    beta: float = ...,
) -> Any: ...
def tensor_core_gemm_int8(
    a: Any,
    b: Any,
) -> Any: ...
