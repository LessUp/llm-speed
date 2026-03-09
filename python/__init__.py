"""
CUDA LLM Kernel Optimization
High-performance attention and GEMM kernels for LLM inference.
"""

try:
    from cuda_llm_ops import (
        naive_attention,
        tiled_attention,
        flash_attention,
        gemm,
        tensor_core_gemm,
        tensor_core_gemm_int8,
    )
except ImportError:
    # Fallback for when the module is not built yet
    import warnings
    warnings.warn("CUDA kernels not built. Run 'pip install -e .' to build.")
    
    def naive_attention(*args, **kwargs):
        raise NotImplementedError("CUDA kernels not built")
    
    def tiled_attention(*args, **kwargs):
        raise NotImplementedError("CUDA kernels not built")
    
    def flash_attention(*args, **kwargs):
        raise NotImplementedError("CUDA kernels not built")
    
    def gemm(*args, **kwargs):
        raise NotImplementedError("CUDA kernels not built")
    
    def tensor_core_gemm(*args, **kwargs):
        raise NotImplementedError("CUDA kernels not built")
    
    def tensor_core_gemm_int8(*args, **kwargs):
        raise NotImplementedError("CUDA kernels not built")

__all__ = [
    'naive_attention',
    'tiled_attention', 
    'flash_attention',
    'gemm',
    'tensor_core_gemm',
    'tensor_core_gemm_int8',
]

__version__ = '0.1.0'
