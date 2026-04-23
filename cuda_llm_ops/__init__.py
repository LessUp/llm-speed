"""High-performance attention and GEMM kernels for LLM inference."""

import re
from pathlib import Path

try:
    from cuda_llm_ops._cuda_llm_ops import (
        flash_attention,
        gemm,
        naive_attention,
        tensor_core_gemm,
        tensor_core_gemm_int8,
        tiled_attention,
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
    "naive_attention",
    "tiled_attention",
    "flash_attention",
    "gemm",
    "tensor_core_gemm",
    "tensor_core_gemm_int8",
]


def _read_local_version() -> str:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():
        return "0.3.0"

    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.MULTILINE)
    return match.group(1) if match else "0.3.0"


try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("cuda_llm_ops")
except Exception:
    __version__ = _read_local_version()
