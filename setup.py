"""
Setup script for CUDA extension compilation only.

Package metadata, dependencies, and optional-dependencies are defined exclusively
in pyproject.toml. Setuptools will automatically read them from there.
This script handles only CUDA extension compilation configuration.
"""

import os
import platform

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _build_cuda_extensions():
    """Build CUDA extensions with configurable architecture support.

    Returns empty list if CUDA_HOME is not set (for pip install without CUDA).
    Set CUDA_HOME to enable CUDA extension compilation.
    """
    if not os.environ.get('CUDA_HOME'):
        return []

    cuda_archs = os.environ.get('CUDA_ARCHS', '70;75;80;86;89;90')

    cuda_sources = [
        'src/naive_attention.cu',
        'src/tiled_attention.cu',
        'src/flash_attention.cu',
        'src/tensor_core_gemm.cu',
        'src/hgemm_kernel.cu',
        'cuda_llm_ops/bindings.cpp',
    ]

    include_dirs = ['include']

    # Platform-aware compiler flags
    if platform.system() == 'Windows':
        extra_compile_args = {
            'cxx': ['/O2', '/std:c++17'],
            'nvcc': ['-O3', '--use_fast_math', '-std=c++17']
        }
    else:
        extra_compile_args = {
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': ['-O3', '--use_fast_math', '-std=c++17', '-Xcompiler', '-fPIC']
        }

    # Add compute capability flags
    for arch in cuda_archs.split(';'):
        extra_compile_args['nvcc'].extend([
            f'-gencode=arch=compute_{arch},code=sm_{arch}',
        ])

    return [
        CUDAExtension(
            name='cuda_llm_ops._cuda_llm_ops',
            sources=cuda_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    ]


setup(
    ext_modules=_build_cuda_extensions(),
    cmdclass={'build_ext': BuildExtension},
)
