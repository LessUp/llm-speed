"""
Setup script for CUDA LLM Kernel Optimization package.
"""

import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# CUDA architectures to compile for
CUDA_ARCHS = os.environ.get('CUDA_ARCHS', '70;75;80;86;89;90')

# Source files
cuda_sources = [
    'src/naive_attention.cu',
    'src/tiled_attention.cu',
    'src/flash_attention.cu',
    'src/tensor_core_gemm.cu',
    'src/hgemm_kernel.cu',
    'python/bindings.cpp',
]

# Include directories
include_dirs = [
    'include',
]

# Compiler flags
extra_compile_args = {
    'cxx': ['-O3', '-std=c++17'],
    'nvcc': [
        '-O3',
        '--use_fast_math',
        '-std=c++17',
        '-Xcompiler', '-fPIC',
    ]
}

# Add architecture flags
for arch in CUDA_ARCHS.split(';'):
    extra_compile_args['nvcc'].extend([
        f'-gencode=arch=compute_{arch},code=sm_{arch}',
    ])

setup(
    name='cuda_llm_ops',
    version='0.1.0',
    description='High-performance CUDA kernels for LLM inference',
    author='CUDA LLM Kernel Optimization',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='cuda_llm_ops',
            sources=cuda_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=2.0.0',
        'numpy',
    ],
    extras_require={
        'test': [
            'pytest',
            'hypothesis',
        ],
        'benchmark': [
            'matplotlib',
            'pandas',
        ],
    },
    python_requires='>=3.8',
)
