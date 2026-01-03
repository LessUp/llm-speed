# CUDA LLM Kernel Optimization

高性能 CUDA 算子库，用于 LLM 推理优化。包含 FlashAttention 实现和高性能 GEMM kernel。

## 特性

- **FlashAttention**: Online Softmax 算法，O(N) 显存复杂度
- **高性能 GEMM**: 支持 FP16/INT8 混合精度，使用 Tensor Core 加速
- **多级优化**: Naive → Tiled → FlashAttention 渐进式优化
- **PyTorch 集成**: 提供 Python 绑定，支持 PyTorch Tensor

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 编译 CUDA 扩展
pip install -e .
```

## 使用

```python
import torch
from cuda_llm_ops import flash_attention, gemm

# Attention
q = torch.randn(1, 32, 1024, 128, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
output = flash_attention(q, k, v, is_causal=True)

# GEMM
a = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
b = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
c = gemm(a, b)
```

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行属性测试
pytest tests/ -v -m property

# 运行基准测试
python benchmarks/benchmark_attention.py
python benchmarks/benchmark_gemm.py
```

## 项目结构

```
├── include/           # CUDA 头文件
│   ├── common.cuh
│   ├── warp_primitives.cuh
│   ├── shared_memory.cuh
│   ├── online_softmax.cuh
│   └── pipeline.cuh
├── src/               # CUDA 源文件
│   ├── naive_attention.cu
│   ├── tiled_attention.cu
│   ├── flash_attention.cu
│   ├── tensor_core_gemm.cu
│   └── hgemm_kernel.cu
├── python/            # Python 绑定
│   ├── __init__.py
│   ├── bindings.cpp
│   └── profiler.py
├── tests/             # 测试文件
│   ├── conftest.py
│   ├── test_attention.py
│   ├── test_gemm.py
│   └── test_interface.py
└── benchmarks/        # 基准测试
    ├── benchmark_attention.py
    └── benchmark_gemm.py
```

## 性能目标

- GEMM: 达到 cuBLAS 90%+ 性能
- FlashAttention: O(N) 显存复杂度，避免 N×N 注意力矩阵

## 支持的 GPU 架构

- Volta (SM 7.0)
- Turing (SM 7.5)
- Ampere (SM 8.0, 8.6)
- Ada Lovelace (SM 8.9)
- Hopper (SM 9.0)
