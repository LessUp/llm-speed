# CUDA LLM Kernel Optimization

[![CI](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/llm-speed/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[English](README.md) | 简体中文
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

高性能 CUDA 算子库，用于 LLM 推理优化。包含 FlashAttention 实现和高性能 GEMM kernel。

## 特性

- **FlashAttention**: Online Softmax 算法，O(N) 显存复杂度，支持 causal mask
- **高性能 GEMM**: 支持 FP32/FP16/INT8 混合精度，Tensor Core (WMMA) 加速
- **多级优化**: Naive → Tiled (shared memory) → FlashAttention (double-buffered) 渐进式优化
- **Register Tiling GEMM**: 128×128 分块 + 寄存器级 8×8 累加 + 双缓冲流水线
- **PyTorch 集成**: pybind11 Python 绑定，直接接受 PyTorch Tensor
- **属性测试**: Hypothesis 驱动的 property-based 测试

## 安装

### Python 安装（推荐）

```bash
# 安装依赖
pip install -r requirements.txt

# 编译 CUDA 扩展
pip install -e .
```

### CMake 构建

```bash
# 使用 CMake Presets
cmake --preset default        # Debug 构建
cmake --preset release        # Release 构建

cmake --build --preset default
cmake --build --preset release
```

## 使用

```python
import torch
from cuda_llm_ops import flash_attention, gemm, tensor_core_gemm

# FlashAttention (支持 causal mask)
q = torch.randn(1, 32, 1024, 128, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
output = flash_attention(q, k, v, is_causal=True)

# 高性能 GEMM (支持转置)
a = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
b = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
c = gemm(a, b, alpha=1.0, beta=0.0, trans_a=False, trans_b=False)

# Tensor Core GEMM (FP16 → FP32)
c_fp32 = tensor_core_gemm(a, b)
```

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行属性测试
pytest tests/ -v -m property

# 运行基准测试
python benchmarks/benchmark_attention.py
python benchmarks/benchmark_gemm.py --output results.json
```

## 项目结构

```
├── include/                # CUDA 头文件（公共工具库）
│   ├── common.cuh          #   配置结构体、CUDA_CHECK、枚举、工具函数
│   ├── warp_primitives.cuh #   warp/block 归约原语 (sum, max, min)
│   ├── shared_memory.cuh   #   shared memory tile 抽象 + 向量化加载
│   ├── online_softmax.cuh  #   在线 softmax 状态机 + FlashAttention 输出更新
│   └── pipeline.cuh        #   软件流水线 + 双缓冲 + async copy 辅助
├── src/                    # CUDA 核心实现
│   ├── naive_attention.cu  #   朴素 attention (shared memory scores)
│   ├── tiled_attention.cu  #   分块 attention (online softmax, shared memory)
│   ├── flash_attention.cu  #   FlashAttention (双缓冲 KV, causal mask)
│   ├── hgemm_kernel.cu     #   寄存器分块 GEMM (双缓冲, 转置支持)
│   └── tensor_core_gemm.cu #   Tensor Core GEMM (WMMA FP16 + INT8)
├── python/                 # Python 绑定
│   ├── __init__.py         #   包入口 + fallback stubs
│   ├── bindings.cpp        #   pybind11 绑定 (PyTorch Tensor ↔ CUDA)
│   └── profiler.py         #   CUDA 性能分析器
├── tests/                  # 属性测试 (Hypothesis)
│   ├── conftest.py         #   测试配置 + fixtures + 参考实现
│   ├── test_attention.py   #   attention 正确性 + softmax 不变量
│   ├── test_gemm.py        #   GEMM 正确性 + 数值稳定性
│   └── test_interface.py   #   Python 接口 + 错误处理
├── benchmarks/             # 基准测试
│   ├── benchmark_attention.py
│   └── benchmark_gemm.py
├── CMakeLists.txt          # CMake 构建（显式源文件列表 + 目标级编译选项）
├── CMakePresets.json        # CMake Presets (default / release / ci)
├── setup.py                # PyTorch 扩展构建
├── pyproject.toml          # 项目元数据 + pytest 配置（版本唯一来源）
├── .clang-format           # 代码风格 (Google 基础 + CUDA 适配)
├── .editorconfig           # 编辑器一致性
└── .gitignore
```

## 性能目标

- **GEMM**: 达到 cuBLAS 90%+ 性能（128×128 分块 + 寄存器累加 + 双缓冲）
- **FlashAttention**: O(N) 显存复杂度，避免 N×N 注意力矩阵，支持 causal mask

## 支持的 GPU 架构

| 架构 | SM | 特性 |
|------|------|------|
| Volta | 7.0 | FP16 Tensor Core |
| Turing | 7.5 | FP16 + INT8 Tensor Core |
| Ampere | 8.0, 8.6 | TF32 + async copy |
| Ada Lovelace | 8.9 | FP8 Tensor Core |
| Hopper | 9.0 | TMA + Warp Group MMA |

## 工程质量

- **CMake 现代化**: 显式源文件列表、目标级 `target_include_directories` / `target_compile_options`、generator expressions 区分 Debug/Release
- **CMake Presets**: `default`（Debug）/ `release` / `ci` 三种预设，`cmake --preset release` 即可构建
- **代码风格**: `.clang-format` (Google 基础 + 4 空格缩进 + 100 列) + `.editorconfig`
- **CI**: GitHub Actions 工作流（Python lint + CUDA build + 测试）
- **版本管理**: `pyproject.toml` 为唯一版本来源，`setup.py` 和 `__init__.py` 自动读取
- **属性测试**: Hypothesis 驱动，覆盖注意力正确性、softmax 不变量、GEMM 数值稳定性、接口兼容性
