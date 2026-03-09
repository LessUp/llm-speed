---
layout: default
title: LLM-Speed
---

# LLM-Speed

CUDA LLM Kernel Optimization — 高性能 LLM 推理算子库，含 FlashAttention (online softmax)、FP16/INT8 GEMM with Tensor Core。

## 核心特性

- **FlashAttention** — Online softmax 实现，支持因果遮罩
- **FP16 HGEMM** — Tensor Core 加速半精度矩阵乘法
- **INT8 GEMM** — SM 75+ Tensor Core 量化矩阵乘
- **Warp Primitives** — 高效 warp-level reduction / scan
- **共享内存优化** — Bank conflict-free 访问模式
- **Python 绑定** — 通过 pybind11 提供 Python 接口

## 算子实现

| Kernel | 关键技术 | 架构要求 |
|--------|---------|---------|
| Naive Attention | 共享内存 QK^T | SM 70+ |
| Tiled Attention | 分块计算 + 流式 softmax | SM 70+ |
| Flash Attention | Online softmax + 因果遮罩 | SM 70+ |
| HGEMM | WMMA Tensor Core (FP16→FP32) | SM 70+ |
| Tensor Core GEMM | INT8/FP16 混合精度 | SM 75+ |

## 快速开始

```bash
# CMake 构建
cmake --preset release
cmake --build build/release -j$(nproc)

# Python 安装
pip install -e .

# 运行测试
pytest tests/
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | CUDA C++17, Python |
| 构建 | CMake 3.18+, setup.py (CUDAExtension) |
| 绑定 | pybind11 v2.11.1 |
| GPU | SM 70+ (Volta → Hopper) |
| 测试 | pytest + Hypothesis |

## 链接

- [GitHub 仓库](https://github.com/LessUp/llm-speed)
- [README](README.md)
