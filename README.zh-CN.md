# CUDA LLM Kernel Optimization

[![CI](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/llm-speed/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[English](README.md) | 简体中文 | [文档站](https://lessup.github.io/llm-speed/)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

LLM-Speed 是一个面向 LLM 推理实验的 CUDA 算子优化项目，覆盖 FlashAttention、Tensor Core GEMM、Python 绑定与验证流程。

## 仓库概览

- `src/` 中是 CUDA kernel 实现，`include/` 中是可复用原语与工具头文件
- `python/`、`setup.py`、`pyproject.toml` 负责 Python 绑定与打包
- `tests/` 与 `benchmarks/` 提供正确性验证与性能评估入口
- GitHub Pages 文档站负责文档导读、阅读路径与项目更新说明

## 快速开始

```bash
pip install -r requirements.txt
pip install -e .

cmake --preset release
cmake --build build/release -j$(nproc)

pytest tests/ -v
```

## 文档

- 项目文档：`https://lessup.github.io/llm-speed/`
- 站点首页说明从哪里开始阅读、接下来该看什么，以及文档如何组织
- 参与协作请查看 `CONTRIBUTING.md`

## 许可

MIT License
