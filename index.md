---
layout: default
title: LLM-Speed
description: CUDA LLM 内核优化项目的文档入口：项目定位、阅读路径与核心页面导航
---

# LLM-Speed

[![GitHub Pages](https://github.com/LessUp/llm-speed/actions/workflows/pages.yml/badge.svg)](https://github.com/LessUp/llm-speed/actions/workflows/pages.yml)
[![CI](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

LLM-Speed 面向“理解并验证 LLM 推理算子如何从朴素实现逐步优化到高性能版本”的学习与工程实践场景，覆盖 FlashAttention、Tensor Core GEMM、Python 绑定和属性测试。

## 项目定位

这是一个把 CUDA 内核实验、Python 集成和验证流程放在同一仓库里的工程化学习项目。仓库 `README` 只保留最小构建入口，这个页面负责说明项目适合谁、先看什么以及重要文档在哪。

## 适合谁

- 想系统理解 FlashAttention、共享内存分块和 Tensor Core GEMM 的开发者
- 想参考 CUDA C++ 与 pybind11 绑定协同组织方式的工程师
- 需要快速定位测试、贡献流程和历史变更记录的维护者

## 从哪里开始

1. 先看 [README](README.md)，完成依赖安装、构建与测试。
2. 再看 [DeepWiki](docs/deepwiki.md)，理解核心 kernel、头文件原语与优化思路。
3. 需要参与协作或追踪演进时，继续查看 [CONTRIBUTING](CONTRIBUTING.md) 与 [更新日志](changelog/)。

## 推荐阅读路径

### 我只想先编译并跑测试

- [README](README.md)
- [CONTRIBUTING](CONTRIBUTING.md)

### 我想先理解优化路线

- [DeepWiki](docs/deepwiki.md)
- `src/`
- `include/`

### 我准备继续维护

- [CONTRIBUTING](CONTRIBUTING.md)
- [更新日志](changelog/)
- [GitHub 仓库](https://github.com/LessUp/llm-speed)

## 核心入口

| 类别 | 页面 | 说明 |
|------|------|------|
| 概览 | [README](README.md) | 仓库定位、最小构建命令与文档链接 |
| 快速开始 | [README](README.md) | 安装依赖、构建扩展与运行测试 |
| 使用指南 | [DeepWiki](docs/deepwiki.md) | 核心 kernel、原语与优化策略说明 |
| 开发指南 | [CONTRIBUTING](CONTRIBUTING.md) | 提交流程、代码规范与测试要求 |
| 归档 | [更新日志](changelog/) | 工作流、文档与实现迭代记录 |
| 外部链接 | [GitHub 仓库](https://github.com/LessUp/llm-speed) | 源码、Issue 与协作入口 |
