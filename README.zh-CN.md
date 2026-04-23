# ⚡ LLM-Speed

[![CI](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml)
[![Pages](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/llm-speed/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-3776AB?logo=python)](https://python.org)
[![CUDA 11.0+](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

[English](README.md) | 简体中文 | [文档站](https://lessup.github.io/llm-speed/)

LLM-Speed 是一个聚焦于 **LLM 推理核心算子** 的 CUDA 内核仓库：提供 FlashAttention 前向实现、Tensor Core GEMM、高性能 GEMM 分块优化，以及一层轻量的 PyTorch 绑定。它不是一个庞大的框架，而是一个适合阅读、验证和做实验的内核实现集合。

## 这个仓库的价值

- **FlashAttention 前向路径**，具备 O(N) 显存行为
- **Tensor Core GEMM** 与优化过的混合精度 GEMM 基线
- **面向 PyTorch 的绑定层**，便于直接从 Python 试验
- **正确性与性能验证脚本**，方便回归和基准对比
- **清晰可读的 CUDA 优化阶梯**：naive → tiled → FlashAttention → Tensor Core

## 仓库内容

| 模块 | 关键文件 | 说明 |
| --- | --- | --- |
| Attention 内核 | `src/naive_attention.cu`, `src/tiled_attention.cu`, `src/flash_attention.cu` | 从正确性基线到显存优化前向 Attention |
| GEMM 内核 | `src/tensor_core_gemm.cu`, `src/hgemm_kernel.cu` | Tensor Core 路径与优化 GEMM |
| 共享原语 | `include/*.cuh` | 在线 softmax、共享内存工具、pipeline、warp 原语 |
| Python 包 | `cuda_llm_ops/` | pybind11 绑定、版本信息、profiler |
| 验证 | `tests/`, `benchmarks/` | CPU-safe 测试、GPU 测试、性能脚本 |

## 环境要求

| 组件 | 最低要求 |
| --- | --- |
| Python | 3.8 |
| CUDA Toolkit | 11.0 |
| PyTorch | 2.0 |
| GPU | Volta / SM70 及以上 |

当前仓库中实际覆盖的精度路径：**FP32、FP16、INT8**。

## 快速开始

```bash
git clone https://github.com/LessUp/llm-speed.git
cd llm-speed

python3 -m venv .venv
. .venv/bin/activate

pip install -U pip setuptools wheel
pip install -r requirements.txt pytest hypothesis ruff pre-commit

# 有 CUDA 环境时编译扩展
pip install -e .
```

快速检查包是否可见：

```bash
python -c "import cuda_llm_ops; print(cuda_llm_ops.__version__)"
```

运行不依赖 GPU 的仓库校验：

```bash
ruff check cuda_llm_ops/ tests/ benchmarks/
pytest tests/ -v -m "not cuda"
pre-commit run --all-files
```

## 示例

```python
import torch
from cuda_llm_ops import flash_attention, tensor_core_gemm

q = torch.randn(2, 8, 2048, 64, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

attn = flash_attention(q, k, v, is_causal=True)

a = torch.randn(1024, 512, device="cuda", dtype=torch.float16)
b = torch.randn(512, 1024, device="cuda", dtype=torch.float16)
c = tensor_core_gemm(a, b)
```

## 文档导航

- [文档总览](docs/README.md)
- [快速开始](docs/setup/quickstart-zh.md)
- [API 参考](docs/api/api-zh.md)
- [架构说明](docs/architecture/architecture-zh.md)
- [性能指南](docs/tutorials/performance-zh.md)
- [故障排除](docs/tutorials/troubleshooting-zh.md)

## 项目结构

```text
llm-speed/
├── cuda_llm_ops/        # Python 包和绑定
├── src/                 # CUDA 内核
├── include/             # CUDA 原语和辅助头文件
├── tests/               # pytest 校验
├── benchmarks/          # 基准脚本
├── docs/                # 面向用户的文档
├── openspec/            # 活跃规范与变更跟踪
├── AGENTS.md            # 共享 AI 工作约定
├── CLAUDE.md            # Claude 专用约定
└── .github/             # workflow 与 Copilot 指令
```

## 贡献方式

本项目使用 **OpenSpec** 管理变更。在修改行为或接口之前：

```bash
/opsx:propose <change-name>
/opsx:apply <change-name>
```

工作约定见 [CONTRIBUTING.md](CONTRIBUTING.md)、[AGENTS.md](AGENTS.md) 和 [CLAUDE.md](CLAUDE.md)。

## 参考资料

1. Dao 等，*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
2. Dao，*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*
3. NVIDIA CUTLASS
