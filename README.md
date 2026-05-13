# ⚡ LLM-Speed

[![CI](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-VitePress-76B900?logo=vitepress)](https://lessup.github.io/llm-speed/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-3776AB?logo=python)](https://python.org)
[![CUDA 11.0+](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://hub.docker.com/)

English | [简体中文](README.zh-CN.md) | [Docs](https://lessup.github.io/llm-speed/) | [Colab Demo](https://colab.research.google.com/github/LessUp/llm-speed/blob/main/notebooks/llm-speed-demo.ipynb)

LLM-Speed is a compact CUDA kernel playground for **LLM inference primitives**: FlashAttention forward kernels, Tensor Core GEMM, high-performance GEMM tiling, and a thin PyTorch binding layer. The goal is not to be a giant framework; it is to be a focused, inspectable implementation that is easy to benchmark, study, and extend.

## Why this repo is useful

- **FlashAttention forward path** with O(N) memory behavior
- **Tensor Core GEMM** and optimized mixed-precision GEMM baselines
- **PyTorch-facing bindings** for experimentation from Python
- **Benchmark and correctness harnesses** for iteration and regression checks
- **Readable CUDA code** that keeps the optimization ladder visible: naive → tiled → FlashAttention → Tensor Core

## What is included

| Area | Key files | Notes |
| --- | --- | --- |
| Attention kernels | `src/naive_attention.cu`, `src/tiled_attention.cu`, `src/flash_attention.cu` | Correctness baseline through memory-efficient forward attention |
| GEMM kernels | `src/tensor_core_gemm.cu`, `src/hgemm_kernel.cu` | Tensor Core path plus optimized GEMM |
| Shared primitives | `include/*.cuh` | Online softmax, shared memory helpers, pipeline, warp ops |
| Python package | `cuda_llm_ops/` | pybind11 bindings, versioned package surface, profiler utilities |
| Validation | `tests/`, `benchmarks/` | CPU-safe tests, GPU tests, performance scripts |

## Requirements

| Component | Minimum |
| --- | --- |
| Python | 3.8 |
| CUDA Toolkit | 11.0 |
| PyTorch | 2.0 |
| GPU | Volta / SM70 or newer |

Supported precision paths in the repository today: **FP32, FP16, INT8**.

## Quick start

```bash
git clone https://github.com/LessUp/llm-speed.git
cd llm-speed

python3 -m venv .venv
. .venv/bin/activate

pip install -U pip setuptools wheel
pip install -r requirements.txt pytest hypothesis ruff pre-commit

# Build the extension when CUDA is available
pip install -e .
```

### Docker (Alternative)

Use Docker for a reproducible development environment:

```bash
# Build and run with GPU support
docker compose up dev

# Run tests in container
docker compose run --rm test

# Run benchmarks
docker compose run --rm benchmark
```

Smoke-test the user-facing package surface:

```bash
python -c "import cuda_llm_ops; print(cuda_llm_ops.__version__)"
```

Run the repository checks that do not require a GPU:

```bash
ruff check cuda_llm_ops/ tests/ benchmarks/
pytest tests/ -v -m "not cuda"
pre-commit run --all-files
```

## Example

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

## Documentation map

- [Docs landing page](docs/README.md)
- [Quick start](docs/setup/quickstart-en.md)
- [API reference](docs/api/api-en.md)
- [Architecture](docs/architecture/architecture-en.md)
- [Performance guide](docs/tutorials/performance-en.md)
- [Troubleshooting](docs/tutorials/troubleshooting-en.md)

## Project structure

```text
llm-speed/
├── cuda_llm_ops/        # Python package and bindings
├── src/                 # CUDA kernels
├── include/             # CUDA helpers and primitives
├── tests/               # pytest-based validation
├── benchmarks/          # benchmark scripts
├── docs/                # user-facing docs
├── openspec/            # active specs and change tracking
├── AGENTS.md            # shared AI workflow contract
├── CLAUDE.md            # Claude-specific defaults
└── .github/             # workflows and Copilot instructions
```

## Contributing

This project is governed by **OpenSpec**. Before changing behavior or interfaces:

```bash
/opsx:propose <change-name>
/opsx:apply <change-name>
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md), and [CLAUDE.md](CLAUDE.md) for the working agreement.

## References

1. Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
2. Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*
3. NVIDIA CUTLASS
