# CUDA LLM Kernel Optimization

[![CI](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/llm-speed/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

English | [简体中文](README.zh-CN.md) | [Docs](https://lessup.github.io/llm-speed/)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

LLM-Speed is a CUDA kernel optimization project for LLM inference experiments, covering FlashAttention, Tensor Core GEMM, Python bindings, and verification workflows.

## Repository Overview

- CUDA kernels in `src/` and reusable primitives in `include/`
- Python bindings and packaging in `python/`, `setup.py`, and `pyproject.toml`
- Tests and benchmarks in `tests/` and `benchmarks/`
- GitHub Pages site for documentation entry, reading paths, and project updates

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .

cmake --preset release
cmake --build build/release -j$(nproc)

pytest tests/ -v
```

## Docs

- Project docs: `https://lessup.github.io/llm-speed/`
- Site home explains where to start, what to read next, and how the docs are organized
- See `CONTRIBUTING.md` for contribution workflow

## License

MIT License
