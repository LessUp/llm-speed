# CUDA LLM Kernel Optimization Documentation

Welcome to the documentation for CUDA LLM Kernel Optimization — a high-performance CUDA kernel library for LLM inference.

---

## 🌐 Language Selection

This documentation is available in:

- **English** (primary)
- **[简体中文](README.zh-CN.md)** (Chinese translation)

---

## 📚 Documentation Structure

### Setup Guides

| Document | Language | Description |
|----------|----------|-------------|
| [Quick Start](setup/quickstart-en.md) | English | Get started in 5 minutes |
| [快速入门](setup/quickstart-zh.md) | 中文 | 5 分钟快速上手 |

### Tutorials & Guides

| Document | Language | Description |
|----------|----------|-------------|
| [Performance Guide](tutorials/performance-en.md) | English | Optimization and tuning |
| [性能指南](tutorials/performance-zh.md) | 中文 | 优化与调优 |
| [Troubleshooting](tutorials/troubleshooting-en.md) | English | Common issues and solutions |
| [故障排除](tutorials/troubleshooting-zh.md) | 中文 | 常见问题与解决方案 |

### Architecture

| Document | Language | Description |
|----------|----------|-------------|
| [Architecture Deep Dive](architecture/architecture-en.md) | English | Technical deep dive |
| [架构设计](architecture/architecture-zh.md) | 中文 | 技术深度解析 |

### API Reference

| Document | Language | Description |
|----------|----------|-------------|
| [API Reference](api/api-en.md) | English | Complete API documentation |
| [API 参考](api/api-zh.md) | 中文 | 完整 API 文档 |

---

## 📋 Specifications

This project follows **Spec-Driven Development (SDD)**. For technical specifications, see the [`/specs`](../specs/) directory:

| Spec | Description |
|------|-------------|
| [Product Requirements](../specs/product/cuda-llm-kernel-optimization.md) | Feature requirements and acceptance criteria |
| [Core Architecture RFC](../specs/rfc/0001-core-architecture.md) | Technical design and architecture |
| [Implementation Tasks RFC](../specs/rfc/0002-implementation-tasks.md) | Implementation plan and task breakdown |

---

## 🚀 Getting Started

### Quick Installation

```bash
git clone https://github.com/LessUp/llm-speed.git
cd llm-speed
pip install -r requirements.txt
pip install -e .
```

### Verify Installation

```bash
python -c "import cuda_llm_ops; print(cuda_llm_ops.__version__)"
```

### Quick Example

```python
import torch
from cuda_llm_ops import flash_attention

# Create inputs
batch, heads, seq_len, head_dim = 2, 8, 512, 64
q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

# Compute attention with O(N) memory
output = flash_attention(q, k, v)
```

---

## 🔗 Related Links

| Resource | Link |
|----------|------|
| GitHub Repository | https://github.com/LessUp/llm-speed |
| Change Log | [../CHANGELOG.md](../CHANGELOG.md) |
| Contributing Guide | [../CONTRIBUTING.md](../CONTRIBUTING.md) |
| AI Agent Instructions | [../AGENTS.md](../AGENTS.md) |
| License | [../LICENSE](../LICENSE) |

---

**Version**: 0.3.0 | **Last Updated**: April 2026
