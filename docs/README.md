# CUDA LLM Kernel Optimization Documentation

Welcome to the documentation for CUDA LLM Kernel Optimization - a high-performance CUDA kernel library for LLM inference.

---

## 🌐 Language Selection / 语言选择

- **[English Documentation](./en/)**
- **[中文文档](./zh-CN/)**

---

## 📚 Documentation Structure

### English Documentation

| Document | Description |
|----------|-------------|
| [Quick Start](en/quickstart.md) | Get started in 5 minutes |
| [API Reference](en/api.md) | Complete API documentation |
| [Architecture](en/architecture.md) | Technical deep dive |
| [Performance Guide](en/performance.md) | Optimization and tuning |
| [Troubleshooting](en/troubleshooting.md) | Common issues and solutions |

### 中文文档

| 文档 | 描述 |
|----------|-------------|
| [快速入门](zh-CN/quickstart.md) | 5 分钟快速上手 |
| [API 参考](zh-CN/api.md) | 完整 API 文档 |
| [架构设计](zh-CN/architecture.md) | 技术深度解析 |
| [性能指南](zh-CN/performance.md) | 优化与调优 |
| [故障排除](zh-CN/troubleshooting.md) | 常见问题与解决方案 |

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

---

## 📖 Key Features

- **FlashAttention**: O(N) memory complexity with online softmax
- **Tensor Core GEMM**: Hardware-accelerated matrix multiplication
- **Bilingual Documentation**: Complete Chinese and English support
- **Production Ready**: Comprehensive error handling and validation

---

## 🔗 Related Links

- [GitHub Repository](https://github.com/LessUp/llm-speed)
- [Change Log](../changelog/CHANGELOG.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [License](../LICENSE)

---

**Version**: 0.3.0 | **Last Updated**: April 2026
