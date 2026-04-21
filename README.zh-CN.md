# ⚡ LLM-Speed

[![CI](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/llm-speed/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](https://github.com/LessUp/llm-speed/releases)

[English](README.md) | 简体中文 | [文档站](https://lessup.github.io/llm-speed/)

![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

面向 LLM 推理的高性能 CUDA 内核库，提供 FlashAttention、Tensor Core GEMM 和 PyTorch 绑定。

> 🚀 **v0.3.0 新特性**: 完整双语文档（中英文）、专业化文档结构、全面的快速入门指南。

---

## ✨ 特性

### Attention 内核
- **⚡ FlashAttention**: O(N) 显存复杂度的在线 Softmax 算法
  - 支持因果掩码，适用于自回归生成
  - 双缓冲流水线，计算与访存重叠
  - 相比标准 Attention 降低高达 98% 显存
- **🔄 Tiled Attention**: 共享内存分块优化
- **📊 Naive Attention**: 用于正确性验证的基准实现

### GEMM 内核
- **🎯 高性能 GEMM**: 寄存器分块，三级分块策略
- **🔢 Tensor Core GEMM**: 使用 WMMA 硬件加速矩阵乘法
  - FP16 输入，FP32 累加
  - INT8 量化 GEMM（需要 Turing+ SM≥7.2）
  - 目标：cuBLAS 性能的 ≥90%
- **📐 矩阵布局支持**: NN、NT、TN、TT 转置组合

### 技术亮点
- 🏦 共享内存填充消除 Bank 冲突
- ⚙️ Warp 级原语高效归约
- 🔄 双缓冲流水线隐藏延迟
- 📥 Ampere+ 架构异步拷贝支持
- ✅ 完善的输入验证与错误处理

---

## 📋 环境要求

| 组件 | 版本 |
|------|------|
| CUDA | 11.0+ |
| Python | 3.8+ |
| PyTorch | 2.0+ |
| GPU | SM 7.0+ (Volta) |

### 支持的 GPU 架构

| 架构 | SM 版本 | Tensor Core | 备注 |
|------|---------|-------------|------|
| Volta | SM 7.0 | FP16 | |
| Turing | SM 7.5 | FP16, INT8 | |
| Ampere | SM 8.0, 8.6 | FP16, INT8, TF32 | BF16: 计划中 |
| Ada Lovelace | SM 8.9 | FP16, INT8 | BF16, FP8: 计划中 |
| Hopper | SM 9.0 | FP16, INT8 | BF16, FP8: 计划中 |

> **注意**: BF16 和 FP8 支持是计划中的功能。当前支持的精度为 FP16、FP32 和 INT8。

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/LessUp/llm-speed.git
cd llm-speed

# 安装依赖
pip install -r requirements.txt

# 编译安装 CUDA 扩展
pip install -e .
```

### FlashAttention

```python
import torch
from cuda_llm_ops import flash_attention

# 创建输入张量 [batch, heads, seq_len, head_dim]
batch, heads, seq_len, head_dim = 2, 8, 512, 64
q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

# 计算 attention
output = flash_attention(q, k, v)

# 使用因果掩码（用于自回归模型）
output_causal = flash_attention(q, k, v, is_causal=True)
```

### GEMM

```python
from cuda_llm_ops import gemm, tensor_core_gemm

# 标准 GEMM
a = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
b = torch.randn(512, 1024, device='cuda', dtype=torch.float16)
c = gemm(a, b)

# Tensor Core GEMM（FP16 输入，FP32 输出）
c = tensor_core_gemm(a, b)
```

---

## 📚 文档

### 英文文档

| 文档 | 描述 |
|----------|-------------|
| [Quick Start](docs/setup/quickstart-en.md) | 5 分钟快速上手 |
| [API Reference](docs/api/api-en.md) | 完整 API 文档 |
| [Architecture](docs/architecture/architecture-en.md) | 技术深度解析 |
| [Performance Guide](docs/tutorials/performance-en.md) | 优化与调优 |
| [Troubleshooting](docs/tutorials/troubleshooting-en.md) | 常见问题与解决方案 |

### 中文文档

| 文档 | 描述 |
|----------|-------------|
| [快速入门](docs/setup/quickstart-zh.md) | 5 分钟快速上手 |
| [API 参考](docs/api/api-zh.md) | 完整 API 文档 |
| [架构设计](docs/architecture/architecture-zh.md) | 技术深度解析 |
| [性能指南](docs/tutorials/performance-zh.md) | 优化与调优 |
| [故障排除](docs/tutorials/troubleshooting-zh.md) | 常见问题与解决方案 |

---

## 📊 性能

### 显存效率

| 实现 | 显存复杂度 | 4K 序列 | 16K 序列 |
|----------------|-------------------|-------------|--------------|
| 标准 Attention | O(N²) | 256 MB | 4 GB |
| FlashAttention | O(N) | 4 MB | 16 MB |

### 相比 PyTorch SDPA 的加速（A100, FP16）

| 序列长度 | 加速比 | 显存节省 |
|-----------------|---------|--------------|
| 512 | 1.2x | 94% |
| 1024 | 1.4x | 97% |
| 2048 | 1.6x | 98% |
| 4096 | 1.8x | 98% |
| 8192 | 2.1x | 98% |

---

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定类别测试
pytest tests/ -v -m cuda        # CUDA 测试
pytest tests/ -v -m property    # 属性测试
pytest tests/ -v -m "not cuda"  # CPU 安全测试

# 运行特定测试文件
pytest tests/test_attention.py -v
```

---

## 📈 性能测试

```bash
# Attention 性能测试
python benchmarks/benchmark_attention.py --seq-lengths 512 1024 2048 4096

# GEMM 性能测试
python benchmarks/benchmark_gemm.py --sizes 1024x1024x1024 2048x2048x2048

# 导出结果到 JSON
python benchmarks/benchmark_attention.py --output results.json
```

---

## 🏗️ 项目结构

```
llm-speed/
├── cuda_llm_ops/           # Python 包（绑定、分析器）
├── src/                    # CUDA 内核实现
├── include/                # 头文件原语
├── tests/                  # 测试套件
├── benchmarks/             # 性能测试脚本
├── docs/                   # 文档（中英文）
│   ├── setup/              # 安装指南
│   ├── api/                # API 参考
│   ├── architecture/       # 架构文档
│   └── tutorials/          # 教程
├── specs/                  # 规格文档 (SDD)
└── .github/workflows/      # CI/CD 流水线
```

---

## 🤝 贡献

详见 [CONTRIBUTING.md](CONTRIBUTING.md)：
- 开发流程
- 代码风格指南
- 测试要求
- 提交信息规范

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)。

---

## 📖 参考资料

1. **FlashAttention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
2. **FlashAttention-2**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023
3. **CUTLASS**: NVIDIA CUTLASS - CUDA Templates for Linear Algebra Subroutines

---

## 🔗 链接

- [文档站点](https://lessup.github.io/llm-speed/)
- [GitHub 发布](https://github.com/LessUp/llm-speed/releases)
- [变更日志](CHANGELOG.md)
- [Issues](https://github.com/LessUp/llm-speed/issues)
- [Discussions](https://github.com/LessUp/llm-speed/discussions)
