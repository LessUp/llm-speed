# CUDA LLM 内核优化

[![CI](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/llm-speed/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/llm-speed/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[English](README.md) | 简体中文 | [文档站](https://lessup.github.io/llm-speed/)

![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

面向 LLM 推理的高性能 CUDA 内核库，提供 FlashAttention、Tensor Core GEMM 和 PyTorch 绑定。

## 特性

### Attention 内核
- **Naive Attention**: 基准实现，O(N²) 显存复杂度
- **Tiled Attention**: 共享内存优化，分块计算
- **FlashAttention**: O(N) 显存复杂度，在线 Softmax 算法
  - 支持因果掩码，适用于自回归生成
  - 双缓冲流水线，计算与访存重叠

### GEMM 内核
- **高性能 GEMM**: 寄存器分块，三级分块策略
- **Tensor Core GEMM**: 使用 WMMA 硬件加速矩阵乘法
  - FP16 输入，FP32 累加
  - INT8 量化 GEMM（需要 Turing+ SM≥7.2）
- **矩阵布局支持**: NN、NT、TN、TT 转置组合

### 技术亮点
- 共享内存填充消除 Bank 冲突
- Warp 级原语高效归约
- 双缓冲流水线隐藏延迟
- Ampere+ 架构异步拷贝支持
- 完善的输入验证与错误处理

## 环境要求

| 组件 | 版本 |
|------|------|
| CUDA | 11.0+ |
| Python | 3.8+ |
| PyTorch | 2.0+ |
| GPU | SM 7.0+ (Volta) |

### 支持的 GPU 架构

| 架构 | SM 版本 | Tensor Core |
|------|---------|-------------|
| Volta | SM 7.0 | FP16 |
| Turing | SM 7.5 | FP16, INT8 |
| Ampere | SM 8.0, 8.6 | FP16, BF16, INT8, TF32 |
| Ada Lovelace | SM 8.9 | FP16, BF16, INT8, FP8 |
| Hopper | SM 9.0 | FP16, BF16, INT8, FP8 |

## 安装

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/LessUp/llm-speed.git
cd llm-speed

# 安装依赖
pip install -r requirements.txt

# 编译安装 CUDA 扩展
pip install -e .
```

### 指定 CUDA 架构编译

```bash
# 为特定 GPU 编译（如 A100 = SM 8.0）
CUDA_ARCHS="80" pip install -e .

# 为多个架构编译
CUDA_ARCHS="80;86;89" pip install -e .
```

### 替代方案：CMake 编译

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 快速开始

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
import torch
from cuda_llm_ops import gemm, tensor_core_gemm

# 标准 GEMM
a = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
b = torch.randn(512, 1024, device='cuda', dtype=torch.float16)
c = gemm(a, b)

# 带缩放
c = gemm(a, b, alpha=2.0, beta=0.5)

# Tensor Core GEMM（FP16 输入，FP32 输出）
c = tensor_core_gemm(a, b)
```

## API 参考

### Attention 函数

| 函数 | 描述 | 显存 |
|------|------|------|
| `naive_attention(q, k, v, scale=0.0)` | 基准实现 | O(N²) |
| `tiled_attention(q, k, v, scale=0.0)` | 共享内存分块 | O(N²) |
| `flash_attention(q, k, v, scale=0.0, is_causal=False)` | 在线 Softmax | O(N) |

**参数说明：**
- `q, k, v`: 输入张量 `[batch, heads, seq_len, head_dim]`
- `scale`: Attention 缩放因子（默认：`1/√head_dim`）
- `is_causal`: 启用因果掩码（仅 FlashAttention）

### GEMM 函数

| 函数 | 描述 | 精度 |
|------|------|------|
| `gemm(a, b, alpha=1.0, beta=0.0, trans_a=False, trans_b=False)` | 高性能 GEMM | FP16/FP32 |
| `tensor_core_gemm(a, b, alpha=1.0, beta=0.0)` | Tensor Core 加速 | FP16→FP32 |
| `tensor_core_gemm_int8(a, b)` | INT8 量化 GEMM | INT8→INT32 |

## 测试

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

## 性能测试

```bash
# Attention 性能测试
python benchmarks/benchmark_attention.py --seq-lengths 512 1024 2048 4096

# GEMM 性能测试
python benchmarks/benchmark_gemm.py --sizes 1024x1024x1024 2048x2048x2048

# 导出结果到 JSON
python benchmarks/benchmark_attention.py --output results.json
```

## 项目结构

```
llm-speed/
├── src/                    # CUDA 内核实现
│   ├── naive_attention.cu  # 基准 Attention
│   ├── tiled_attention.cu  # 分块优化
│   ├── flash_attention.cu  # FlashAttention（O(N) 显存）
│   ├── tensor_core_gemm.cu # Tensor Core GEMM
│   └── hgemm_kernel.cu     # 高性能 GEMM
├── include/                # 头文件原语
│   ├── common.cuh          # 核心类型和工具
│   ├── online_softmax.cuh  # 在线 Softmax 算法
│   ├── warp_primitives.cuh # Warp 级操作
│   ├── shared_memory.cuh   # 共享内存管理
│   └── pipeline.cuh        # 流水线工具
├── python/                 # Python 绑定
│   ├── bindings.cpp        # pybind11 绑定
│   ├── __init__.py         # 模块接口
│   └── profiler.py         # 性能分析器
├── tests/                  # 测试套件
│   ├── conftest.py         # 测试配置
│   ├── test_attention.py   # Attention 测试
│   ├── test_gemm.py        # GEMM 测试
│   └── test_interface.py   # 接口测试
├── benchmarks/             # 性能测试脚本
├── docs/                   # 文档
└── changelog/              # 变更历史
```

## 文档

- [API 参考](docs/api.md) - 详细 API 文档
- [技术深潜](docs/deepwiki.md) - 技术细节详解
- [性能指南](docs/performance.md) - 优化建议
- [贡献指南](CONTRIBUTING.md) - 参与贡献

## 性能

### FlashAttention 显存使用

| 序列长度 | 标准 Attention | FlashAttention | 降低比例 |
|----------|---------------|----------------|----------|
| 1024 | 4 MB | 0.25 MB | 94% |
| 2048 | 16 MB | 0.5 MB | 97% |
| 4096 | 64 MB | 1 MB | 98% |

### GEMM 性能目标

目标：矩阵规模 ≥1024×1024 时达到 cuBLAS 性能的 90% 以上

## 贡献

详见 [CONTRIBUTING.md](CONTRIBUTING.md)：
- 开发流程
- 代码风格指南
- 测试要求
- 提交信息规范

## 许可证

MIT License - 详见 [LICENSE](LICENSE)。

## 参考资料

1. **FlashAttention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
2. **FlashAttention-2**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023
3. **CUTLASS**: NVIDIA CUTLASS - CUDA Templates for Linear Algebra Subroutines
