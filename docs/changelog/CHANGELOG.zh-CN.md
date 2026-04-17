# 变更日志

本项目所有重要的变更都将记录在此文件中。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [未发布]

### 新增
- 完整的中英文双语文档体系
- 新增的文档结构：快速入门、架构设计、故障排除指南
- 专业的文档分类和导航

### 变更
- 重构文档架构，分为 `docs/en/` 和 `docs/zh-CN/` 目录
- 优化 API 参考文档，提供更多使用示例
- 改进性能调优指南的专业性

---

## [0.3.0] - 2026-04-16

### 新增
- **双语文档体系**：完整的中英文文档支持
- **快速入门指南**：帮助新用户快速上手
- **故障排除指南**：系统性解决常见问题
- **架构设计文档**：深入的技术细节解析

### 变更
- 重构 `docs/` 目录为双语结构
- 优化 API 参考文档的组织方式
- 改进 changelog 的专业性

### 文档
- 新增 `docs/en/` 目录（英文文档）
- 新增 `docs/zh-CN/` 目录（中文文档）
- 更新 README 文档结构

---

## [0.2.0] - 2026-04-16

### 新增
- CPU 安全的 CI 工作流，使用 Python 语法验证
- GitHub Pages 部署同时支持 `master` 和 `main` 分支
- 基于路径的 Pages 工作流过滤
- API 参考文档（`docs/api.md`）
- 性能调优指南（`docs/performance.md`）

### 变更
- 文档架构重构：README 作为仓库入口，index.md 作为文档首页
- DeepWiki 指南明确为主要使用文档
- CI 工作流简化为 Python 代码检查和 CPU 安全冒烟测试
- 移除 `pytest ... || true` 回退逻辑
- 使用 ruff 格式化所有 Python 文件

### 修复
- Pages 工作流现在正确地在 `master` 分支上触发
- Python 代码格式化问题（ruff format）

---

## [0.1.2] - 2026-03-10

### 变更
- CI 工作流权限标准化（`contents: read`）和并发配置
- Pages 工作流添加 `actions/configure-pages@v5` 步骤
- Pages 工作流添加 `paths` 触发器过滤

---

## [0.1.1] - 2025-02-27

### 新增
- FlashAttention 双缓冲流水线实现
  - K/V tiles 使用交替缓冲区实现计算/加载重叠
  - 因果掩码早退优化，当 tile 超出因果窗口时退出
- INT8 Tensor Core GEMM Python 绑定（`tensor_core_gemm_int8`）
  - INT8 输入，INT32 累加输出
  - 运行时 SM 版本检查（需要 Turing+ SM≥7.2）
  - 使用 Hypothesis 的属性测试
- `pyproject.toml` 用于 pytest 标记配置（cuda, slow, property）
- 性能测试脚本 JSON 导出支持（`--output` 标志）
- 性能测试中的 GPU 峰值显存跟踪

### 变更
- FlashAttention 线程利用率从 25% 提升至 100%
  - 分为两个阶段：Softmax 状态更新（轻量）和输出更新（重量）
  - 阶段 2 所有线程协作以提高并行性
- 移除未使用的静态模板 FlashAttention 内核（16KB 寄存器溢出）
- 所有内核消除共享内存 Bank 冲突：
  - Attention 内核 `+1` 填充
  - Tensor Core GEMM `+8` half 填充
- HGEMM 内核增强双缓冲和 Bank 冲突填充

### 修复
- **[关键]** `tiled_attention.cu`：当 `head_dim > 64` 时的共享内存越界访问
  - 从固定 `BLOCK_K=64` 改为基于实际 `head_dim` 的动态共享内存
- **[关键]** `tensor_core_gemm.cu`：INT8 主机包装器始终不可用
  - 从 `#if __CUDA_ARCH__` 改为运行时 SM 版本检查
- **[关键]** `naive_attention.cu`：当 `block_sum` 为零时除零风险
  - 除法前添加保护检查
- GEMM 索引计算中的整数溢出（改为 `int64_t`）
- FlashAttention 和在线 Softmax 中的除零保护
- 规范文档与实际实现同步：
  - 接口签名修正
  - 流水线 API 描述重写
  - 需求矩阵中的实现状态更新

### 移除
- 死代码：未使用的 `softmax_row` 设备函数和 `naive_attention_kernel`
- 性能测试脚本中未使用的 `import time`

---

## [0.1.0] - 2025-02-13

### 新增
- 初始项目基础设施
  - MIT 许可证文件
  - CUDA/Python/IDE 的 `.gitignore`
  - 一致的代码格式化 `.editorconfig`
  - README 中的标准徽章（许可证、CUDA、C++、Python）
- 核心 CUDA 内核
  - `naive_attention.cu`：O(N²) 显存的基准 Attention
  - `tiled_attention.cu`：共享内存分块优化
  - `flash_attention.cu`：在线 Softmax 的 O(N) 显存
  - `tensor_core_gemm.cu`：基于 WMMA 的 Tensor Core GEMM
  - `hgemm_kernel.cu`：带寄存器分块的高性能 GEMM
- 头文件原语库
  - `common.cuh`：核心类型（`AttentionConfig`、`GemmConfig`、`KernelMetrics`），CUDA_CHECK 宏
  - `online_softmax.cuh`：FlashAttention 的在线 Softmax 算法
  - `warp_primitives.cuh`：Warp 级操作（reduce_sum、reduce_max、broadcast）
  - `shared_memory.cuh`：共享内存管理，填充工具
  - `pipeline.cuh`：双缓冲，异步拷贝（Ampere+），软件流水线
- 通过 pybind11 的 Python 绑定
  - `naive_attention`、`tiled_attention`、`flash_attention`
  - `gemm`、`tensor_core_gemm`
- 使用 pytest 和 Hypothesis 的测试套件
- Attention 和 GEMM 的性能测试脚本

---

## 版本历史摘要

| 版本 | 日期 | 亮点 |
|---------|------|------------|
| 未发布 | - | 文档重构、API 参考、性能指南 |
| 0.3.0 | 2026-04-16 | 双语文档体系、专业化文档 |
| 0.2.0 | 2026-04-16 | CI/CD 修复、文档架构、Python 格式化 |
| 0.1.2 | 2026-03-10 | 工作流深度标准化 |
| 0.1.1 | 2025-02-27 | 双缓冲、INT8 绑定、关键错误修复 |
| 0.1.0 | 2025-02-13 | 初始发布 |

---

## 迁移指南

### 升级到 0.3.0

无破坏性变更。文档 URL 保持稳定。

### 升级到 0.2.0

无破坏性变更。文档 URL 保持稳定。

### 升级到 0.1.1

**INT8 GEMM 用户**：新的 `tensor_core_gemm_int8` 函数需要 Turing+ GPU。检查 SM 版本：

```python
import torch
capability = torch.cuda.get_device_capability()
if capability[0] >= 7 and capability[1] >= 2:
    c = tensor_core_gemm_int8(a_int8, b_int8)
else:
    print("INT8 Tensor Core 需要 Turing+（SM 7.2+）")
```

**Attention 用户**：所有 Attention 函数现在有正确的除零保护。无需代码变更。

---

[0.1.0]: https://github.com/LessUp/llm-speed/releases/tag/v0.1.0
[0.1.1]: https://github.com/LessUp/llm-speed/compare/v0.1.0...v0.1.1
[0.1.2]: https://github.com/LessUp/llm-speed/compare/v0.1.1...v0.1.2
[0.2.0]: https://github.com/LessUp/llm-speed/compare/v0.1.2...v0.2.0
[0.3.0]: https://github.com/LessUp/llm-speed/compare/v0.2.0...v0.3.0
[未发布]: https://github.com/LessUp/llm-speed/compare/v0.3.0...HEAD
