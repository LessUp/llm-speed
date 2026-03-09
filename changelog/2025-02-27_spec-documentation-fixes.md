# Spec 文档完善与修复

**日期**: 2025-02-27
**范围**: `.kiro/specs/cuda-llm-kernel-optimization/` 下的 design.md, requirements.md, tasks.md

## 变更摘要

对比实际代码与 spec 文档，修正了所有不一致之处，补充了缺失信息。

## design.md (v1.0 → v1.1)

### 接口签名修正
- **Naive Attention**: `naive_attention_kernel` → `naive_attention_simple_kernel`，添加 `__restrict__` 限定符
- **Tiled Attention**: 移除不存在的 `BLOCK_K` 模板参数，改为 `<T, BLOCK_M, BLOCK_N>`
- **FlashAttention**: 移除 `float* L` 参数（backward 未实现）；更新共享内存布局描述（含 row_max/row_sum/rescale）
- **Tensor Core GEMM**: 补充 tiled 版和 INT8 版 kernel 签名；修正 tiled 版 `b_frag` 布局为 `row_major`
- **HGEMM**: `hgemm_kernel` → `hgemm_register_tiled_kernel`；添加 `trans_a`/`trans_b` 参数和 double buffering 描述

### 结构体与 API 修正
- **OnlineSoftmaxState**: 移除不存在的 `float* output` 字段
- **PipelineScheduler**: 完全重写，匹配实际 API（`set_buffer`/`advance_load`/`advance_compute` 等）；添加 `DoubleBuffer`、async copy helpers、`software_pipeline` 模板
- **Warp Primitives**: 补充 `warp_reduce_min`、`block_reduce_max`、`warp_broadcast`、`warp_shuffle_xor`
- **CUDA_CHECK**: 更新为包含 `__FILE__` 和 `__LINE__` 的实际版本

### Python 接口重写
- 移除不存在的 `torch.autograd.Function` 类和独立 `.py` 文件描述
- 改为描述实际的 pybind11 C++ 绑定 (`bindings.cpp` → `cuda_llm_ops` 模块)
- 错误处理从 Python 异常类改为 `TORCH_CHECK` 宏描述

### 其他
- 架构图: "Softmax Kernels" → "Warp Primitives"
- 新增 **已知限制** 章节（无 backward、BF16 未实现、INT8 无 Python 绑定、流水线未集成到 FlashAttention 等）

## requirements.md (v1.0 → v1.1)

- 项目范围: 标注 BF16/INT8 的实际实现状态（脚注）
- 接口描述: 添加 "通过 pybind11 C++ 绑定"
- 需求追溯矩阵: 新增 **实现状态** 列；添加 `warp_primitives.cuh` 文件引用；拆分流水线需求 6.1-6.3 和 6.4-6.5 分别标注状态

## tasks.md

### 任务描述修正
- Task 2.4: `python/naive_attention.py` → `python/bindings.cpp` 中的包装函数
- Task 5.5: 移除 `torch.autograd.Function` 描述，改为 pybind11 注册
- Task 9.5: `python/gemm.py` → `python/bindings.cpp` 中的包装函数

### 状态修正
- Task 11 (流水线优化): `[x]` → `[ ]`（因 11.2 未完成）
- Task 11.2 (流水线集成到 FlashAttention): `[x]` → `[ ]`（flash_attention.cu 未使用 pipeline.cuh）
- Task 14 (Final Checkpoint): `[x]` → `[ ]`（存在未完成任务）

### 新增待办任务 (Backlog)
- Task 15: INT8 GEMM Python 绑定
- Task 16: BF16 精度支持
- Task 17: FlashAttention Backward Pass

### 表格更新
- 实现状态总结表: 添加备注列，标注各模块实际完成情况
- 依赖关系图: 添加 Backlog 任务依赖关系
