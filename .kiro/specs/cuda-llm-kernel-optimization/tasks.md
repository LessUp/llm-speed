# Implementation Plan: CUDA LLM Kernel Optimization

## Overview

本实现计划将设计文档分解为可执行的编码任务，从基础设施搭建开始，逐步实现 Naive Attention、Tiled Attention、FlashAttention 和高性能 GEMM kernel。每个阶段都包含正确性验证和属性测试。

## Tasks

- [x] 1. 项目基础设施搭建
  - [x] 1.1 创建项目目录结构和 CMake 构建系统
    - 创建 `src/`, `include/`, `python/`, `tests/` 目录
    - 配置 CMakeLists.txt 支持 CUDA 编译
    - 配置 pybind11 用于 Python 绑定
    - _Requirements: 8.1, 8.2_

  - [x] 1.2 实现通用工具函数和类型定义
    - 创建 `include/common.cuh` 定义数据类型和配置结构
    - 实现 CUDA 错误检查宏
    - 定义 MatrixLayout、AttentionConfig、GemmConfig 结构
    - _Requirements: 8.3_

  - [x] 1.3 搭建测试框架
    - 配置 pytest 和 Hypothesis
    - 创建测试辅助函数（随机张量生成、误差计算）
    - _Requirements: 7.3_

- [x] 2. Naive Attention 实现
  - [x] 2.1 实现 Naive Attention Kernel
    - 创建 `src/naive_attention.cu`
    - 实现 Q*K^T 矩阵乘法
    - 实现 Softmax 归一化
    - 实现完整的 Softmax(Q*K^T)*V 计算
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [x] 2.2 编写 Attention 正确性属性测试
    - **Property 1: Attention 计算正确性**
    - **Validates: Requirements 1.1, 1.3, 1.4, 1.5**

  - [x] 2.3 编写 Softmax 不变量属性测试
    - **Property 2: Softmax 数学不变量**
    - **Validates: Requirements 1.2**

  - [x] 2.4 实现 Python 绑定 (Naive Attention)
    - 创建 `python/naive_attention.py`
    - 使用 pybind11 绑定 CUDA kernel
    - _Requirements: 8.1, 8.4_

- [x] 3. Checkpoint - Naive Attention 验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 4. Tiled Attention 实现
  - [x] 4.1 实现共享内存管理器
    - 创建 `include/shared_memory.cuh`
    - 实现 tile 划分逻辑
    - 实现合并访存加载函数
    - 实现 bank conflict 消除（padding）
    - _Requirements: 2.1, 2.2, 2.4_

  - [x] 4.2 实现 Tiled Attention Kernel
    - 创建 `src/tiled_attention.cu`
    - 使用共享内存进行分块计算
    - 实现 tile 内矩阵乘法
    - _Requirements: 2.3, 2.5_

  - [x] 4.3 编写 Tiled Attention 正确性测试
    - 验证 Tiled 实现与 Naive 实现数值一致
    - _Requirements: 1.5_

- [x] 5. FlashAttention 实现
  - [x] 5.1 实现 Online Softmax 算法
    - 创建 `include/online_softmax.cuh`
    - 实现流式最大值和指数和计算
    - 实现累积输出更新逻辑
    - _Requirements: 3.1_

  - [x] 5.2 实现 FlashAttention Forward Kernel
    - 创建 `src/flash_attention.cu`
    - 在单个 kernel 内完成完整 attention 计算
    - 实现因果掩码支持
    - _Requirements: 3.2, 3.3, 3.4, 3.5_

  - [x] 5.3 编写 FlashAttention 一致性属性测试
    - **Property 3: FlashAttention 与标准实现一致性**
    - **Validates: Requirements 3.1, 3.6**

  - [x] 5.4 编写因果掩码正确性属性测试
    - **Property 4: 因果掩码正确性**
    - **Validates: Requirements 3.5**

  - [x] 5.5 实现 Python 绑定 (FlashAttention)
    - 创建 `python/flash_attention.py`
    - 实现 torch.autograd.Function 包装
    - _Requirements: 8.1, 8.4_

- [x] 6. Checkpoint - FlashAttention 验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 7. Warp-Level Primitives 实现
  - [x] 7.1 实现 Warp Shuffle 规约函数
    - 创建 `include/warp_primitives.cuh`
    - 实现 warp_reduce_sum 和 warp_reduce_max
    - 实现 block_reduce_sum
    - _Requirements: 5.4_

- [x] 8. Tensor Core GEMM 实现
  - [x] 8.1 实现基础 Tensor Core GEMM Kernel
    - 创建 `src/tensor_core_gemm.cu`
    - 使用 WMMA API 实现 FP16 矩阵乘法
    - 实现维度对齐检查和 padding
    - _Requirements: 4.1, 4.2, 4.4_

  - [x] 8.2 编写 FP16 GEMM 正确性属性测试
    - **Property 5: FP16 GEMM 正确性**
    - **Validates: Requirements 4.2, 5.1**

  - [x] 8.3 实现 INT8 Tensor Core GEMM
    - 扩展支持 INT8 输入和 INT32 累加
    - _Requirements: 4.3_

  - [x] 8.4 编写 INT8 GEMM 正确性属性测试
    - **Property 6: INT8 GEMM 正确性**
    - **Validates: Requirements 4.3, 5.2**

- [x] 9. 高性能 GEMM 实现
  - [x] 9.1 实现 Register Tiling GEMM Kernel
    - 创建 `src/hgemm_kernel.cu`
    - 实现多级 tiling（Block -> Warp -> Thread）
    - 实现寄存器级数据复用
    - _Requirements: 5.3_

  - [x] 9.2 实现矩阵布局支持
    - 支持 NN, NT, TN, TT 四种布局
    - _Requirements: 5.6_

  - [x] 9.3 编写矩阵布局等价性属性测试
    - **Property 7: 矩阵布局等价性**
    - **Validates: Requirements 5.6**

  - [x] 9.4 编写维度对齐处理属性测试
    - **Property 8: 维度对齐处理**
    - **Validates: Requirements 4.4**

  - [x] 9.5 实现 Python 绑定 (GEMM)
    - 创建 `python/gemm.py`
    - 支持多种精度和布局选项
    - _Requirements: 8.2, 8.5_

- [x] 10. Checkpoint - GEMM 验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 11. 流水线优化
  - [x] 11.1 实现 Pipeline Scheduler
    - 创建 `include/pipeline.cuh`
    - 实现 Double Buffering
    - 实现可配置的流水线深度（2-4 级）
    - _Requirements: 6.1, 6.2, 6.4_

  - [x] 11.2 将流水线集成到 FlashAttention
    - 修改 `src/flash_attention.cu` 使用流水线
    - 实现数据预取与计算重叠
    - _Requirements: 6.3, 6.5_

  - [x] 11.3 编写流水线配置正确性属性测试
    - **Property 9: 流水线深度配置正确性**
    - **Validates: Requirements 6.4**

- [x] 12. Python 接口完善
  - [x] 12.1 实现输入验证和错误处理
    - 添加维度检查、dtype 检查
    - 实现明确的错误消息
    - _Requirements: 8.3_

  - [x] 12.2 编写 Python 接口兼容性属性测试
    - **Property 10: Python 接口兼容性**
    - **Validates: Requirements 8.1, 8.2**

  - [x] 12.3 编写批量和多头支持属性测试
    - **Property 11: 批量和多头支持**
    - **Validates: Requirements 8.4**

  - [x] 12.4 编写任意形状矩阵支持属性测试
    - **Property 12: 任意形状矩阵支持**
    - **Validates: Requirements 8.5**

  - [x] 12.5 编写无效输入错误处理属性测试
    - **Property 13: 无效输入错误处理**
    - **Validates: Requirements 8.3**

- [x] 13. 性能分析工具
  - [x] 13.1 实现 Profiler 接口
    - 创建 `python/profiler.py`
    - 集成 CUDA 事件计时
    - 实现 TFLOPS 和带宽计算
    - _Requirements: 7.1, 7.2_

  - [x] 13.2 实现性能基准测试脚本
    - 创建 `benchmarks/` 目录
    - 实现与 cuBLAS 的对比测试
    - 实现瓶颈分析报告
    - _Requirements: 7.4, 7.5_

- [x] 14. Final Checkpoint - 完整验证
  - 确保所有测试通过
  - 运行性能基准测试
  - 如有问题请询问用户

## Notes

- 每个任务都引用了具体的需求条款以确保可追溯性
- Checkpoint 任务用于阶段性验证
- 属性测试验证通用正确性属性，单元测试验证边界情况
- 所有测试任务都是必须完成的

## 实现状态总结

| 模块 | 状态 | 关键文件 |
|------|------|----------|
| 项目基础设施 | ✅ 完成 | CMakeLists.txt, common.cuh |
| Naive Attention | ✅ 完成 | naive_attention.cu |
| Tiled Attention | ✅ 完成 | tiled_attention.cu, shared_memory.cuh |
| FlashAttention | ✅ 完成 | flash_attention.cu, online_softmax.cuh |
| Tensor Core GEMM | ✅ 完成 | tensor_core_gemm.cu |
| 高性能 GEMM | ✅ 完成 | hgemm_kernel.cu |
| 流水线优化 | ✅ 完成 | pipeline.cuh |
| Python 接口 | ✅ 完成 | bindings.cpp, __init__.py |
| 性能分析工具 | ✅ 完成 | profiler.py, benchmarks/ |

## 依赖关系

```
1. 基础设施 ─┬─> 2. Naive Attention ─> 3. Checkpoint
             │
             ├─> 4. Tiled Attention ─┬─> 5. FlashAttention ─> 6. Checkpoint
             │                       │
             │                       └─> 11. 流水线优化
             │
             ├─> 7. Warp Primitives ─> 8. Tensor Core GEMM ─> 9. 高性能 GEMM ─> 10. Checkpoint
             │
             └─> 12. Python 接口 ─> 13. 性能分析 ─> 14. Final Checkpoint
```
