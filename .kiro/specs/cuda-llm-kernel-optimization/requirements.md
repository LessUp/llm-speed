# Requirements Document

## Introduction

本项目旨在实现基于 CUDA 的高性能 LLM 核心算子优化，重点复现并优化 FlashAttention 算法，以及实现支持混合精度（FP16/INT8）的高性能 GEMM kernel。目标是解决大语言模型推理中 Transformer 注意力机制和矩阵乘法的性能瓶颈，逼近或超越 cuBLAS/Vendor 库的性能。

### 项目范围

- **目标 GPU 架构**: NVIDIA Volta (SM70+), Ampere (SM80+), Hopper (SM90+)
- **支持精度**: FP32, FP16, BF16, INT8
- **核心算子**: Attention (Naive, Tiled, Flash), GEMM (Naive, Tensor Core, Register Tiling)
- **接口**: Python/PyTorch 兼容接口

### 性能目标

- GEMM: 达到 cuBLAS 90%+ 性能（矩阵规模 ≥ 1024×1024）
- FlashAttention: 显存复杂度 O(N)，相比标准实现减少 50%+ 显存占用
- 流水线优化: 相比非流水线版本提升 20%+ 性能

## Glossary

- **Attention_Kernel**: 实现 Transformer 注意力机制计算的 CUDA kernel，计算 Softmax(Q·K^T/√d)·V
- **GEMM_Kernel**: 通用矩阵乘法（General Matrix Multiply）的 CUDA kernel 实现，计算 C = α·A·B + β·C
- **FlashAttention_Engine**: 实现 Online Softmax 技巧的高效注意力计算引擎，避免 O(N²) 显存占用
- **Shared_Memory_Manager**: 管理 GPU 共享内存分块和数据传输的组件，负责 tile 划分和 bank conflict 消除
- **Tensor_Core_Accelerator**: 利用 WMMA/MMA 指令进行矩阵乘加速的组件，支持 FP16/INT8 精度
- **Pipeline_Scheduler**: 管理计算与访存流水线重叠的调度器，实现 Double/Multi-Buffering
- **Profiler**: 使用 Nsight Compute 进行性能分析的工具接口，报告 TFLOPS、带宽、占用率等指标
- **Verification_Module**: 使用 PyTorch 进行结果正确性验证的模块，支持数值误差容差检查
- **Online_Softmax**: FlashAttention 核心算法，流式计算 max 和 exp sum，避免存储完整注意力矩阵
- **Register_Tiling**: 多级分块策略中的寄存器级分块，最大化数据复用减少访存
- **Warp_Shuffle**: CUDA warp 级别的数据交换指令，用于高效规约操作

## Requirements

### Requirement 1: Naive Baseline 实现

**User Story:** 作为算子开发工程师，我需要实现基础的 Softmax + Attention 逻辑作为性能基准，以便后续优化有明确的对比目标。

#### Acceptance Criteria

1. THE Attention_Kernel SHALL 实现标准的 Q*K^T 矩阵乘法计算
2. THE Attention_Kernel SHALL 实现 Softmax 归一化操作，对注意力分数进行归一化处理
3. THE Attention_Kernel SHALL 实现 Softmax(Q*K^T)*V 的完整注意力输出计算
4. WHEN 输入序列长度为 N 时，THE Attention_Kernel SHALL 正确处理 N×N 的注意力矩阵
5. THE Verification_Module SHALL 验证 Naive 实现与 PyTorch 参考实现的数值误差在 1e-3 以内（FP32）或 1e-2 以内（FP16）

### Requirement 2: Tiling 分块优化

**User Story:** 作为算子开发工程师，我需要利用 GPU 共享内存进行数据分块，以减少全局内存访问次数，提升计算效率。

#### Acceptance Criteria

1. THE Shared_Memory_Manager SHALL 将输入矩阵划分为适合共享内存大小的 tile 块
2. WHEN 加载数据到共享内存时，THE Shared_Memory_Manager SHALL 使用合并访存（Coalesced Access）模式
3. THE Attention_Kernel SHALL 在共享内存中完成 tile 内的矩阵乘法计算
4. WHEN 发生 Bank Conflict 时，THE Shared_Memory_Manager SHALL 通过 padding 或数据重排消除冲突
5. THE Attention_Kernel SHALL 相比 Naive 实现在全局内存带宽利用率上提升至少 2 倍

### Requirement 3: FlashAttention 核心算法复现

**User Story:** 作为算子开发工程师，我需要实现 FlashAttention 的 Online Softmax 技巧，在单个 Kernel 内完成计算，避免中间矩阵的显存读写。

#### Acceptance Criteria

1. THE FlashAttention_Engine SHALL 实现 Online Softmax 算法，支持流式计算最大值和指数和
2. THE FlashAttention_Engine SHALL 在单个 CUDA Kernel 内完成完整的注意力计算
3. WHEN 处理长度为 N 的序列时，THE FlashAttention_Engine SHALL 将显存复杂度从 O(N²) 降低到 O(N)
4. THE FlashAttention_Engine SHALL 避免存储完整的 N×N 注意力矩阵到全局内存
5. THE FlashAttention_Engine SHALL 支持因果掩码（Causal Mask）用于自回归生成场景
6. THE Verification_Module SHALL 验证 FlashAttention 输出与标准实现的数值一致性

### Requirement 4: Tensor Core 加速

**User Story:** 作为算子开发工程师，我需要利用 Tensor Core 进行矩阵乘加速，以充分发挥现代 GPU 的计算能力。

#### Acceptance Criteria

1. THE Tensor_Core_Accelerator SHALL 使用 WMMA API 或 MMA PTX 指令进行矩阵乘法
2. THE Tensor_Core_Accelerator SHALL 支持 FP16 精度的矩阵乘累加操作
3. THE Tensor_Core_Accelerator SHALL 支持 INT8 精度的矩阵乘累加操作
4. WHEN 使用 Tensor Core 时，THE GEMM_Kernel SHALL 确保矩阵维度满足对齐要求（如 16×16×16）
5. THE Tensor_Core_Accelerator SHALL 相比纯 CUDA Core 实现获得至少 4 倍的吞吐量提升

### Requirement 5: 高性能 GEMM Kernel 实现

**User Story:** 作为算子开发工程师，我需要实现支持混合精度的高性能 GEMM kernel，性能逼近或超越 cuBLAS。

#### Acceptance Criteria

1. THE GEMM_Kernel SHALL 支持 FP16 输入和 FP32 累加的混合精度计算
2. THE GEMM_Kernel SHALL 支持 INT8 输入和 INT32 累加的量化计算
3. THE GEMM_Kernel SHALL 实现 Register Tiling 以减少共享内存压力
4. THE GEMM_Kernel SHALL 使用 Warp Shuffle 指令进行高效的规约计算
5. WHEN 矩阵规模大于 1024×1024 时，THE GEMM_Kernel SHALL 达到 cuBLAS 性能的 90% 以上
6. THE GEMM_Kernel SHALL 支持转置和非转置的矩阵布局（NN, NT, TN, TT）

### Requirement 6: 流水线优化

**User Story:** 作为算子开发工程师，我需要实现计算与访存的流水线重叠，以隐藏内存延迟，最大化 GPU 利用率。

#### Acceptance Criteria

1. THE Pipeline_Scheduler SHALL 实现 Double Buffering 技术，使用两组缓冲区交替进行数据加载和计算
2. THE Pipeline_Scheduler SHALL 实现 Software Pipelining，将数据预取与计算重叠
3. WHEN 执行流水线时，THE Pipeline_Scheduler SHALL 确保计算单元和内存单元的并行利用
4. THE Pipeline_Scheduler SHALL 支持可配置的流水线深度（至少支持 2-4 级）
5. THE Attention_Kernel SHALL 通过流水线优化相比非流水线版本获得至少 20% 的性能提升

### Requirement 7: 性能分析与验证

**User Story:** 作为算子开发工程师，我需要使用专业工具进行性能分析和正确性验证，以确保优化效果和计算正确性。

#### Acceptance Criteria

1. THE Profiler SHALL 使用 Nsight Compute 收集 kernel 的性能指标
2. THE Profiler SHALL 报告计算吞吐量（TFLOPS）、内存带宽利用率和 SM 占用率
3. THE Verification_Module SHALL 使用 PyTorch 作为参考实现进行数值验证
4. WHEN 进行性能对比时，THE Profiler SHALL 与 cuBLAS/cuDNN 的对应实现进行基准测试
5. THE Profiler SHALL 识别并报告性能瓶颈（计算受限 vs 访存受限）

### Requirement 8: 接口与集成

**User Story:** 作为算子开发工程师，我需要提供清晰的 Python 接口，以便与 PyTorch 等深度学习框架集成。

#### Acceptance Criteria

1. THE Attention_Kernel SHALL 提供 Python 绑定接口，支持 PyTorch Tensor 输入输出
2. THE GEMM_Kernel SHALL 提供 Python 绑定接口，支持 PyTorch Tensor 输入输出
3. WHEN 输入参数无效时，THE Attention_Kernel SHALL 返回明确的错误信息
4. THE Attention_Kernel SHALL 支持批量处理（Batch）和多头注意力（Multi-Head）
5. THE GEMM_Kernel SHALL 支持任意形状的矩阵输入（在对齐约束范围内）


## 需求追溯矩阵

| 需求 | 设计组件 | 正确性属性 | 实现文件 |
|------|----------|------------|----------|
| 1.1-1.5 | Naive Attention Kernel | Property 1, 2 | naive_attention.cu |
| 2.1-2.5 | Shared Memory Manager | - | shared_memory.cuh, tiled_attention.cu |
| 3.1-3.6 | FlashAttention Engine | Property 3, 4 | flash_attention.cu, online_softmax.cuh |
| 4.1-4.5 | Tensor Core Accelerator | Property 5, 6, 8 | tensor_core_gemm.cu |
| 5.1-5.6 | GEMM Kernel | Property 5, 6, 7, 12 | hgemm_kernel.cu |
| 6.1-6.5 | Pipeline Scheduler | Property 9 | pipeline.cuh |
| 7.1-7.5 | Profiler | - | profiler.py, benchmarks/ |
| 8.1-8.5 | Python Interface | Property 10, 11, 13 | bindings.cpp, __init__.py |

## 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0 | 2026-01-01 | 初始版本，完成所有核心需求定义 |
