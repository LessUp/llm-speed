# FlashAttention Double Buffering & INT8 GEMM Python 绑定

**日期**: 2025-02-27
**范围**: Task 11.2, Task 15

## Task 11.2: FlashAttention Double Buffering

### 变更文件
- `src/flash_attention.cu` — 核心改动

### 实现细节
- 将 `smem_K` 和 `smem_V` 从单缓冲改为双缓冲（`smem_K_buf[2 * kv_buf_size]`, `smem_V_buf[2 * kv_buf_size]`）
- 新增 `pipeline.cuh` include
- 新增 `SMEM_K(buf)` / `SMEM_V(buf)` 宏用于访问交替缓冲区
- 新增 `LOAD_KV_TILE(buf_idx, col_start)` 宏统一加载逻辑
- Prologue: 加载第一个 K/V tile 到 buffer 0
- 主循环: 计算当前 `cur_buf` 的同时，预取下一个 tile 到 `next_buf`
- Causal mask 场景: 当下一个 tile 超出因果窗口时跳过 prefetch
- Host wrapper: `smem_size` 从 `2 * BLOCK_N * hd_stride` 增加到 `4 * BLOCK_N * hd_stride`

### 共享内存影响
- 典型配置 (BLOCK_M=32, BLOCK_N=32, head_dim=64): 约 30KB → 45KB
- Ampere 上可接受（最大 164KB/SM）

## Task 15: INT8 GEMM Python 绑定

### 变更文件
- `python/bindings.cpp` — 添加 `tensor_core_gemm_int8_wrapper` 和 pybind11 注册
- `python/__init__.py` — 导出 `tensor_core_gemm_int8`
- `tests/test_gemm.py` — 新增 `TestINT8GEMM` 测试类

### 实现细节
- wrapper 函数: 验证 INT8 dtype、2D shape、维度匹配、CUDA device、contiguous
- 输出 dtype: INT32
- 运行时架构检查: 由底层 `tensor_core_gemm_int8()` C++ 函数在 `tensor_core_gemm.cu` 中执行 SM≥7.2 检查

### 测试覆盖
- **属性测试**: 随机 INT8 矩阵 (对齐 WMMA 8x32x16) 与 INT32 参考实现精确匹配
- **单元测试**: 全 1 矩阵验证、错误处理（错误 dtype、维度不匹配）
- 在非 Turing+ 架构上自动 skip

## Spec 文档同步更新
- `design.md`: 更新 FlashAttention 共享内存布局（K/V 双缓冲）；已知限制从 6 项缩减为 4 项
- `requirements.md`: 流水线集成和 Python 接口状态更新为 ✅ 完成
- `tasks.md`: Task 11/11.2/15 标记完成；实现状态表和依赖图更新
