# 2025-02-27 项目全面优化

## Bug 修复

### tiled_attention.cu — 共享内存越界访问 [严重]
- **问题**: `smem_Q/K/V` 声明为 `[BLOCK_M][BLOCK_K+1]`（BLOCK_K=64），但加载循环使用 `head_dim`（可达 128）作为列索引，导致越界写入
- **修复**: 移除 `BLOCK_K` 模板参数，改用动态共享内存，尺寸基于实际 `head_dim`；同时添加 `+1` padding 避免 bank conflict

### tensor_core_gemm.cu — INT8 host wrapper 永远不可用 [严重]
- **问题**: `tensor_core_gemm_int8()` 被 `#if __CUDA_ARCH__ >= 720` 包裹，但 `__CUDA_ARCH__` 在 host 代码中未定义，导致永远走 `#else` 抛异常
- **修复**: 改为运行时查询 `cudaGetDeviceProperties` 检查 SM 版本，并在两个预处理分支中都保留 kernel launch 代码

## 性能优化

### flash_attention.cu — 线程利用率从 25% 提升到 100%
- **问题**: 动态 kernel 的 softmax+output 更新只用 `tid < BLOCK_M`（32/128 线程），75% 线程空转
- **修复**: 拆分为两阶段——Phase 1 softmax（1 thread/row，O(BLOCK_N) 操作，轻量）；Phase 2 output update（ALL threads 协作，O(BLOCK_M × head_dim × BLOCK_N) 操作，重量）

### flash_attention.cu — 移除未使用的静态模板 kernel
- **问题**: `flash_attention_forward_kernel`（静态 HEAD_DIM 模板版）每线程声明 `float output[BLOCK_M * HEAD_DIM]` = 16KB/线程，严重寄存器溢出
- **修复**: 完全移除，仅保留优化后的动态 kernel

### flash_attention.cu + tiled_attention.cu — 共享内存 bank conflict 消除
- 所有共享内存数组添加 `+1` stride padding

### tensor_core_gemm.cu — 共享内存 bank conflict 消除
- `smem_A` 和 `smem_B` 添加 `+8` half 元素 padding
- 更新 WMMA `load_matrix_sync` 的 stride 参数

### hgemm_kernel.cu — 双缓冲 + bank conflict padding
- 实现双缓冲（double buffering）：分配两套共享内存 tile，计算当前 tile 时预加载下一个
- 隐藏全局内存加载延迟，提升计算/访存重叠度

## 构建系统

### Windows 兼容性
- `CMakeLists.txt`: `-Xcompiler -fPIC` 改为条件编译（`if(NOT WIN32)`）
- `setup.py`: 根据 `platform.system()` 选择编译器标志（Windows 用 `/O2 /std:c++17`，Linux 用 `-O3 -std=c++17 -fPIC`）

### CMakeLists.txt 清理
- 移除空的 `file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")` 及其引用

## 代码质量

### 死代码清理
- `naive_attention.cu`: 移除未使用的 `softmax_row` 设备函数和 `naive_attention_kernel`
- `benchmark_attention.py` / `benchmark_gemm.py`: 移除未使用的 `import time`

### Benchmark 路径处理
- `sys.path.insert(0, '..')` 替换为 `Path(__file__).resolve().parent.parent`（不依赖 cwd）

### 添加 pyproject.toml
- pytest markers 配置（cuda, slow, property）
- 测试路径配置

## 功能增强

### Benchmark 结果导出
- 两个 benchmark 脚本新增 `--output` 参数，支持 JSON 格式导出

### GPU 显存追踪
- benchmark 运行时记录 `peak_memory_mb` 和 `input_memory_mb`

## 修改文件清单
- `src/tiled_attention.cu`
- `src/tensor_core_gemm.cu`
- `src/flash_attention.cu`
- `src/naive_attention.cu`
- `src/hgemm_kernel.cu`
- `CMakeLists.txt`
- `setup.py`
- `benchmarks/benchmark_attention.py`
- `benchmarks/benchmark_gemm.py`
- `pyproject.toml` (新增)
