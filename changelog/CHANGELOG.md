# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive API reference documentation (`docs/api.md`)
- Performance tuning guide (`docs/performance.md`)
- Enhanced CLAUDE.md with quick reference tables and common tasks

### Changed
- Completely restructured documentation architecture
- Unified all changelog entries into single CHANGELOG.md
- Optimized .kiro specs documentation format with tables and status indicators
- Enhanced Git Pages configuration with better SEO and navigation
- Improved CI workflow reliability

## [0.2.0] - 2026-04-16

### Added
- CPU-safe CI workflow with Python syntax validation
- GitHub Pages deployment triggers on both `master` and `main` branches
- Path-based filtering for Pages workflow to reduce unnecessary builds
- API reference documentation (`docs/api.md`)
- Performance tuning guide (`docs/performance.md`)

### Changed
- Documentation architecture restructured: README as repository entry, index.md as docs homepage
- DeepWiki guide clarified as primary usage documentation
- CI workflow simplified to Python lint and CPU-safe smoke tests (removed invalid CUDA jobs)
- Removed `pytest ... || true` fallback logic; accept exit code 5 when no CPU-safe tests available
- All Python files formatted with ruff

### Fixed
- Pages workflow now correctly triggers on the actual `master` branch
- Python code formatting issues (ruff format)

## [0.1.2] - 2026-03-10

### Changed
- Standardized CI workflow permissions (`contents: read`) and concurrency configuration
- Added `actions/configure-pages@v5` step to Pages workflow
- Added `paths` trigger filtering to Pages workflow

## [0.1.1] - 2025-02-27

### Added
- FlashAttention double buffering pipeline implementation
  - K/V tiles use alternating buffers for compute/load overlap
  - Causal mask early-exit optimization when tile exceeds causal window
- INT8 Tensor Core GEMM Python binding (`tensor_core_gemm_int8`)
  - INT8 input, INT32 accumulation output
  - Runtime SM version check (requires Turing+ SM≥7.2)
  - Property-based tests with Hypothesis
- `pyproject.toml` for pytest markers configuration (cuda, slow, property)
- JSON export support for benchmark scripts (`--output` flag)
- GPU peak memory tracking in benchmarks

### Changed
- FlashAttention thread utilization improved from 25% to 100%
  - Split into two phases: softmax state update (light) and output update (heavy)
  - All threads cooperate in Phase 2 for better parallelism
- Removed unused static template FlashAttention kernel (16KB register overflow)
- Shared memory bank conflict elimination across all kernels:
  - `+1` padding for attention kernels
  - `+8` half padding for Tensor Core GEMM
- HGEMM kernel enhanced with double buffering and bank conflict padding

### Fixed
- **[Critical]** `tiled_attention.cu`: Shared memory out-of-bounds access when `head_dim > 64`
  - Changed from fixed `BLOCK_K=64` to dynamic shared memory based on actual `head_dim`
- **[Critical]** `tensor_core_gemm.cu`: INT8 host wrapper always unavailable
  - Changed from `#if __CUDA_ARCH__` to runtime SM version check
- **[Critical]** `naive_attention.cu`: Divide-by-zero risk when `block_sum` is zero
  - Added protection check before division
- Integer overflow in GEMM index calculations (changed to `int64_t`)
- Divide-by-zero protection in FlashAttention and online softmax
- Spec documentation synced with actual implementation:
  - Interface signatures corrected
  - Pipeline API descriptions rewritten
  - Implementation status updated in requirements matrix

### Removed
- Dead code: unused `softmax_row` device function and `naive_attention_kernel`
- Unused `import time` in benchmark scripts

## [0.1.0] - 2025-02-13

### Added
- Initial project infrastructure
  - MIT LICENSE file
  - `.gitignore` for CUDA/Python/IDE
  - `.editorconfig` for consistent code formatting
  - Standard badges in README (License, CUDA, C++, Python)
- Core CUDA kernels
  - `naive_attention.cu`: Baseline attention with O(N²) memory
  - `tiled_attention.cu`: Shared memory tiled optimization
  - `flash_attention.cu`: O(N) memory with online softmax
  - `tensor_core_gemm.cu`: WMMA-based Tensor Core GEMM
  - `hgemm_kernel.cu`: High-performance GEMM with register tiling
- Header primitives library
  - `common.cuh`: Core types and utilities
  - `online_softmax.cuh`: Online softmax algorithm
  - `warp_primitives.cuh`: Warp-level operations
  - `shared_memory.cuh`: Shared memory management
  - `pipeline.cuh`: Memory prefetch pipeline utilities
- Python bindings via pybind11
  - `naive_attention`, `tiled_attention`, `flash_attention`
  - `gemm`, `tensor_core_gemm`
- Test suite with pytest and Hypothesis
- Benchmark scripts for attention and GEMM

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| Unreleased | - | Documentation restructure, API reference, performance guide |
| 0.2.0 | 2026-04-16 | CI/CD fixes, documentation architecture, Python formatting |
| 0.1.2 | 2026-03-10 | Workflow deep standardization |
| 0.1.1 | 2025-02-27 | Double buffering, INT8 binding, critical bug fixes |
| 0.1.0 | 2025-02-13 | Initial release |

## Migration Guide

### Upgrading to 0.2.0

No breaking changes. Documentation URLs remain stable.

### Upgrading to 0.1.1

**INT8 GEMM users**: The new `tensor_core_gemm_int8` function requires Turing+ GPU. Check SM version:

```python
import torch
capability = torch.cuda.get_device_capability()
if capability[0] >= 7 and capability[1] >= 2:
    c = tensor_core_gemm_int8(a_int8, b_int8)
else:
    print("INT8 Tensor Core requires Turing+ (SM 7.2+)")
```

**Attention users**: All attention functions now have proper divide-by-zero protection. No code changes required.

[Unreleased]: https://github.com/LessUp/llm-speed/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/LessUp/llm-speed/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/LessUp/llm-speed/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/LessUp/llm-speed/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/LessUp/llm-speed/releases/tag/v0.1.0
