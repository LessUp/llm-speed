# Changelog

This file tracks notable, user-relevant project changes. Repository restructuring noise and repetitive documentation churn are intentionally omitted.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-04-30

### Added
- Complete OpenSpec integration with automated change management
- Bilingual documentation (Chinese/English) for all user-facing docs
- GitHub Pages documentation site with professional design
- Comprehensive type stubs (`_cuda_llm_ops.pyi`) for IDE support

### Changed
- Normalized the repository around OpenSpec as the governance layer
- Simplified CI/CD pipeline to essential high-signal checks
- Unified Python version configuration (3.8 minimum, 3.12 tested)
- Refined AI workflow guidance (AGENTS.md, CLAUDE.md, Copilot instructions)

### Fixed
- CMake Python module path/name mismatch for pybind11 build target
- Version fallback behavior for editable or unbundled Python imports
- Type annotations in Python stubs using proper `torch.Tensor` types

### Technical Details
- **Code Quality**: Zero technical debt (no TODO/FIXME/XXX annotations)
- **Test Coverage**: pytest + Hypothesis property-based testing
- **Documentation**: 12 bilingual documentation files
- **GPU Support**: Volta (SM70) through Hopper (SM90)

### Archived Changes
See `openspec/changes/archive/` for detailed change specifications:
- `2026-04-23-project-closeout` - Major cleanup and normalization
- `2026-04-27-final-cleanup` - Residual issue fixes

### Deferred Features
The following features are complete but deferred for future releases:
- `bf16-support` - BF16 precision kernels (see `deferred/bf16-support.patch`)
- `flashattention-backward` - FlashAttention backward pass

## [0.3.0] - 2026-04-16

### Added
- Bilingual documentation set for setup, API, architecture, and troubleshooting
- CPU-safe CI checks and GitHub Pages publishing
- OpenSpec workflow integration

## [0.1.1] - 2025-02-27

### Added
- FlashAttention double buffering pipeline improvements
- INT8 Tensor Core GEMM binding and tests

### Fixed
- Shared memory, divide-by-zero, and runtime availability issues across kernels and bindings

## [0.1.0] - 2025-02-13

### Added
- Initial CUDA attention and GEMM kernels
  - Naive attention, Tiled attention, FlashAttention forward
  - Tensor Core GEMM, Optimized HGEMM
- Shared CUDA helper primitives
  - Online softmax, Pipeline, Shared memory utilities, Warp primitives
- Python bindings via pybind11
- Benchmark and pytest-based validation scaffolding
