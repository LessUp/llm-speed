# Changelog

This file tracks notable, user-relevant project changes. Repository restructuring noise and repetitive documentation churn are intentionally omitted.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- Normalized the repository around OpenSpec as the only active governance layer.
- Simplified documentation, AI workflow guidance, and GitHub Pages structure.
- Reduced CI and Pages automation to higher-signal closeout-era checks.

### Fixed
- Corrected the CMake Python module path/name mismatch for the pybind11 build target.
- Improved version fallback behavior for editable or unbundled Python imports.

## [0.3.0] - 2026-04-16

### Added
- Bilingual documentation set for setup, API, architecture, and troubleshooting.
- CPU-safe CI checks and GitHub Pages publishing.

## [0.1.1] - 2025-02-27

### Added
- FlashAttention double buffering pipeline improvements.
- INT8 Tensor Core GEMM binding and tests.

### Fixed
- Shared memory, divide-by-zero, and runtime availability issues across kernels and bindings.

## [0.1.0] - 2025-02-13

### Added
- Initial CUDA attention and GEMM kernels.
- Shared CUDA helper primitives and Python bindings.
- Benchmark and pytest-based validation scaffolding.
