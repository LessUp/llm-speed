# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Complete SDD (Spec-Driven Development) documentation structure
- README.md files for all `/specs` subdirectories (product/, rfc/, api/, db/, testing/)
- Professional AGENTS.md with AI workflow instructions
- Updated CONTRIBUTING.md with SDD workflow guidelines

### Changed
- Restructured `/specs` directory with proper README files for each subdirectory
- Updated README.md with English primary and Chinese link
- Cleaned up project structure, removed `.qwen` directory
- Unified documentation organization following GitHub best practices

### Removed
- `.qwen` directory (AI tool configuration, no longer needed)

---

## [0.3.0] - 2026-04-16

### Added
- **Bilingual Documentation**: Complete Chinese and English documentation support
- **Quick Start Guide**: Help new users get started quickly
- **Troubleshooting Guide**: Systematically resolve common issues
- **Architecture Design Documentation**: In-depth technical detail analysis

### Changed
- Refactored `docs/` directory into bilingual structure
- Optimized organization of API reference documentation
- Improved professionalism of changelog

---

## [0.2.0] - 2026-04-16

### Added
- CPU-safe CI workflow with Python syntax validation
- GitHub Pages deployment triggers on both `master` and `main` branches
- Path-based filtering for Pages workflow
- Comprehensive API reference documentation (`docs/api.md`)

### Changed
- Updated GitHub Actions workflows for better performance
- Improved documentation structure and organization

---

## [0.1.2] - 2026-03-10

### Changed
- Standardized CI/CD workflows
- Updated build configurations

---

## [0.1.1] - 2025-02-27

### Added
- INT8 GEMM kernel support (Turing+)
- Python bindings for INT8 operations

### Fixed
- Various bug fixes in attention kernels

---

## [0.1.0] - 2025-02-13

### Added
- Initial release
- Naive attention kernel implementation
- Tiled attention with shared memory optimization
- FlashAttention with O(N) memory complexity
- Tensor Core GEMM (FP16/INT8)
- High-performance GEMM with register tiling
- Python bindings via pybind11
- Basic test suite with property-based testing

---

For detailed release notes and full changelog, see [docs/changelog/](docs/changelog/) or visit [GitHub Releases](https://github.com/LessUp/llm-speed/releases).
