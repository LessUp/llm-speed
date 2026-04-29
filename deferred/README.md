# Deferred Work

This directory contains completed but intentionally deferred patches.

## BF16 Support

**File:** `bf16-support.patch` (22KB)

**Status:** Deferred backlog — see `openspec/changes/archive/2026-04-23-bf16-support/`

**Apply with:**
```bash
git apply deferred/bf16-support.patch
```

**Scope:**
- BF16 attention kernels (naive, tiled, flash)
- BF16 GEMM kernels
- Python bindings with SM80+ runtime check
- Tests for BF16 precision

**Why deferred:**
- Project closeout focused on stability and simplification
- BF16 is a feature expansion, not a fix
- Will be revisited when project scope expands

**Hardware requirement:** SM80+ (Ampere) for native BF16
