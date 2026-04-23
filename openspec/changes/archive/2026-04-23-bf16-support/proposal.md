# Proposal: BF16 Precision Support

> **Status:** Deferred backlog. This change is intentionally outside the active closeout scope.

## Why

BFloat16 (BF16) is increasingly important for LLM inference due to:
- Better dynamic range than FP16 (same exponent range as FP32)
- Native support on Ampere+ GPUs (SM80+)
- Standard precision for many LLM model weights (LLaMA, etc.)

## What Changes

- Add BF16 attention kernels (naive, tiled, flash)
- Add BF16 GEMM kernels (standard, tensor core)
- Add Python bindings for BF16 tensors
- Add BF16-specific tests and benchmarks

## Capabilities

### New Capabilities
- BF16 Attention: FlashAttention with BF16 precision
- BF16 GEMM: Tensor Core GEMM with BF16 input

### Modified Capabilities
- Python Interface (REQ-8): Support BF16 tensor inputs
- Profiling (REQ-7): BF16 benchmarks

## Impact

- Hardware requirement: SM80+ (Ampere) for native BF16
- Fallback: FP32 emulation on older hardware (optional)
- Risk: Medium - new precision requires careful numerical verification

## Success Criteria

- BF16 kernels functionally correct
- Numerical accuracy within BF16 tolerance (≤1e-2 vs FP32 reference)
- Python interface accepts and returns BF16 tensors
- Performance comparable to FP16 implementation
