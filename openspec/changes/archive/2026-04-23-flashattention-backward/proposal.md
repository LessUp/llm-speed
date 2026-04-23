# Proposal: FlashAttention Backward Pass

> **Status:** Deferred backlog. This change is intentionally outside the active closeout scope.

## Why

Current implementation only supports forward pass. Backward pass is essential for:
- Model training
- Fine-tuning workflows
- Gradient-based optimization
- Full autograd support in PyTorch

## What Changes

- Implement FlashAttention backward kernel
- Create `torch.autograd.Function` wrapper
- Add gradient verification tests

## Capabilities

### New Capabilities
- FlashAttention Backward: Gradient computation for Q, K, V

### Modified Capabilities
- FlashAttention Algorithm (REQ-3): Add backward pass
- Python Interface (REQ-8): Add autograd support

## Impact

- Memory: O(N) with recompute strategy
- Risk: Medium - numerical precision in gradients requires careful implementation
- Dependencies: Requires forward pass implementation (complete)

## Success Criteria

- Gradients match PyTorch reference within tolerance
- Memory efficiency maintained (O(N))
- PyTorch autograd integration works correctly
- Performance comparable to forward pass
