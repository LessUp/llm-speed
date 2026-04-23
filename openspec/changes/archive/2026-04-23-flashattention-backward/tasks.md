# Tasks: FlashAttention Backward Pass

## 1. Kernel Implementation
- [ ] 1.1 Implement backward kernel (`src/flash_attention_backward.cu`)
- [ ] 1.2 Handle causal mask in backward correctly
- [ ] 1.3 Optimize memory access patterns
- [ ] 1.4 Implement recompute strategy for memory efficiency

## 2. Python Integration
- [ ] 2.1 Create `cuda_llm_ops/flash_attention.py` with autograd wrapper
- [ ] 2.2 Create `FlashAttentionFunction` class extending `torch.autograd.Function`
- [ ] 2.3 Expose via `flash_attention_with_grad` function
- [ ] 2.4 Save tensors for backward correctly in forward pass
- [ ] 2.5 Update `cuda_llm_ops/__init__.py` with new exports

## 3. Gradient Verification
- [ ] 3.1 Implement gradient check tests (`tests/test_flash_attention_backward.py`)
- [ ] 3.2 Compare with PyTorch SDPA gradients
- [ ] 3.3 Test numerical precision (tolerance matching forward pass)
- [ ] 3.4 Test with various batch sizes and sequence lengths
- [ ] 3.5 Test causal and non-causal modes

## 4. Performance
- [ ] 4.1 Benchmark backward pass
- [ ] 4.2 Profile memory usage (verify O(N))
- [ ] 4.3 Optimize tile sizes if needed
- [ ] 4.4 Compare with FlashAttention-2 reference if available

## 5. Documentation
- [ ] 5.1 Update API documentation
- [ ] 5.2 Add usage examples
- [ ] 5.3 Document autograd behavior
