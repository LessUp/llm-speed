# Tasks: BF16 Precision Support

## 1. Attention Kernels
- [ ] 1.1 Implement BF16 FlashAttention kernel (`src/flash_attention_bf16.cu`)
- [ ] 1.2 Add online softmax BF16 variant
- [ ] 1.3 Implement BF16 tiled attention kernel
- [ ] 1.4 Implement BF16 naive attention kernel

## 2. GEMM Kernels
- [ ] 2.1 Implement BF16 GEMM kernel (`src/hgemm_bf16_kernel.cu`)
- [ ] 2.2 Implement BF16 Tensor Core GEMM (`src/tensor_core_gemm_bf16.cu`)
- [ ] 2.3 Add register tiling optimization
- [ ] 2.4 Support NN, NT, TN, TT layouts

## 3. Python Bindings
- [ ] 3.1 Add BF16 tensor detection in `cuda_llm_ops/bindings.cpp`
- [ ] 3.2 Expose BF16 attention functions to Python
- [ ] 3.3 Expose BF16 GEMM functions to Python
- [ ] 3.4 Add SM version runtime check
- [ ] 3.5 Update `cuda_llm_ops/__init__.py` with BF16 exports

## 4. Testing
- [ ] 4.1 Create `tests/test_attention_bf16.py`
- [ ] 4.2 Create `tests/test_gemm_bf16.py`
- [ ] 4.3 Add numerical accuracy tests (≤1e-2 vs FP32 reference)
- [ ] 4.4 Add property-based tests for BF16
- [ ] 4.5 Add SM80+ guard for BF16 tests

## 5. Benchmarks
- [ ] 5.1 Add BF16 attention benchmarks
- [ ] 5.2 Add BF16 GEMM benchmarks
- [ ] 5.3 Compare performance vs FP16
