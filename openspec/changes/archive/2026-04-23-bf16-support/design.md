# Design: BF16 Precision Support

## Technical Approach

### Hardware Requirements
- Minimum: SM80 (Ampere) for native BF16
- CUDA 11.0+ for `nv_bfloat16` type
- Tensor Core support for BF16 matrix operations

### Kernel Implementation

BF16 kernels follow the same optimization patterns as FP16:
- Register tiling with BF16 loads/stores
- FP32 accumulation for numerical precision
- Tensor Core utilization via WMMA BF16 instructions

### Attention Kernel

```cpp
// src/flash_attention_bf16.cu
template<int BLOCK_M, int BLOCK_N>
__global__ void flash_attention_bf16_kernel(
    const nv_bfloat16* Q,
    const nv_bfloat16* K,
    const nv_bfloat16* V,
    nv_bfloat16* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool is_causal
);
```

### GEMM Kernel

```cpp
// src/hgemm_bf16_kernel.cu
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void hgemm_bf16_kernel(
    const nv_bfloat16* A,
    const nv_bfloat16* B,
    nv_bfloat16* C,
    int M, int N, int K, float alpha, float beta
);
```

### Numerical Precision

- Accumulation: FP32
- Output: BF16
- Tolerance: ≤1e-2 vs FP32 reference

### Files to Create/Modify

| File | Action |
|------|--------|
| `src/flash_attention_bf16.cu` | Create |
| `src/hgemm_bf16_kernel.cu` | Create |
| `src/tensor_core_gemm_bf16.cu` | Create |
| `cuda_llm_ops/bindings.cpp` | Modify - add BF16 bindings |
| `tests/test_attention_bf16.py` | Create |
| `tests/test_gemm_bf16.py` | Create |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| SM version mismatch | Runtime check with clear error message |
| Numerical precision issues | FP32 accumulation, extensive testing |
| Performance regression | Benchmark vs FP16, optimize if needed |
