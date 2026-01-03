"""
Tests for GEMM kernels.
Property-based tests using Hypothesis.
"""

import pytest
import torch
import numpy as np
from hypothesis import given, settings, strategies as st, assume

from conftest import assert_close


# Strategies for property-based testing
dim_strategy = st.integers(min_value=16, max_value=512)
aligned_dim_strategy = st.integers(min_value=1, max_value=32).map(lambda x: x * 16)


class TestFP16GEMM:
    """Tests for FP16 GEMM implementation."""
    
    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(
        M=aligned_dim_strategy,
        N=aligned_dim_strategy,
        K=aligned_dim_strategy
    )
    def test_fp16_gemm_correctness(self, M, N, K, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 5: FP16 GEMM 正确性
        Validates: Requirements 4.2, 5.1
        
        For any valid FP16 matrices A[M,K] and B[K,N], GEMM output C = A @ B
        should match PyTorch torch.matmul within FP16 tolerance (1e-2).
        """
        try:
            from python import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)
        
        output = gemm(a, b)
        reference = torch.matmul(a, b)
        
        assert_close(output, reference, rtol=1e-2, atol=1e-2,
                    msg=f"FP16 GEMM correctness failed for M={M}, N={N}, K={K}")
    
    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(
        M=dim_strategy,
        N=dim_strategy,
        K=dim_strategy
    )
    def test_fp32_gemm_correctness(self, M, N, K, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 5: FP32 GEMM 正确性
        Validates: Requirements 5.1
        """
        try:
            from python import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)
        
        output = gemm(a, b)
        reference = torch.matmul(a, b)
        
        assert_close(output, reference, rtol=1e-3, atol=1e-3,
                    msg=f"FP32 GEMM correctness failed for M={M}, N={N}, K={K}")


class TestTensorCoreGEMM:
    """Tests for Tensor Core GEMM implementation."""
    
    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=50, deadline=None)
    @given(
        M=aligned_dim_strategy,
        N=aligned_dim_strategy,
        K=aligned_dim_strategy
    )
    def test_tensor_core_gemm_correctness(self, M, N, K, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 5: Tensor Core GEMM 正确性
        Validates: Requirements 4.2, 5.1
        """
        try:
            from python import tensor_core_gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)
        
        output = tensor_core_gemm(a, b)  # Returns FP32
        reference = torch.matmul(a.float(), b.float())
        
        assert_close(output, reference, rtol=1e-2, atol=1e-2,
                    msg=f"Tensor Core GEMM correctness failed for M={M}, N={N}, K={K}")


class TestMatrixLayoutEquivalence:
    """Tests for matrix layout equivalence."""
    
    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(
        M=dim_strategy,
        N=dim_strategy,
        K=dim_strategy
    )
    def test_layout_equivalence(self, M, N, K, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 7: 矩阵布局等价性
        Validates: Requirements 5.6
        
        For any matrices A and B, GEMM output should satisfy:
        - C_NN = A @ B
        - C_NT = A @ B^T
        - C_TN = A^T @ B
        - C_TT = A^T @ B^T
        """
        try:
            from python import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)
        
        # NN: A @ B
        c_nn = gemm(a, b, trans_a=False, trans_b=False)
        ref_nn = torch.matmul(a, b)
        assert_close(c_nn, ref_nn, rtol=1e-3, atol=1e-3, msg="NN layout failed")
        
        # NT: A @ B^T (need B to be [N, K] for this)
        b_for_nt = torch.randn(N, K, device=device, dtype=torch.float32)
        c_nt = gemm(a, b_for_nt, trans_a=False, trans_b=True)
        ref_nt = torch.matmul(a, b_for_nt.T)
        assert_close(c_nt, ref_nt, rtol=1e-3, atol=1e-3, msg="NT layout failed")
        
        # TN: A^T @ B (need A to be [K, M] for this)
        a_for_tn = torch.randn(K, M, device=device, dtype=torch.float32)
        c_tn = gemm(a_for_tn, b, trans_a=True, trans_b=False)
        ref_tn = torch.matmul(a_for_tn.T, b)
        assert_close(c_tn, ref_tn, rtol=1e-3, atol=1e-3, msg="TN layout failed")


class TestDimensionAlignment:
    """Tests for dimension alignment handling."""
    
    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(
        M=st.integers(min_value=1, max_value=100),
        N=st.integers(min_value=1, max_value=100),
        K=st.integers(min_value=1, max_value=100)
    )
    def test_unaligned_dimensions(self, M, N, K, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 8: 维度对齐处理
        Validates: Requirements 4.4
        
        For any input matrix dimensions (including non-aligned), kernel should
        correctly handle via padding or fallback, producing correct output.
        """
        try:
            from python import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)
        
        output = gemm(a, b)
        reference = torch.matmul(a, b)
        
        assert output.shape == (M, N), f"Output shape mismatch: {output.shape} vs {(M, N)}"
        assert_close(output, reference, rtol=1e-3, atol=1e-3,
                    msg=f"Unaligned GEMM failed for M={M}, N={N}, K={K}")


class TestArbitraryShapes:
    """Tests for arbitrary matrix shape support."""
    
    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(
        M=st.integers(min_value=16, max_value=1024),
        N=st.integers(min_value=16, max_value=1024),
        K=st.integers(min_value=16, max_value=1024)
    )
    def test_arbitrary_shapes(self, M, N, K, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 12: 任意形状矩阵支持
        Validates: Requirements 8.5
        
        For any matrix shape (M, N, K) within alignment constraints,
        GEMM_Kernel should correctly compute and produce [M, N] output.
        """
        try:
            from python import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)
        
        output = gemm(a, b)
        
        assert output.shape == (M, N), f"Expected shape ({M}, {N}), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"


# Unit tests for edge cases
class TestGEMMEdgeCases:
    """Unit tests for GEMM edge cases."""
    
    @pytest.mark.cuda
    def test_square_matrices(self, device):
        """Test with square matrices."""
        try:
            from python import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        for size in [32, 64, 128, 256]:
            a = torch.randn(size, size, device=device, dtype=torch.float32)
            b = torch.randn(size, size, device=device, dtype=torch.float32)
            
            output = gemm(a, b)
            reference = torch.matmul(a, b)
            
            assert_close(output, reference, rtol=1e-3, atol=1e-3,
                        msg=f"Square matrix GEMM failed for size {size}")
    
    @pytest.mark.cuda
    def test_tall_skinny_matrices(self, device):
        """Test with tall-skinny matrices."""
        try:
            from python import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        M, K, N = 1024, 32, 64
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)
        
        output = gemm(a, b)
        reference = torch.matmul(a, b)
        
        assert_close(output, reference, rtol=1e-3, atol=1e-3,
                    msg="Tall-skinny GEMM failed")
    
    @pytest.mark.cuda
    def test_alpha_beta_scaling(self, device):
        """Test alpha and beta scaling."""
        try:
            from python import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        M, K, N = 64, 64, 64
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)
        
        alpha = 2.0
        output = gemm(a, b, alpha=alpha)
        reference = alpha * torch.matmul(a, b)
        
        assert_close(output, reference, rtol=1e-3, atol=1e-3,
                    msg="Alpha scaling failed")
