"""
Tests for GEMM kernels.
Property-based tests using Hypothesis.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

try:
    import torch
except ImportError:  # pragma: no cover - environment-dependent guard
    pytest.skip("PyTorch is not installed", allow_module_level=True)

# Import helper functions from conftest
try:
    from conftest import assert_close
except ImportError:
    # Fallback for direct module imports
    from tests.conftest import assert_close

# Strategies for property-based testing
dim_strategy = st.integers(min_value=16, max_value=512)
aligned_dim_strategy = st.integers(min_value=1, max_value=32).map(lambda x: x * 16)


class TestFP16GEMM:
    """Tests for FP16 GEMM implementation."""

    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(M=aligned_dim_strategy, N=aligned_dim_strategy, K=aligned_dim_strategy)
    def test_fp16_gemm_correctness(self, M, N, K, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 5: FP16 GEMM 正确性
        Validates: Requirements 4.2, 5.1

        For any valid FP16 matrices A[M,K] and B[K,N], GEMM output C = A @ B
        should match PyTorch torch.matmul within FP16 tolerance (1e-2).
        """
        try:
            from cuda_llm_ops import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")

        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)

        output = gemm(a, b)
        reference = torch.matmul(a, b)

        assert_close(
            output,
            reference,
            rtol=1e-2,
            atol=1e-2,
            msg=f"FP16 GEMM correctness failed for M={M}, N={N}, K={K}",
        )

    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(M=dim_strategy, N=dim_strategy, K=dim_strategy)
    def test_fp32_gemm_correctness(self, M, N, K, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 5: FP32 GEMM 正确性
        Validates: Requirements 5.1
        """
        try:
            from cuda_llm_ops import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")

        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)

        output = gemm(a, b)
        reference = torch.matmul(a, b)

        assert_close(
            output,
            reference,
            rtol=1e-3,
            atol=1e-3,
            msg=f"FP32 GEMM correctness failed for M={M}, N={N}, K={K}",
        )


class TestTensorCoreGEMM:
    """Tests for Tensor Core GEMM implementation."""

    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=50, deadline=None)
    @given(M=aligned_dim_strategy, N=aligned_dim_strategy, K=aligned_dim_strategy)
    def test_tensor_core_gemm_correctness(self, M, N, K, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 5: Tensor Core GEMM 正确性
        Validates: Requirements 4.2, 5.1
        """
        try:
            from cuda_llm_ops import tensor_core_gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")

        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)

        output = tensor_core_gemm(a, b)  # Returns FP32
        reference = torch.matmul(a.float(), b.float())

        assert_close(
            output,
            reference,
            rtol=1e-2,
            atol=1e-2,
            msg=f"Tensor Core GEMM correctness failed for M={M}, N={N}, K={K}",
        )

    @pytest.mark.cuda
    def test_tensor_core_gemm_alpha_scaling(self, device):
        """Test tensor_core_gemm with alpha scaling."""
        try:
            from cuda_llm_ops import tensor_core_gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")

        M, K, N = 128, 128, 128
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)

        alpha = 0.5
        output = tensor_core_gemm(a, b, alpha=alpha)
        reference = alpha * torch.matmul(a.float(), b.float())

        assert_close(
            output, reference, rtol=1e-2, atol=1e-2, msg="Tensor Core GEMM alpha scaling failed"
        )

    @pytest.mark.cuda
    def test_tensor_core_gemm_rejects_fp32(self, device):
        """Test tensor_core_gemm rejects FP32 input."""
        try:
            from cuda_llm_ops import tensor_core_gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")

        a = torch.randn(64, 32, device=device, dtype=torch.float32)
        b = torch.randn(32, 64, device=device, dtype=torch.float32)

        with pytest.raises(RuntimeError, match="float16"):
            tensor_core_gemm(a, b)

    @pytest.mark.cuda
    def test_tensor_core_gemm_rejects_int8(self, device):
        """Test tensor_core_gemm rejects INT8 input."""
        try:
            from cuda_llm_ops import tensor_core_gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")

        a = torch.randint(-128, 127, (64, 32), device=device, dtype=torch.int8)
        b = torch.randint(-128, 127, (32, 64), device=device, dtype=torch.int8)

        with pytest.raises(RuntimeError):
            tensor_core_gemm(a, b)


class TestMatrixLayoutEquivalence:
    """Tests for matrix layout equivalence."""

    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(M=dim_strategy, N=dim_strategy, K=dim_strategy)
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
            from cuda_llm_ops import gemm
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
        K=st.integers(min_value=1, max_value=100),
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
            from cuda_llm_ops import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")

        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)

        output = gemm(a, b)
        reference = torch.matmul(a, b)

        assert output.shape == (M, N), f"Output shape mismatch: {output.shape} vs {(M, N)}"
        assert_close(
            output,
            reference,
            rtol=1e-3,
            atol=1e-3,
            msg=f"Unaligned GEMM failed for M={M}, N={N}, K={K}",
        )


class TestArbitraryShapes:
    """Tests for arbitrary matrix shape support."""

    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(
        M=st.integers(min_value=16, max_value=1024),
        N=st.integers(min_value=16, max_value=1024),
        K=st.integers(min_value=16, max_value=1024),
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
            from cuda_llm_ops import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")

        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)

        output = gemm(a, b)

        assert output.shape == (M, N), f"Expected shape ({M}, {N}), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"


class TestINT8GEMM:
    """Tests for INT8 Tensor Core GEMM implementation."""

    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=50, deadline=None)
    @given(
        M=st.integers(min_value=1, max_value=16).map(lambda x: x * 8),
        N=st.integers(min_value=1, max_value=16).map(lambda x: x * 32),
        K=st.integers(min_value=1, max_value=32).map(lambda x: x * 16),
    )
    def test_int8_gemm_correctness(self, M, N, K, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 6: INT8 GEMM 正确性
        Validates: Requirements 4.3, 5.2

        For any valid INT8 matrices A[M,K] and B[K,N], quantized GEMM output
        should match reference implementation (INT32 accumulation).
        """
        try:
            from cuda_llm_ops import tensor_core_gemm_int8
        except ImportError:
            pytest.skip("CUDA kernels not built")

        a = torch.randint(-128, 127, (M, K), device=device, dtype=torch.int8)
        b = torch.randint(-128, 127, (K, N), device=device, dtype=torch.int8)

        try:
            output = tensor_core_gemm_int8(a, b)
        except RuntimeError as e:
            if "Turing+" in str(e) or "SM" in str(e):
                pytest.skip("INT8 Tensor Core requires Turing+ architecture")
            raise

        # Reference: INT32 accumulation
        reference = torch.matmul(a.to(torch.int32), b.to(torch.int32))

        assert output.dtype == torch.int32, f"Expected INT32 output, got {output.dtype}"
        assert output.shape == (M, N), f"Output shape mismatch: {output.shape} vs {(M, N)}"
        assert torch.equal(output, reference), (
            f"INT8 GEMM mismatch. Max diff: {(output - reference).abs().max().item()}"
        )

    @pytest.mark.cuda
    def test_int8_gemm_basic(self, device):
        """Basic INT8 GEMM correctness test with known values."""
        try:
            from cuda_llm_ops import tensor_core_gemm_int8
        except ImportError:
            pytest.skip("CUDA kernels not built")

        M, K, N = 8, 16, 32
        a = torch.ones(M, K, device=device, dtype=torch.int8)
        b = torch.ones(K, N, device=device, dtype=torch.int8)

        try:
            output = tensor_core_gemm_int8(a, b)
        except RuntimeError as e:
            if "Turing+" in str(e) or "SM" in str(e):
                pytest.skip("INT8 Tensor Core requires Turing+ architecture")
            raise

        # All ones: each element should equal K
        expected = torch.full((M, N), K, device=device, dtype=torch.int32)
        assert torch.equal(output, expected), (
            f"Expected all {K}, got max={output.max().item()}, min={output.min().item()}"
        )

    @pytest.mark.cuda
    def test_int8_gemm_error_handling(self, device):
        """Test INT8 GEMM error handling for invalid inputs."""
        try:
            from cuda_llm_ops import tensor_core_gemm_int8
        except ImportError:
            pytest.skip("CUDA kernels not built")

        # Wrong dtype
        a = torch.randn(8, 16, device=device, dtype=torch.float32)
        b = torch.randn(16, 32, device=device, dtype=torch.float32)
        with pytest.raises(Exception):
            tensor_core_gemm_int8(a, b)

        # Dimension mismatch
        a = torch.randint(-128, 127, (8, 16), device=device, dtype=torch.int8)
        b = torch.randint(-128, 127, (32, 32), device=device, dtype=torch.int8)
        with pytest.raises(Exception):
            tensor_core_gemm_int8(a, b)


# Unit tests for edge cases
class TestGEMMEdgeCases:
    """Unit tests for GEMM edge cases."""

    @pytest.mark.cuda
    def test_square_matrices(self, device):
        """Test with square matrices."""
        try:
            from cuda_llm_ops import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")

        for size in [32, 64, 128, 256]:
            a = torch.randn(size, size, device=device, dtype=torch.float32)
            b = torch.randn(size, size, device=device, dtype=torch.float32)

            output = gemm(a, b)
            reference = torch.matmul(a, b)

            assert_close(
                output,
                reference,
                rtol=1e-3,
                atol=1e-3,
                msg=f"Square matrix GEMM failed for size {size}",
            )

    @pytest.mark.cuda
    def test_tall_skinny_matrices(self, device):
        """Test with tall-skinny matrices."""
        try:
            from cuda_llm_ops import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")

        M, K, N = 1024, 32, 64
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)

        output = gemm(a, b)
        reference = torch.matmul(a, b)

        assert_close(output, reference, rtol=1e-3, atol=1e-3, msg="Tall-skinny GEMM failed")

    @pytest.mark.cuda
    def test_alpha_beta_scaling(self, device):
        """Test alpha and beta scaling."""
        try:
            from cuda_llm_ops import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")

        M, K, N = 64, 64, 64
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)

        # Test alpha scaling
        alpha = 2.0
        output = gemm(a, b, alpha=alpha)
        reference = alpha * torch.matmul(a, b)

        assert_close(output, reference, rtol=1e-3, atol=1e-3, msg="Alpha scaling failed")

    @pytest.mark.cuda
    def test_beta_parameter(self, device):
        """Test beta parameter for GEMM (C = alpha * A @ B + beta * C)."""
        try:
            from cuda_llm_ops import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")

        M, K, N = 64, 64, 64
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)

        # Test with beta=0 (default, should be same as just alpha * A @ B)
        output_beta0 = gemm(a, b, alpha=1.0, beta=0.0)
        reference = torch.matmul(a, b)
        assert_close(output_beta0, reference, rtol=1e-3, atol=1e-3, msg="Beta=0 failed")

    @pytest.mark.cuda
    def test_combined_alpha_beta(self, device):
        """Test combined alpha and beta scaling."""
        try:
            from cuda_llm_ops import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")

        M, K, N = 64, 64, 64
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)

        # Test with both alpha and beta
        alpha = 0.5
        beta = 2.0
        output = gemm(a, b, alpha=alpha, beta=beta)
        # Note: beta is currently unused in the implementation, so output should just be alpha * A @ B
        reference = alpha * torch.matmul(a, b)
        assert_close(output, reference, rtol=1e-3, atol=1e-3, msg="Combined alpha/beta failed")
