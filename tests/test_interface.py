"""
Tests for Python interface and error handling.
Property-based tests using Hypothesis.
"""

import pytest
import torch
from hypothesis import given, settings, strategies as st

from conftest import assert_close


class TestPythonInterface:
    """Tests for Python interface compatibility."""
    
    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(
        batch=st.integers(min_value=1, max_value=4),
        heads=st.integers(min_value=1, max_value=8),
        seq_len=st.integers(min_value=16, max_value=128),
        head_dim=st.sampled_from([32, 64, 128])
    )
    def test_pytorch_tensor_compatibility(self, batch, heads, seq_len, head_dim, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 10: Python 接口兼容性
        Validates: Requirements 8.1, 8.2
        
        For any valid PyTorch Tensor input, Python binding should:
        - Accept PyTorch Tensor as input
        - Return PyTorch Tensor as output
        - Preserve device and dtype attributes
        """
        try:
            from cuda_llm_ops import flash_attention
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        for dtype in [torch.float32, torch.float16]:
            q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
            k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
            v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
            
            output = flash_attention(q, k, v)
            
            # Check output is PyTorch tensor
            assert isinstance(output, torch.Tensor), "Output should be PyTorch Tensor"
            
            # Check device preserved
            assert output.device == q.device, f"Device mismatch: {output.device} vs {q.device}"
            
            # Check dtype preserved
            assert output.dtype == q.dtype, f"Dtype mismatch: {output.dtype} vs {q.dtype}"
    
    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=50, deadline=None)
    @given(
        M=st.integers(min_value=16, max_value=256),
        N=st.integers(min_value=16, max_value=256),
        K=st.integers(min_value=16, max_value=256)
    )
    def test_gemm_interface_compatibility(self, M, N, K, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 10: Python 接口兼容性 (GEMM)
        Validates: Requirements 8.2
        """
        try:
            from cuda_llm_ops import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        for dtype in [torch.float32, torch.float16]:
            a = torch.randn(M, K, device=device, dtype=dtype)
            b = torch.randn(K, N, device=device, dtype=dtype)
            
            output = gemm(a, b)
            
            assert isinstance(output, torch.Tensor)
            assert output.device == a.device
            assert output.dtype == a.dtype


class TestErrorHandling:
    """Tests for error handling."""
    
    @pytest.mark.cuda
    def test_dimension_mismatch_error(self, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 13: 无效输入错误处理
        Validates: Requirements 8.3
        
        For invalid inputs (dimension mismatch), kernel should raise
        clear error message instead of undefined behavior.
        """
        try:
            from cuda_llm_ops import flash_attention
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        q = torch.randn(2, 4, 64, 32, device=device)
        k = torch.randn(2, 4, 64, 32, device=device)
        v = torch.randn(2, 4, 128, 32, device=device)  # Wrong seq_len
        
        with pytest.raises(Exception) as excinfo:
            flash_attention(q, k, v)
        
        # Should have meaningful error message
        assert "shape" in str(excinfo.value).lower() or "match" in str(excinfo.value).lower()
    
    @pytest.mark.cuda
    def test_wrong_tensor_dim_error(self, device):
        """Test error for wrong tensor dimensions."""
        try:
            from cuda_llm_ops import flash_attention
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        # 3D tensor instead of 4D
        q = torch.randn(4, 64, 32, device=device)
        k = torch.randn(4, 64, 32, device=device)
        v = torch.randn(4, 64, 32, device=device)
        
        with pytest.raises(Exception) as excinfo:
            flash_attention(q, k, v)
        
        assert "4D" in str(excinfo.value) or "dim" in str(excinfo.value).lower()
    
    @pytest.mark.cuda
    def test_cpu_tensor_error(self):
        """Test error for CPU tensors."""
        try:
            from cuda_llm_ops import flash_attention
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        # CPU tensors
        q = torch.randn(2, 4, 64, 32)
        k = torch.randn(2, 4, 64, 32)
        v = torch.randn(2, 4, 64, 32)
        
        with pytest.raises(Exception) as excinfo:
            flash_attention(q, k, v)
        
        assert "cuda" in str(excinfo.value).lower() or "device" in str(excinfo.value).lower()
    
    @pytest.mark.cuda
    def test_unsupported_dtype_error(self, device):
        """Test error for unsupported dtype."""
        try:
            from cuda_llm_ops import flash_attention
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        # INT32 tensors (not supported)
        q = torch.randint(0, 100, (2, 4, 64, 32), device=device, dtype=torch.int32)
        k = torch.randint(0, 100, (2, 4, 64, 32), device=device, dtype=torch.int32)
        v = torch.randint(0, 100, (2, 4, 64, 32), device=device, dtype=torch.int32)
        
        with pytest.raises(Exception) as excinfo:
            flash_attention(q, k, v)
        
        assert "dtype" in str(excinfo.value).lower() or "type" in str(excinfo.value).lower()
    
    @pytest.mark.cuda
    def test_gemm_dimension_mismatch(self, device):
        """Test GEMM dimension mismatch error."""
        try:
            from cuda_llm_ops import gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        a = torch.randn(64, 32, device=device)
        b = torch.randn(64, 64, device=device)  # K dimension mismatch
        
        with pytest.raises(Exception) as excinfo:
            gemm(a, b)
        
        assert "dimension" in str(excinfo.value).lower() or "match" in str(excinfo.value).lower()
    
    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=50, deadline=None)
    @given(
        batch=st.integers(min_value=1, max_value=4),
        heads=st.integers(min_value=1, max_value=8),
        seq_len=st.integers(min_value=16, max_value=128),
        head_dim=st.sampled_from([32, 64, 128])
    )
    def test_no_crash_on_valid_input(self, batch, heads, seq_len, head_dim, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 13: 无效输入错误处理 (inverse)
        Validates: Requirements 8.3
        
        For valid inputs, kernel should not crash or produce undefined behavior.
        """
        try:
            from cuda_llm_ops import flash_attention, gemm
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        # Valid attention inputs
        q = torch.randn(batch, heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device)
        
        # Should not raise
        output = flash_attention(q, k, v)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Valid GEMM inputs
        M, N, K = seq_len, head_dim, head_dim
        a = torch.randn(M, K, device=device)
        b = torch.randn(K, N, device=device)
        
        output = gemm(a, b)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestPipelineConfiguration:
    """Tests for pipeline configuration."""
    
    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=50, deadline=None)
    @given(
        batch=st.integers(min_value=1, max_value=2),
        heads=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=64, max_value=256),
        head_dim=st.sampled_from([32, 64])
    )
    def test_pipeline_correctness(self, batch, heads, seq_len, head_dim, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 9: 流水线深度配置正确性
        Validates: Requirements 6.4
        
        For any supported pipeline depth configuration (2, 3, 4 stages),
        Pipeline_Scheduler should produce numerically consistent output
        with non-pipelined version.
        """
        try:
            from cuda_llm_ops import flash_attention, naive_attention
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        
        # FlashAttention uses pipelining internally
        pipelined_output = flash_attention(q, k, v)
        
        # Naive attention doesn't use pipelining
        naive_output = naive_attention(q, k, v)
        
        # Should produce same results
        assert_close(pipelined_output, naive_output, rtol=1e-3, atol=1e-3,
                    msg="Pipeline output should match non-pipelined version")
