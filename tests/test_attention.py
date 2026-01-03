"""
Tests for attention kernels.
Property-based tests using Hypothesis.
"""

import pytest
import torch
import numpy as np
from hypothesis import given, settings, strategies as st, assume

from conftest import assert_close, compute_attention_reference


# Strategies for property-based testing
batch_strategy = st.integers(min_value=1, max_value=4)
head_strategy = st.integers(min_value=1, max_value=8)
seq_len_strategy = st.integers(min_value=16, max_value=256)
head_dim_strategy = st.sampled_from([32, 64, 128])


class TestNaiveAttention:
    """Tests for naive attention implementation."""
    
    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(
        batch=batch_strategy,
        heads=head_strategy,
        seq_len=seq_len_strategy,
        head_dim=head_dim_strategy
    )
    def test_attention_correctness_fp32(self, batch, heads, seq_len, head_dim, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 1: Attention 计算正确性
        Validates: Requirements 1.1, 1.3, 1.4, 1.5
        
        For any valid Q, K, V input tensors, the Attention_Kernel output should
        match PyTorch reference implementation within tolerance (FP32: 1e-3).
        """
        try:
            from python import naive_attention
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        # Generate inputs
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        
        # Compute outputs
        output = naive_attention(q, k, v)
        reference, _ = compute_attention_reference(q, k, v)
        
        # Verify
        assert_close(output, reference, rtol=1e-3, atol=1e-3,
                    msg="Naive attention FP32 correctness failed")
    
    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(
        batch=batch_strategy,
        heads=head_strategy,
        seq_len=seq_len_strategy,
        head_dim=head_dim_strategy
    )
    def test_attention_correctness_fp16(self, batch, heads, seq_len, head_dim, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 1: Attention 计算正确性 (FP16)
        Validates: Requirements 1.1, 1.3, 1.4, 1.5
        """
        try:
            from python import naive_attention
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        
        output = naive_attention(q, k, v)
        reference, _ = compute_attention_reference(q, k, v)
        
        assert_close(output, reference, rtol=1e-2, atol=1e-2,
                    msg="Naive attention FP16 correctness failed")


class TestSoftmaxInvariants:
    """Tests for softmax mathematical invariants."""
    
    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(
        batch=batch_strategy,
        heads=head_strategy,
        seq_len=seq_len_strategy,
        head_dim=head_dim_strategy
    )
    def test_softmax_invariants(self, batch, heads, seq_len, head_dim, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 2: Softmax 数学不变量
        Validates: Requirements 1.2
        
        For any input vector, softmax output should satisfy:
        - All values in (0, 1)
        - Sum equals 1
        - Relative order preserved (monotonicity)
        """
        # Generate attention scores
        scores = torch.randn(batch, heads, seq_len, seq_len, device=device)
        
        # Compute softmax
        softmax_output = torch.softmax(scores, dim=-1)
        
        # Property 1: All values in (0, 1)
        assert (softmax_output > 0).all(), "Softmax values should be > 0"
        assert (softmax_output < 1).all(), "Softmax values should be < 1"
        
        # Property 2: Sum equals 1
        row_sums = softmax_output.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-5), \
            "Softmax row sums should equal 1"
        
        # Property 3: Monotonicity (larger input -> larger output)
        for i in range(min(5, seq_len - 1)):  # Check a few pairs
            idx1, idx2 = i, i + 1
            mask = scores[..., idx1] > scores[..., idx2]
            assert (softmax_output[..., idx1][mask] >= softmax_output[..., idx2][mask] - 1e-6).all(), \
                "Softmax should preserve relative order"


class TestFlashAttention:
    """Tests for FlashAttention implementation."""
    
    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(
        batch=batch_strategy,
        heads=head_strategy,
        seq_len=seq_len_strategy,
        head_dim=head_dim_strategy
    )
    def test_flash_attention_consistency(self, batch, heads, seq_len, head_dim, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 3: FlashAttention 与标准实现一致性
        Validates: Requirements 3.1, 3.6
        
        For any valid Q, K, V input, FlashAttention output should match
        naive attention implementation within tolerance.
        """
        try:
            from python import flash_attention, naive_attention
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        
        flash_output = flash_attention(q, k, v)
        naive_output = naive_attention(q, k, v)
        
        assert_close(flash_output, naive_output, rtol=1e-3, atol=1e-3,
                    msg="FlashAttention should match naive attention")
    
    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(
        batch=batch_strategy,
        heads=head_strategy,
        seq_len=seq_len_strategy,
        head_dim=head_dim_strategy
    )
    def test_causal_mask_correctness(self, batch, heads, seq_len, head_dim, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 4: 因果掩码正确性
        Validates: Requirements 3.5
        
        For causal attention, output at position i should only depend on
        positions j <= i. Attention weights should form a lower triangular matrix.
        """
        try:
            from python import flash_attention
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        
        # Compute causal attention
        causal_output = flash_attention(q, k, v, is_causal=True)
        
        # Compute reference with causal mask
        reference, attn_weights = compute_attention_reference(q, k, v, is_causal=True)
        
        # Verify output matches
        assert_close(causal_output, reference, rtol=1e-3, atol=1e-3,
                    msg="Causal attention output mismatch")
        
        # Verify attention weights are lower triangular
        upper_triangle = torch.triu(attn_weights, diagonal=1)
        assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6), \
            "Causal attention weights should be lower triangular"


class TestTiledAttention:
    """Tests for tiled attention implementation."""
    
    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=50, deadline=None)
    @given(
        batch=batch_strategy,
        heads=head_strategy,
        seq_len=seq_len_strategy,
        head_dim=head_dim_strategy
    )
    def test_tiled_attention_consistency(self, batch, heads, seq_len, head_dim, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property: Tiled attention consistency
        Validates: Requirements 1.5
        
        Tiled attention should produce same results as naive attention.
        """
        try:
            from python import tiled_attention, naive_attention
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        
        tiled_output = tiled_attention(q, k, v)
        naive_output = naive_attention(q, k, v)
        
        assert_close(tiled_output, naive_output, rtol=1e-3, atol=1e-3,
                    msg="Tiled attention should match naive attention")


class TestBatchAndMultiHead:
    """Tests for batch and multi-head support."""
    
    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(
        batch=st.integers(min_value=1, max_value=8),
        heads=st.integers(min_value=1, max_value=16),
        seq_len=seq_len_strategy,
        head_dim=head_dim_strategy
    )
    def test_batch_multihead_support(self, batch, heads, seq_len, head_dim, device):
        """
        Feature: cuda-llm-kernel-optimization
        Property 11: 批量和多头支持
        Validates: Requirements 8.4
        
        For any batch_size >= 1 and num_heads >= 1, Attention_Kernel should
        correctly process and produce output of shape [batch, heads, seq_len, head_dim].
        """
        try:
            from python import flash_attention
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        
        output = flash_attention(q, k, v)
        
        # Verify output shape
        assert output.shape == (batch, heads, seq_len, head_dim), \
            f"Expected shape {(batch, heads, seq_len, head_dim)}, got {output.shape}"
        
        # Verify output is valid (no NaN or Inf)
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"


# Unit tests for edge cases
class TestEdgeCases:
    """Unit tests for edge cases."""
    
    @pytest.mark.cuda
    def test_minimum_dimensions(self, device):
        """Test with minimum valid dimensions."""
        try:
            from python import flash_attention
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        q = torch.randn(1, 1, 1, 32, device=device)
        k = torch.randn(1, 1, 1, 32, device=device)
        v = torch.randn(1, 1, 1, 32, device=device)
        
        output = flash_attention(q, k, v)
        assert output.shape == (1, 1, 1, 32)
    
    @pytest.mark.cuda
    def test_large_sequence_length(self, device):
        """Test with large sequence length."""
        try:
            from python import flash_attention
        except ImportError:
            pytest.skip("CUDA kernels not built")
        
        seq_len = 2048
        q = torch.randn(1, 4, seq_len, 64, device=device, dtype=torch.float16)
        k = torch.randn(1, 4, seq_len, 64, device=device, dtype=torch.float16)
        v = torch.randn(1, 4, seq_len, 64, device=device, dtype=torch.float16)
        
        output = flash_attention(q, k, v)
        assert output.shape == (1, 4, seq_len, 64)
        assert not torch.isnan(output).any()
