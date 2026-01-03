"""
Pytest configuration and fixtures for CUDA kernel tests.
"""

import pytest
import torch
import numpy as np
from typing import Tuple


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "property: mark test as property-based test")


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def device(cuda_available):
    """Get the device to use for tests."""
    if cuda_available:
        return torch.device("cuda")
    pytest.skip("CUDA not available")


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def attention_inputs(device):
    """Generate random attention inputs."""
    def _generate(batch_size=2, num_heads=4, seq_len=64, head_dim=32, dtype=torch.float32):
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        return q, k, v
    return _generate


@pytest.fixture
def gemm_inputs(device):
    """Generate random GEMM inputs."""
    def _generate(M=64, N=64, K=64, dtype=torch.float32):
        a = torch.randn(M, K, device=device, dtype=dtype)
        b = torch.randn(K, N, device=device, dtype=dtype)
        return a, b
    return _generate


def assert_close(actual, expected, rtol=1e-3, atol=1e-3, msg=""):
    """Assert that two tensors are close."""
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = (actual - expected).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        raise AssertionError(
            f"{msg}\nMax diff: {max_diff}, Mean diff: {mean_diff}, "
            f"rtol: {rtol}, atol: {atol}"
        )


def compute_attention_reference(q, k, v, scale=None, is_causal=False):
    """Compute attention using PyTorch reference implementation."""
    if scale is None:
        scale = 1.0 / (q.size(-1) ** 0.5)
    
    # Q @ K^T
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Causal mask
    if is_causal:
        seq_len = q.size(-2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Attention @ V
    output = torch.matmul(attn_weights, v)
    
    return output, attn_weights
