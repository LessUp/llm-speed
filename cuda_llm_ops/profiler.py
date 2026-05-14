"""
Performance profiler for CUDA kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch


class Bottleneck(Enum):
    COMPUTE_BOUND = "compute_bound"
    MEMORY_BOUND = "memory_bound"
    LATENCY_BOUND = "latency_bound"


@dataclass
class KernelMetrics:
    """Performance metrics for a kernel execution."""

    elapsed_ms: float
    tflops: float
    memory_bandwidth_gb: float
    sm_occupancy: float = 0.0
    l2_hit_rate: float = 0.0
    bottleneck: Bottleneck = Bottleneck.MEMORY_BOUND


class CUDAProfiler:
    """CUDA kernel profiler using CUDA events."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def measure_time(
        self, func: Callable, *args, warmup: int = 10, iterations: int = 100, **kwargs
    ) -> float:
        """Measure kernel execution time in milliseconds."""
        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)

        torch.cuda.synchronize()

        # Measure
        self.start_event.record()
        for _ in range(iterations):
            func(*args, **kwargs)
        self.end_event.record()

        torch.cuda.synchronize()

        total_time = self.start_event.elapsed_time(self.end_event)
        return total_time / iterations

    def profile_attention(
        self,
        func: Callable,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        warmup: int = 10,
        iterations: int = 100,
    ) -> KernelMetrics:
        """Profile attention kernel and compute metrics."""
        # Create inputs
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Measure time
        elapsed_ms = self.measure_time(func, q, k, v, warmup=warmup, iterations=iterations)

        # Compute FLOPs
        # Attention: 2 * batch * heads * seq^2 * head_dim (Q@K^T)
        #          + 2 * batch * heads * seq^2 * head_dim (softmax@V)
        #          + seq^2 (softmax)
        flops = (
            batch_size
            * num_heads
            * (
                4 * seq_len * seq_len * head_dim  # Q@K^T and P@V
                + 5 * seq_len * seq_len  # softmax (exp, sum, div)
            )
        )
        tflops = (flops / 1e12) / (elapsed_ms / 1000)

        # Compute memory bandwidth
        # Read: Q, K, V (3 * batch * heads * seq * head_dim)
        # Write: O (batch * heads * seq * head_dim)
        bytes_per_elem = 2 if dtype == torch.float16 else 4
        memory_bytes = batch_size * num_heads * seq_len * head_dim * 4 * bytes_per_elem
        memory_bandwidth_gb = (memory_bytes / 1e9) / (elapsed_ms / 1000)

        # Determine bottleneck (simplified heuristic)
        # Typical GPU: ~300 TFLOPS FP16, ~2 TB/s memory bandwidth
        compute_intensity = flops / memory_bytes  # FLOPs per byte
        if compute_intensity > 100:
            bottleneck = Bottleneck.COMPUTE_BOUND
        else:
            bottleneck = Bottleneck.MEMORY_BOUND

        return KernelMetrics(
            elapsed_ms=elapsed_ms,
            tflops=tflops,
            memory_bandwidth_gb=memory_bandwidth_gb,
            bottleneck=bottleneck,
        )

    def profile_gemm(
        self,
        func: Callable,
        M: int,
        N: int,
        K: int,
        dtype: torch.dtype = torch.float16,
        warmup: int = 10,
        iterations: int = 100,
    ) -> KernelMetrics:
        """Profile GEMM kernel and compute metrics."""
        # Create inputs
        a = torch.randn(M, K, device="cuda", dtype=dtype)
        b = torch.randn(K, N, device="cuda", dtype=dtype)

        # Measure time
        elapsed_ms = self.measure_time(func, a, b, warmup=warmup, iterations=iterations)

        # Compute FLOPs: 2 * M * N * K (multiply-add)
        flops = 2 * M * N * K
        tflops = (flops / 1e12) / (elapsed_ms / 1000)

        # Compute memory bandwidth
        bytes_per_elem = 2 if dtype == torch.float16 else 4
        memory_bytes = (M * K + K * N + M * N) * bytes_per_elem
        memory_bandwidth_gb = (memory_bytes / 1e9) / (elapsed_ms / 1000)

        # Determine bottleneck
        compute_intensity = flops / memory_bytes
        if compute_intensity > 100:
            bottleneck = Bottleneck.COMPUTE_BOUND
        else:
            bottleneck = Bottleneck.MEMORY_BOUND

        return KernelMetrics(
            elapsed_ms=elapsed_ms,
            tflops=tflops,
            memory_bandwidth_gb=memory_bandwidth_gb,
            bottleneck=bottleneck,
        )

    def compare_with_reference(
        self,
        custom_func: Callable,
        reference_func: Callable,
        *args,
        warmup: int = 10,
        iterations: int = 100,
        **kwargs,
    ) -> dict[str, float]:
        """Compare custom kernel with reference implementation."""
        custom_time = self.measure_time(
            custom_func, *args, warmup=warmup, iterations=iterations, **kwargs
        )
        reference_time = self.measure_time(
            reference_func, *args, warmup=warmup, iterations=iterations, **kwargs
        )

        return {
            "custom_ms": custom_time,
            "reference_ms": reference_time,
            "speedup": reference_time / custom_time,
            "relative_perf": custom_time / reference_time,
        }


def print_benchmark_results(results: list[dict], title: str = "Benchmark Results"):
    """Pretty print benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}")

    for r in results:
        print(f"\nConfiguration: {r}")
        if "speedup" in r:
            print(f"  Speedup: {r['speedup']:.2f}x")
        if "relative_perf" in r:
            print(f"  Relative Performance: {r['relative_perf'] * 100:.1f}% of cuBLAS")
        if "bottleneck" in r:
            print(f"  Bottleneck: {r['bottleneck']}")
