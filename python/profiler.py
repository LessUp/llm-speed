"""
Performance profiler for CUDA kernels.
"""

import torch
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from enum import Enum


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
        self, 
        func: Callable, 
        *args, 
        warmup: int = 10, 
        iterations: int = 100,
        **kwargs
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
        iterations: int = 100
    ) -> KernelMetrics:
        """Profile attention kernel and compute metrics."""
        # Create inputs
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       device='cuda', dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        # Measure time
        elapsed_ms = self.measure_time(func, q, k, v, warmup=warmup, iterations=iterations)
        
        # Compute FLOPs
        # Attention: 2 * batch * heads * seq^2 * head_dim (Q@K^T) 
        #          + 2 * batch * heads * seq^2 * head_dim (softmax@V)
        #          + seq^2 (softmax)
        flops = batch_size * num_heads * (
            4 * seq_len * seq_len * head_dim +  # Q@K^T and P@V
            5 * seq_len * seq_len  # softmax (exp, sum, div)
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
            bottleneck=bottleneck
        )
    
    def profile_gemm(
        self,
        func: Callable,
        M: int,
        N: int,
        K: int,
        dtype: torch.dtype = torch.float16,
        warmup: int = 10,
        iterations: int = 100
    ) -> KernelMetrics:
        """Profile GEMM kernel and compute metrics."""
        # Create inputs
        a = torch.randn(M, K, device='cuda', dtype=dtype)
        b = torch.randn(K, N, device='cuda', dtype=dtype)
        
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
            bottleneck=bottleneck
        )
    
    def compare_with_reference(
        self,
        custom_func: Callable,
        reference_func: Callable,
        *args,
        warmup: int = 10,
        iterations: int = 100,
        **kwargs
    ) -> Dict[str, float]:
        """Compare custom kernel with reference implementation."""
        custom_time = self.measure_time(custom_func, *args, warmup=warmup, 
                                        iterations=iterations, **kwargs)
        reference_time = self.measure_time(reference_func, *args, warmup=warmup,
                                           iterations=iterations, **kwargs)
        
        return {
            'custom_ms': custom_time,
            'reference_ms': reference_time,
            'speedup': reference_time / custom_time,
            'relative_perf': custom_time / reference_time
        }


def benchmark_attention(
    seq_lengths: List[int] = [512, 1024, 2048, 4096],
    batch_size: int = 1,
    num_heads: int = 32,
    head_dim: int = 128,
    dtype: torch.dtype = torch.float16
) -> List[Dict]:
    """Benchmark attention kernels across different sequence lengths."""
    from . import flash_attention
    
    profiler = CUDAProfiler()
    results = []
    
    for seq_len in seq_lengths:
        # Profile custom implementation
        custom_metrics = profiler.profile_attention(
            flash_attention, batch_size, num_heads, seq_len, head_dim, dtype
        )
        
        # Profile PyTorch reference
        def pytorch_attention(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)
        
        ref_metrics = profiler.profile_attention(
            pytorch_attention, batch_size, num_heads, seq_len, head_dim, dtype
        )
        
        results.append({
            'seq_len': seq_len,
            'custom_ms': custom_metrics.elapsed_ms,
            'custom_tflops': custom_metrics.tflops,
            'reference_ms': ref_metrics.elapsed_ms,
            'reference_tflops': ref_metrics.tflops,
            'speedup': ref_metrics.elapsed_ms / custom_metrics.elapsed_ms,
            'bottleneck': custom_metrics.bottleneck.value
        })
    
    return results


def benchmark_gemm(
    sizes: List[Tuple[int, int, int]] = [(1024, 1024, 1024), (2048, 2048, 2048), 
                                          (4096, 4096, 4096)],
    dtype: torch.dtype = torch.float16
) -> List[Dict]:
    """Benchmark GEMM kernels across different matrix sizes."""
    from . import gemm
    
    profiler = CUDAProfiler()
    results = []
    
    for M, N, K in sizes:
        # Profile custom implementation
        custom_metrics = profiler.profile_gemm(gemm, M, N, K, dtype)
        
        # Profile PyTorch reference (cuBLAS)
        def pytorch_gemm(a, b):
            return torch.matmul(a, b)
        
        ref_metrics = profiler.profile_gemm(pytorch_gemm, M, N, K, dtype)
        
        results.append({
            'M': M, 'N': N, 'K': K,
            'custom_ms': custom_metrics.elapsed_ms,
            'custom_tflops': custom_metrics.tflops,
            'reference_ms': ref_metrics.elapsed_ms,
            'reference_tflops': ref_metrics.tflops,
            'relative_perf': custom_metrics.tflops / ref_metrics.tflops,
            'bottleneck': custom_metrics.bottleneck.value
        })
    
    return results


def print_benchmark_results(results: List[Dict], title: str = "Benchmark Results"):
    """Pretty print benchmark results."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for r in results:
        print(f"\nConfiguration: {r}")
        if 'speedup' in r:
            print(f"  Speedup: {r['speedup']:.2f}x")
        if 'relative_perf' in r:
            print(f"  Relative Performance: {r['relative_perf']*100:.1f}% of cuBLAS")
        if 'bottleneck' in r:
            print(f"  Bottleneck: {r['bottleneck']}")
