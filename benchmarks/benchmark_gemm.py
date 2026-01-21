#!/usr/bin/env python3
"""
Benchmark script for GEMM kernels.
Compares custom implementations with cuBLAS reference.
"""

import argparse
import torch
import time
from typing import List, Dict, Tuple
import sys
sys.path.insert(0, '..')


def benchmark_kernel(
    func,
    a: torch.Tensor,
    b: torch.Tensor,
    warmup: int = 10,
    iterations: int = 100,
    **kwargs
) -> float:
    """Benchmark a kernel and return average time in ms."""
    # Warmup
    for _ in range(warmup):
        func(a, b, **kwargs)
    
    torch.cuda.synchronize()
    
    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        func(a, b, **kwargs)
    end.record()
    
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / iterations


def compute_gemm_flops(M: int, N: int, K: int) -> int:
    """Compute FLOPs for GEMM: 2 * M * N * K (multiply-add)."""
    return 2 * M * N * K


def benchmark_gemm(
    sizes: List[Tuple[int, int, int]] = [(1024, 1024, 1024), (2048, 2048, 2048), 
                                          (4096, 4096, 4096), (8192, 8192, 8192)],
    dtype: torch.dtype = torch.float16,
    warmup: int = 10,
    iterations: int = 100
) -> List[Dict]:
    """Benchmark GEMM implementations."""
    try:
        from cuda_llm_ops import gemm, tensor_core_gemm
        has_custom = True
    except ImportError:
        print("Warning: Custom kernels not built. Only benchmarking cuBLAS.")
        has_custom = False
    
    results = []
    
    for M, N, K in sizes:
        print(f"\nBenchmarking M={M}, N={N}, K={K}...")
        
        # Create inputs
        a = torch.randn(M, K, device='cuda', dtype=dtype)
        b = torch.randn(K, N, device='cuda', dtype=dtype)
        
        result = {
            'M': M, 'N': N, 'K': K,
            'dtype': str(dtype),
        }
        
        # cuBLAS reference (via PyTorch)
        def cublas_gemm(a, b):
            return torch.matmul(a, b)
        
        cublas_time = benchmark_kernel(cublas_gemm, a, b, warmup, iterations)
        result['cublas_ms'] = cublas_time
        
        flops = compute_gemm_flops(M, N, K)
        result['cublas_tflops'] = (flops / 1e12) / (cublas_time / 1000)
        
        if has_custom:
            # Custom GEMM
            try:
                custom_time = benchmark_kernel(gemm, a, b, warmup, iterations)
                result['custom_ms'] = custom_time
                result['custom_tflops'] = (flops / 1e12) / (custom_time / 1000)
                result['custom_relative'] = result['custom_tflops'] / result['cublas_tflops']
            except Exception as e:
                print(f"  Custom GEMM failed: {e}")
                result['custom_ms'] = float('inf')
                result['custom_relative'] = 0
            
            # Tensor Core GEMM (FP16 only)
            if dtype == torch.float16:
                try:
                    tc_time = benchmark_kernel(tensor_core_gemm, a, b, warmup, iterations)
                    result['tensor_core_ms'] = tc_time
                    result['tensor_core_tflops'] = (flops / 1e12) / (tc_time / 1000)
                    result['tensor_core_relative'] = result['tensor_core_tflops'] / result['cublas_tflops']
                except Exception as e:
                    print(f"  Tensor Core GEMM failed: {e}")
                    result['tensor_core_ms'] = float('inf')
                    result['tensor_core_relative'] = 0
        
        results.append(result)
    
    return results


def print_results(results: List[Dict]):
    """Pretty print benchmark results."""
    print("\n" + "="*100)
    print("GEMM BENCHMARK RESULTS")
    print("="*100)
    
    # Header
    print(f"\n{'Size':>20} | {'cuBLAS':>12} | {'Custom':>12} | {'TC GEMM':>12} | {'Custom %':>10} | {'TC %':>10}")
    print(f"{'(M x N x K)':>20} | {'(ms)':>12} | {'(ms)':>12} | {'(ms)':>12} | {'of cuBLAS':>10} | {'of cuBLAS':>10}")
    print("-"*100)
    
    for r in results:
        size_str = f"{r['M']}x{r['N']}x{r['K']}"
        cublas_ms = r.get('cublas_ms', float('inf'))
        custom_ms = r.get('custom_ms', float('inf'))
        tc_ms = r.get('tensor_core_ms', float('inf'))
        custom_rel = r.get('custom_relative', 0) * 100
        tc_rel = r.get('tensor_core_relative', 0) * 100
        
        tc_str = f"{tc_ms:.3f}" if tc_ms < float('inf') else "N/A"
        tc_rel_str = f"{tc_rel:.1f}%" if tc_rel > 0 else "N/A"
        
        print(f"{size_str:>20} | {cublas_ms:>12.3f} | {custom_ms:>12.3f} | {tc_str:>12} | {custom_rel:>9.1f}% | {tc_rel_str:>10}")
    
    print("\n" + "="*100)
    print("TFLOPS COMPARISON")
    print("="*100)
    
    print(f"\n{'Size':>20} | {'cuBLAS':>12} | {'Custom':>12} | {'TC GEMM':>12}")
    print(f"{'(M x N x K)':>20} | {'(TFLOPS)':>12} | {'(TFLOPS)':>12} | {'(TFLOPS)':>12}")
    print("-"*100)
    
    for r in results:
        size_str = f"{r['M']}x{r['N']}x{r['K']}"
        cublas_tflops = r.get('cublas_tflops', 0)
        custom_tflops = r.get('custom_tflops', 0)
        tc_tflops = r.get('tensor_core_tflops', 0)
        
        tc_str = f"{tc_tflops:.2f}" if tc_tflops > 0 else "N/A"
        
        print(f"{size_str:>20} | {cublas_tflops:>12.2f} | {custom_tflops:>12.2f} | {tc_str:>12}")
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    avg_custom_rel = sum(r.get('custom_relative', 0) for r in results) / len(results) * 100
    avg_tc_rel = sum(r.get('tensor_core_relative', 0) for r in results if r.get('tensor_core_relative', 0) > 0)
    tc_count = sum(1 for r in results if r.get('tensor_core_relative', 0) > 0)
    if tc_count > 0:
        avg_tc_rel = avg_tc_rel / tc_count * 100
    
    print(f"\nAverage Custom GEMM performance: {avg_custom_rel:.1f}% of cuBLAS")
    if tc_count > 0:
        print(f"Average Tensor Core GEMM performance: {avg_tc_rel:.1f}% of cuBLAS")
    
    # Check if we meet the 90% target
    meets_target = avg_custom_rel >= 90
    print(f"\nTarget (90% of cuBLAS): {'✓ MET' if meets_target else '✗ NOT MET'}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark GEMM kernels')
    parser.add_argument('--sizes', type=str, nargs='+',
                       default=['1024x1024x1024', '2048x2048x2048', 
                               '4096x4096x4096', '8192x8192x8192'],
                       help='Matrix sizes as MxNxK')
    parser.add_argument('--dtype', type=str, default='fp16',
                       choices=['fp16', 'fp32'],
                       help='Data type')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Benchmark iterations')
    
    args = parser.parse_args()
    
    # Parse sizes
    sizes = []
    for s in args.sizes:
        parts = s.split('x')
        if len(parts) == 3:
            sizes.append((int(parts[0]), int(parts[1]), int(parts[2])))
        else:
            print(f"Invalid size format: {s}, expected MxNxK")
            return
    
    dtype = torch.float16 if args.dtype == 'fp16' else torch.float32
    
    print(f"Configuration:")
    print(f"  Sizes: {sizes}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Iterations: {args.iterations}")
    
    results = benchmark_gemm(
        sizes=sizes,
        dtype=dtype,
        warmup=args.warmup,
        iterations=args.iterations
    )
    
    print_results(results)


if __name__ == '__main__':
    main()
