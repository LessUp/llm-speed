#!/usr/bin/env python3
"""
Benchmark script for attention kernels.
Compares custom implementations with PyTorch/cuDNN reference.
"""

import argparse
import torch
import time
from typing import List, Dict, Tuple
import sys
sys.path.insert(0, '..')


def benchmark_kernel(
    func,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup: int = 10,
    iterations: int = 100,
    **kwargs
) -> float:
    """Benchmark a kernel and return average time in ms."""
    # Warmup
    for _ in range(warmup):
        func(q, k, v, **kwargs)
    
    torch.cuda.synchronize()
    
    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        func(q, k, v, **kwargs)
    end.record()
    
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / iterations


def compute_attention_flops(batch: int, heads: int, seq_len: int, head_dim: int) -> int:
    """Compute FLOPs for attention."""
    # Q @ K^T: 2 * batch * heads * seq * seq * head_dim
    # Softmax: ~5 * batch * heads * seq * seq
    # P @ V: 2 * batch * heads * seq * seq * head_dim
    return batch * heads * (4 * seq_len * seq_len * head_dim + 5 * seq_len * seq_len)


def benchmark_attention(
    seq_lengths: List[int] = [512, 1024, 2048, 4096],
    batch_size: int = 1,
    num_heads: int = 32,
    head_dim: int = 128,
    dtype: torch.dtype = torch.float16,
    warmup: int = 10,
    iterations: int = 100
) -> List[Dict]:
    """Benchmark attention implementations."""
    try:
        from python import naive_attention, tiled_attention, flash_attention
        has_custom = True
    except ImportError:
        print("Warning: Custom kernels not built. Only benchmarking PyTorch.")
        has_custom = False
    
    results = []
    
    for seq_len in seq_lengths:
        print(f"\nBenchmarking seq_len={seq_len}...")
        
        # Create inputs
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       device='cuda', dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        result = {
            'seq_len': seq_len,
            'batch_size': batch_size,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'dtype': str(dtype),
        }
        
        # PyTorch reference (uses cuDNN Flash Attention when available)
        def pytorch_attention(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)
        
        pytorch_time = benchmark_kernel(pytorch_attention, q, k, v, warmup, iterations)
        result['pytorch_ms'] = pytorch_time
        
        flops = compute_attention_flops(batch_size, num_heads, seq_len, head_dim)
        result['pytorch_tflops'] = (flops / 1e12) / (pytorch_time / 1000)
        
        if has_custom:
            # Naive attention
            try:
                naive_time = benchmark_kernel(naive_attention, q, k, v, warmup, iterations)
                result['naive_ms'] = naive_time
                result['naive_tflops'] = (flops / 1e12) / (naive_time / 1000)
                result['naive_speedup'] = pytorch_time / naive_time
            except Exception as e:
                print(f"  Naive attention failed: {e}")
                result['naive_ms'] = float('inf')
            
            # Tiled attention
            try:
                tiled_time = benchmark_kernel(tiled_attention, q, k, v, warmup, iterations)
                result['tiled_ms'] = tiled_time
                result['tiled_tflops'] = (flops / 1e12) / (tiled_time / 1000)
                result['tiled_speedup'] = pytorch_time / tiled_time
            except Exception as e:
                print(f"  Tiled attention failed: {e}")
                result['tiled_ms'] = float('inf')
            
            # Flash attention
            try:
                flash_time = benchmark_kernel(flash_attention, q, k, v, warmup, iterations)
                result['flash_ms'] = flash_time
                result['flash_tflops'] = (flops / 1e12) / (flash_time / 1000)
                result['flash_speedup'] = pytorch_time / flash_time
            except Exception as e:
                print(f"  Flash attention failed: {e}")
                result['flash_ms'] = float('inf')
        
        results.append(result)
    
    return results


def print_results(results: List[Dict]):
    """Pretty print benchmark results."""
    print("\n" + "="*80)
    print("ATTENTION BENCHMARK RESULTS")
    print("="*80)
    
    # Header
    print(f"\n{'Seq Len':>8} | {'PyTorch':>10} | {'Naive':>10} | {'Tiled':>10} | {'Flash':>10} | {'Best Speedup':>12}")
    print(f"{'':>8} | {'(ms)':>10} | {'(ms)':>10} | {'(ms)':>10} | {'(ms)':>10} | {'':>12}")
    print("-"*80)
    
    for r in results:
        pytorch_ms = r.get('pytorch_ms', float('inf'))
        naive_ms = r.get('naive_ms', float('inf'))
        tiled_ms = r.get('tiled_ms', float('inf'))
        flash_ms = r.get('flash_ms', float('inf'))
        
        best_custom = min(naive_ms, tiled_ms, flash_ms)
        speedup = pytorch_ms / best_custom if best_custom < float('inf') else 0
        
        print(f"{r['seq_len']:>8} | {pytorch_ms:>10.3f} | {naive_ms:>10.3f} | {tiled_ms:>10.3f} | {flash_ms:>10.3f} | {speedup:>12.2f}x")
    
    print("\n" + "="*80)
    print("TFLOPS COMPARISON")
    print("="*80)
    
    print(f"\n{'Seq Len':>8} | {'PyTorch':>12} | {'Naive':>12} | {'Tiled':>12} | {'Flash':>12}")
    print(f"{'':>8} | {'(TFLOPS)':>12} | {'(TFLOPS)':>12} | {'(TFLOPS)':>12} | {'(TFLOPS)':>12}")
    print("-"*80)
    
    for r in results:
        pytorch_tflops = r.get('pytorch_tflops', 0)
        naive_tflops = r.get('naive_tflops', 0)
        tiled_tflops = r.get('tiled_tflops', 0)
        flash_tflops = r.get('flash_tflops', 0)
        
        print(f"{r['seq_len']:>8} | {pytorch_tflops:>12.2f} | {naive_tflops:>12.2f} | {tiled_tflops:>12.2f} | {flash_tflops:>12.2f}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark attention kernels')
    parser.add_argument('--seq-lengths', type=int, nargs='+', 
                       default=[512, 1024, 2048, 4096],
                       help='Sequence lengths to benchmark')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--num-heads', type=int, default=32,
                       help='Number of attention heads')
    parser.add_argument('--head-dim', type=int, default=128,
                       help='Head dimension')
    parser.add_argument('--dtype', type=str, default='fp16',
                       choices=['fp16', 'fp32'],
                       help='Data type')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Benchmark iterations')
    
    args = parser.parse_args()
    
    dtype = torch.float16 if args.dtype == 'fp16' else torch.float32
    
    print(f"Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num heads: {args.num_heads}")
    print(f"  Head dim: {args.head_dim}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Iterations: {args.iterations}")
    
    results = benchmark_attention(
        seq_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        warmup=args.warmup,
        iterations=args.iterations
    )
    
    print_results(results)


if __name__ == '__main__':
    main()
