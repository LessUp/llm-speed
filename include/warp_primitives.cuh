#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Warp-level reduction for sum
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level reduction for max
template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for min
template<typename T>
__device__ __forceinline__ T warp_reduce_min(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = min(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level reduction for sum using shared memory
template<typename T, int BLOCK_SIZE>
__device__ __forceinline__ T block_reduce_sum(T val, T* smem) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    // Warp-level reduction
    val = warp_reduce_sum(val);
    
    // Write warp results to shared memory
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction in first warp
    constexpr int num_warps = BLOCK_SIZE / 32;
    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : T(0);
        val = warp_reduce_sum(val);
    }
    
    return val;
}

// Block-level reduction for max using shared memory
template<typename T, int BLOCK_SIZE>
__device__ __forceinline__ T block_reduce_max(T val, T* smem) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    // Warp-level reduction
    val = warp_reduce_max(val);
    
    // Write warp results to shared memory
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction in first warp
    constexpr int num_warps = BLOCK_SIZE / 32;
    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : T(-1e30f);
        val = warp_reduce_max(val);
    }
    
    return val;
}

// Warp shuffle broadcast
template<typename T>
__device__ __forceinline__ T warp_broadcast(T val, int src_lane) {
    return __shfl_sync(0xffffffff, val, src_lane);
}

// Warp shuffle exchange (butterfly pattern)
template<typename T>
__device__ __forceinline__ T warp_shuffle_xor(T val, int mask) {
    return __shfl_xor_sync(0xffffffff, val, mask);
}
