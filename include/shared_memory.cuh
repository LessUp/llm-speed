#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Shared memory tile with padding to avoid bank conflicts
template<typename T, int ROWS, int COLS, int PAD = 1>
struct SharedMemoryTile {
    T data[ROWS][COLS + PAD];
    
    __device__ __forceinline__ T& operator()(int row, int col) {
        return data[row][col];
    }
    
    __device__ __forceinline__ const T& operator()(int row, int col) const {
        return data[row][col];
    }
    
    __device__ __forceinline__ T* row_ptr(int row) {
        return data[row];
    }
};

// Coalesced load from global memory to shared memory
template<typename T, int BLOCK_SIZE>
__device__ __forceinline__ void load_tile_coalesced(
    const T* __restrict__ global_ptr,
    T* __restrict__ smem_ptr,
    int global_stride,
    int smem_stride,
    int rows,
    int cols,
    int tid
) {
    int total_elements = rows * cols;
    for (int i = tid; i < total_elements; i += BLOCK_SIZE) {
        int row = i / cols;
        int col = i % cols;
        smem_ptr[row * smem_stride + col] = global_ptr[row * global_stride + col];
    }
}

// Coalesced load with bounds checking
template<typename T, int BLOCK_SIZE>
__device__ __forceinline__ void load_tile_coalesced_bounded(
    const T* __restrict__ global_ptr,
    T* __restrict__ smem_ptr,
    int global_stride,
    int smem_stride,
    int rows,
    int cols,
    int max_rows,
    int max_cols,
    int row_offset,
    int col_offset,
    int tid,
    T default_val = T(0)
) {
    int total_elements = rows * cols;
    for (int i = tid; i < total_elements; i += BLOCK_SIZE) {
        int local_row = i / cols;
        int local_col = i % cols;
        int global_row = row_offset + local_row;
        int global_col = col_offset + local_col;
        
        if (global_row < max_rows && global_col < max_cols) {
            smem_ptr[local_row * smem_stride + local_col] = 
                global_ptr[global_row * global_stride + global_col];
        } else {
            smem_ptr[local_row * smem_stride + local_col] = default_val;
        }
    }
}

// Store tile from shared memory to global memory
template<typename T, int BLOCK_SIZE>
__device__ __forceinline__ void store_tile_coalesced(
    const T* __restrict__ smem_ptr,
    T* __restrict__ global_ptr,
    int smem_stride,
    int global_stride,
    int rows,
    int cols,
    int tid
) {
    int total_elements = rows * cols;
    for (int i = tid; i < total_elements; i += BLOCK_SIZE) {
        int row = i / cols;
        int col = i % cols;
        global_ptr[row * global_stride + col] = smem_ptr[row * smem_stride + col];
    }
}

// Vectorized load (float4)
__device__ __forceinline__ void load_float4(
    const float* __restrict__ src,
    float* __restrict__ dst
) {
    float4 tmp = *reinterpret_cast<const float4*>(src);
    dst[0] = tmp.x;
    dst[1] = tmp.y;
    dst[2] = tmp.z;
    dst[3] = tmp.w;
}

// Vectorized store (float4)
__device__ __forceinline__ void store_float4(
    const float* __restrict__ src,
    float* __restrict__ dst
) {
    float4 tmp = make_float4(src[0], src[1], src[2], src[3]);
    *reinterpret_cast<float4*>(dst) = tmp;
}

// Vectorized load (half2)
__device__ __forceinline__ void load_half2(
    const half* __restrict__ src,
    half* __restrict__ dst
) {
    half2 tmp = *reinterpret_cast<const half2*>(src);
    dst[0] = __low2half(tmp);
    dst[1] = __high2half(tmp);
}

// Transpose tile in shared memory
template<typename T, int TILE_SIZE>
__device__ __forceinline__ void transpose_tile(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int src_stride,
    int dst_stride,
    int tid,
    int block_size
) {
    for (int i = tid; i < TILE_SIZE * TILE_SIZE; i += block_size) {
        int row = i / TILE_SIZE;
        int col = i % TILE_SIZE;
        dst[col * dst_stride + row] = src[row * src_stride + col];
    }
}
