#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Pipeline scheduler for double/multi-buffering
template<int NUM_STAGES>
class PipelineScheduler {
public:
    struct StageBuffer {
        void* smem_ptr;
        int tile_idx;
        bool valid;
    };
    
    StageBuffer buffers[NUM_STAGES];
    int current_stage;
    int compute_stage;
    
    __device__ __forceinline__ PipelineScheduler() 
        : current_stage(0), compute_stage(0) {
        #pragma unroll
        for (int i = 0; i < NUM_STAGES; i++) {
            buffers[i].smem_ptr = nullptr;
            buffers[i].tile_idx = -1;
            buffers[i].valid = false;
        }
    }
    
    __device__ __forceinline__ void set_buffer(int stage, void* ptr) {
        buffers[stage].smem_ptr = ptr;
    }
    
    __device__ __forceinline__ int get_load_stage() const {
        return current_stage;
    }
    
    __device__ __forceinline__ int get_compute_stage() const {
        return compute_stage;
    }
    
    __device__ __forceinline__ void* get_load_buffer() {
        return buffers[current_stage].smem_ptr;
    }
    
    __device__ __forceinline__ void* get_compute_buffer() {
        return buffers[compute_stage].smem_ptr;
    }
    
    __device__ __forceinline__ void advance_load() {
        buffers[current_stage].valid = true;
        current_stage = (current_stage + 1) % NUM_STAGES;
    }
    
    __device__ __forceinline__ void advance_compute() {
        buffers[compute_stage].valid = false;
        compute_stage = (compute_stage + 1) % NUM_STAGES;
    }
    
    __device__ __forceinline__ bool is_compute_ready() const {
        return buffers[compute_stage].valid;
    }
};

// Double buffering helper for attention
template<typename T, int BLOCK_SIZE>
struct DoubleBuffer {
    T* buffer[2];
    int current;
    
    __device__ __forceinline__ DoubleBuffer(T* buf0, T* buf1) 
        : current(0) {
        buffer[0] = buf0;
        buffer[1] = buf1;
    }
    
    __device__ __forceinline__ T* get_load_buffer() {
        return buffer[current];
    }
    
    __device__ __forceinline__ T* get_compute_buffer() {
        return buffer[1 - current];
    }
    
    __device__ __forceinline__ void swap() {
        current = 1 - current;
    }
};

// Async copy helpers (for Ampere+)
#if __CUDA_ARCH__ >= 800

template<int BYTES>
__device__ __forceinline__ void async_copy(void* dst, const void* src) {
    static_assert(BYTES == 4 || BYTES == 8 || BYTES == 16, 
                  "async_copy only supports 4, 8, or 16 bytes");
    
    if constexpr (BYTES == 16) {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(static_cast<unsigned>(reinterpret_cast<uintptr_t>(dst))),
               "l"(src)
        );
    } else if constexpr (BYTES == 8) {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 8;\n"
            :: "r"(static_cast<unsigned>(reinterpret_cast<uintptr_t>(dst))),
               "l"(src)
        );
    } else {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4;\n"
            :: "r"(static_cast<unsigned>(reinterpret_cast<uintptr_t>(dst))),
               "l"(src)
        );
    }
}

__device__ __forceinline__ void async_copy_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__ void async_copy_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void async_copy_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

#else

// Fallback for older architectures
template<int BYTES>
__device__ __forceinline__ void async_copy(void* dst, const void* src) {
    if constexpr (BYTES == 16) {
        *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
    } else if constexpr (BYTES == 8) {
        *reinterpret_cast<float2*>(dst) = *reinterpret_cast<const float2*>(src);
    } else {
        *reinterpret_cast<float*>(dst) = *reinterpret_cast<const float*>(src);
    }
}

__device__ __forceinline__ void async_copy_commit() {}

template<int N>
__device__ __forceinline__ void async_copy_wait() {
    __syncthreads();
}

__device__ __forceinline__ void async_copy_wait_all() {
    __syncthreads();
}

#endif

// Software pipelining helper
template<int STAGES, typename LoadFunc, typename ComputeFunc>
__device__ void software_pipeline(
    int num_iterations,
    LoadFunc load_fn,
    ComputeFunc compute_fn
) {
    // Prologue: fill pipeline
    #pragma unroll
    for (int i = 0; i < STAGES - 1 && i < num_iterations; i++) {
        load_fn(i, i % STAGES);
        async_copy_commit();
    }
    
    // Main loop
    for (int i = 0; i < num_iterations; i++) {
        // Wait for data
        async_copy_wait<STAGES - 2>();
        __syncthreads();
        
        // Compute on current stage
        int compute_stage = i % STAGES;
        compute_fn(i, compute_stage);
        
        // Load next tile (if available)
        int load_iter = i + STAGES - 1;
        if (load_iter < num_iterations) {
            int load_stage = load_iter % STAGES;
            load_fn(load_iter, load_stage);
            async_copy_commit();
        }
        
        __syncthreads();
    }
    
    // Epilogue: drain pipeline
    async_copy_wait_all();
}
