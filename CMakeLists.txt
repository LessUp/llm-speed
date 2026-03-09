cmake_minimum_required(VERSION 3.18)
project(cuda_llm_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Find packages
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# pybind11 (pinned to v2.11.1)
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# ---------- Explicit source list (no GLOB_RECURSE) ----------
set(CUDA_SOURCES
    src/naive_attention.cu
    src/tiled_attention.cu
    src/flash_attention.cu
    src/hgemm_kernel.cu
    src/tensor_core_gemm.cu
)

# ---------- Main library ----------
add_library(cuda_kernels SHARED ${CUDA_SOURCES})

target_include_directories(cuda_kernels
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(cuda_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<PLATFORM_ID:Windows>>>:-Xcompiler=-fPIC>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -lineinfo>
)

target_link_libraries(cuda_kernels PRIVATE CUDA::cudart CUDA::cublas)

# ---------- Python module ----------
pybind11_add_module(cuda_llm_ops python/bindings.cpp ${CUDA_SOURCES})

target_include_directories(cuda_llm_ops
    PRIVATE ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(cuda_llm_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<PLATFORM_ID:Windows>>>:-Xcompiler=-fPIC>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -lineinfo>
)

target_link_libraries(cuda_llm_ops PRIVATE CUDA::cudart CUDA::cublas)

# ---------- Install ----------
install(TARGETS cuda_kernels LIBRARY DESTINATION lib)
install(TARGETS cuda_llm_ops LIBRARY DESTINATION python)
