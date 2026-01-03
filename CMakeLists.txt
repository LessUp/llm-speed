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

# pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

# Source files
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")

# Main library
add_library(cuda_kernels SHARED ${CUDA_SOURCES} ${CPP_SOURCES})
target_link_libraries(cuda_kernels CUDA::cudart CUDA::cublas)

# Python module
pybind11_add_module(cuda_llm_ops python/bindings.cpp ${CUDA_SOURCES})
target_link_libraries(cuda_llm_ops PRIVATE CUDA::cudart CUDA::cublas)

# Install
install(TARGETS cuda_kernels LIBRARY DESTINATION lib)
install(TARGETS cuda_llm_ops LIBRARY DESTINATION python)
