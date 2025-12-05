cmake_minimum_required(VERSION 3.20)
project(binfer_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# H100-specific optimizations (sm_90)
set(CMAKE_CUDA_ARCHITECTURES 90)

# cuBLAS
find_library(CUBLAS_LIBRARY cublas HINTS ${CUDAToolkit_LIBRARY_DIR})

# Source files
set(KERNEL_SOURCES
    kernels/gemm.cu
    kernels/rmsnorm.cu
    kernels/rotary.cu
    kernels/activation.cu
    kernels/attention.cu
    kernels/paged_attention.cu
    kernels/bf16_kernels.cu
    kernels/moe_kernels.cu
    kernels/moe_fused.cu
    kernels/moe_optimized.cu
    kernels/moe_cutlass.cu
    kernels/moe_marlin.cu
    kernels/gds.cu
)

# Build shared library for FFI
add_library(binfer_cuda SHARED
    ${KERNEL_SOURCES}
)

target_include_directories(binfer_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass/tools/util/include
)

target_link_libraries(binfer_cuda
    CUDA::cudart
    CUDA::cuda_driver
    ${CUBLAS_LIBRARY}
    ${CMAKE_DL_LIBS}
)

# Install
install(TARGETS binfer_cuda
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES include/binfer.h
    DESTINATION include
)
