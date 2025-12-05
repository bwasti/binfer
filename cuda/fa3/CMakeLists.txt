# FlashAttention-3 Build Configuration for Binfer
cmake_minimum_required(VERSION 3.20)
project(binfer_fa3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Hopper architecture (sm_90a for TMA/GMMA features)
set(CMAKE_CUDA_ARCHITECTURES 90a)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Path to flash-attention hopper directory (submodule)
set(FA3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/flash-attention/hopper")
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/flash-attention/csrc/cutlass")

# Check paths exist
if(NOT EXISTS "${FA3_DIR}")
    message(FATAL_ERROR "FlashAttention-3 not found at ${FA3_DIR}")
endif()
if(NOT EXISTS "${CUTLASS_DIR}/include")
    message(FATAL_ERROR "CUTLASS not found at ${CUTLASS_DIR}")
endif()

# FA3 kernel instantiation sources (only the ones we need)
set(FA3_KERNEL_SOURCES
    ${FA3_DIR}/instantiations/flash_fwd_hdim64_bf16_sm90.cu
    ${FA3_DIR}/instantiations/flash_fwd_hdim96_bf16_sm90.cu
    ${FA3_DIR}/instantiations/flash_fwd_hdim128_bf16_sm90.cu
    ${FA3_DIR}/instantiations/flash_fwd_hdim192_bf16_sm90.cu
    ${FA3_DIR}/instantiations/flash_fwd_hdim256_bf16_sm90.cu
    ${FA3_DIR}/instantiations/flash_fwd_hdim64_fp16_sm90.cu
    ${FA3_DIR}/instantiations/flash_fwd_hdim96_fp16_sm90.cu
    ${FA3_DIR}/instantiations/flash_fwd_hdim128_fp16_sm90.cu
    ${FA3_DIR}/instantiations/flash_fwd_hdim192_fp16_sm90.cu
    ${FA3_DIR}/instantiations/flash_fwd_hdim256_fp16_sm90.cu
)

# Our API shim
set(FA3_API_SOURCES
    fa3_api.cu
)

# Build shared library
add_library(binfer_fa3 SHARED
    ${FA3_API_SOURCES}
    ${FA3_KERNEL_SOURCES}
)

# Include directories
target_include_directories(binfer_fa3 PUBLIC
    ${FA3_DIR}
    ${CUTLASS_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Compile options for FA3
target_compile_options(binfer_fa3 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -O3
        -lineinfo
    >
)

# Link libraries
target_link_libraries(binfer_fa3
    CUDA::cudart
)

# Disable unused instantiation warnings
target_compile_definitions(binfer_fa3 PRIVATE
    FLASHATTENTION_DISABLE_HDIM32
    FLASHATTENTION_DISABLE_HDIMDIFF64
    FLASHATTENTION_DISABLE_HDIMDIFF192
)

# Install
install(TARGETS binfer_fa3
    LIBRARY DESTINATION lib
)
