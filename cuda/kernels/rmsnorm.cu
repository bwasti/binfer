#include "binfer.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Get current stream from gemm.cu (for graph capture)
extern "C" void* binfer_get_current_stream();
#define GET_CURRENT_STREAM() ((cudaStream_t)binfer_get_current_stream())

// RMSNorm kernel
// Each block handles one row (seq position)
// Uses warp reduction for efficiency
template<int BLOCK_SIZE>
__global__ void rmsnorm_kernel_f16(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    int hidden_size,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __half* row_input = input + row * hidden_size;
    __half* row_output = output + row * hidden_size;

    // Compute sum of squares using thread-local accumulator
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = __half2float(row_input[i]);
        sum_sq += val * val;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Block reduction using shared memory
    __shared__ float shared_sum[32];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) {
        shared_sum[warp_id] = sum_sq;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        sum_sq = (lane_id < (BLOCK_SIZE / 32)) ? shared_sum[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    // Broadcast result
    __shared__ float rsqrt_val;
    if (tid == 0) {
        rsqrt_val = rsqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();

    // Apply normalization and weight
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = __half2float(row_input[i]);
        float w = __half2float(weight[i]);
        row_output[i] = __float2half(val * rsqrt_val * w);
    }
}

extern "C" BinferError binfer_rmsnorm_f16(
    const void* input,
    const void* weight,
    void* output,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps
) {
    const int total_rows = batch_size * seq_len;
    cudaStream_t stream = GET_CURRENT_STREAM();

    // Choose block size based on hidden size
    if (hidden_size <= 1024) {
        rmsnorm_kernel_f16<256><<<total_rows, 256, 0, stream>>>(
            (const __half*)input,
            (const __half*)weight,
            (__half*)output,
            hidden_size,
            eps
        );
    } else if (hidden_size <= 4096) {
        rmsnorm_kernel_f16<512><<<total_rows, 512, 0, stream>>>(
            (const __half*)input,
            (const __half*)weight,
            (__half*)output,
            hidden_size,
            eps
        );
    } else {
        rmsnorm_kernel_f16<1024><<<total_rows, 1024, 0, stream>>>(
            (const __half*)input,
            (const __half*)weight,
            (__half*)output,
            hidden_size,
            eps
        );
    }

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}
