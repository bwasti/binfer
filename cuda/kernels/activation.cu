#include "binfer.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// SiLU (Swish) activation: x * sigmoid(x)
__global__ void silu_kernel_f16(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    size_t numel
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float x = __half2float(input[idx]);
    float sigmoid_x = 1.0f / (1.0f + expf(-x));
    output[idx] = __float2half(x * sigmoid_x);
}

extern "C" BinferError binfer_silu_f16(
    const void* input,
    void* output,
    size_t numel
) {
    const int block_size = 256;
    const int num_blocks = (numel + block_size - 1) / block_size;

    silu_kernel_f16<<<num_blocks, block_size>>>(
        (const __half*)input,
        (__half*)output,
        numel
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// GELU activation (approximate)
__global__ void gelu_kernel_f16(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    size_t numel
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float x = __half2float(input[idx]);
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608028654f;
    float cdf = 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
    output[idx] = __float2half(x * cdf);
}

extern "C" BinferError binfer_gelu_f16(
    const void* input,
    void* output,
    size_t numel
) {
    const int block_size = 256;
    const int num_blocks = (numel + block_size - 1) / block_size;

    gelu_kernel_f16<<<num_blocks, block_size>>>(
        (const __half*)input,
        (__half*)output,
        numel
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// SwiGLU: silu(gate) * up
// Fused kernel for efficiency
__global__ void swiglu_kernel_f16(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ output,
    size_t numel
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);

    // SiLU on gate
    float sigmoid_g = 1.0f / (1.0f + expf(-g));
    float silu_g = g * sigmoid_g;

    output[idx] = __float2half(silu_g * u);
}

extern "C" BinferError binfer_swiglu_f16(
    const void* gate,
    const void* up,
    void* output,
    size_t numel
) {
    const int block_size = 256;
    const int num_blocks = (numel + block_size - 1) / block_size;

    swiglu_kernel_f16<<<num_blocks, block_size>>>(
        (const __half*)gate,
        (const __half*)up,
        (__half*)output,
        numel
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Element-wise add
__global__ void add_kernel_f16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ output,
    size_t numel
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    output[idx] = __hadd(a[idx], b[idx]);
}

extern "C" BinferError binfer_add_f16(
    const void* a,
    const void* b,
    void* output,
    size_t numel
) {
    const int block_size = 256;
    const int num_blocks = (numel + block_size - 1) / block_size;

    add_kernel_f16<<<num_blocks, block_size>>>(
        (const __half*)a,
        (const __half*)b,
        (__half*)output,
        numel
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Element-wise multiply
__global__ void mul_kernel_f16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ output,
    size_t numel
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    output[idx] = __hmul(a[idx], b[idx]);
}

extern "C" BinferError binfer_mul_f16(
    const void* a,
    const void* b,
    void* output,
    size_t numel
) {
    const int block_size = 256;
    const int num_blocks = (numel + block_size - 1) / block_size;

    mul_kernel_f16<<<num_blocks, block_size>>>(
        (const __half*)a,
        (const __half*)b,
        (__half*)output,
        numel
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Softmax with temperature
template<int BLOCK_SIZE>
__global__ void softmax_kernel_f16(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int vocab_size,
    float temperature
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __half* row_input = input + row * vocab_size;
    __half* row_output = output + row * vocab_size;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float val = __half2float(row_input[i]) / temperature;
        max_val = fmaxf(max_val, val);
    }

    // Warp reduction for max
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    __shared__ float shared_max[32];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) shared_max[warp_id] = max_val;
    __syncthreads();

    if (warp_id == 0) {
        max_val = (lane_id < BLOCK_SIZE / 32) ? shared_max[lane_id] : -INFINITY;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
    }

    __shared__ float global_max;
    if (tid == 0) global_max = max_val;
    __syncthreads();

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float val = __half2float(row_input[i]) / temperature - global_max;
        sum_exp += expf(val);
    }

    // Warp reduction for sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }

    __shared__ float shared_sum[32];
    if (lane_id == 0) shared_sum[warp_id] = sum_exp;
    __syncthreads();

    if (warp_id == 0) {
        sum_exp = (lane_id < BLOCK_SIZE / 32) ? shared_sum[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
        }
    }

    __shared__ float global_sum;
    if (tid == 0) global_sum = sum_exp;
    __syncthreads();

    // Compute softmax
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float val = __half2float(row_input[i]) / temperature - global_max;
        row_output[i] = __float2half(expf(val) / global_sum);
    }
}

extern "C" BinferError binfer_softmax_f16(
    const void* input,
    void* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    float temperature
) {
    const int total_rows = batch_size * seq_len;

    softmax_kernel_f16<1024><<<total_rows, 1024>>>(
        (const __half*)input,
        (__half*)output,
        vocab_size,
        temperature
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Embedding lookup
__global__ void embedding_kernel_f16(
    const __half* __restrict__ weight,
    const int32_t* __restrict__ input_ids,
    __half* __restrict__ output,
    int seq_len,
    int hidden_size
) {
    const int batch_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int dim_idx = threadIdx.x + blockIdx.z * blockDim.x;

    if (dim_idx >= hidden_size) return;

    const int input_idx = batch_idx * seq_len + seq_idx;
    const int token_id = input_ids[input_idx];
    const int output_idx = (batch_idx * seq_len + seq_idx) * hidden_size + dim_idx;

    output[output_idx] = weight[token_id * hidden_size + dim_idx];
}

extern "C" BinferError binfer_embedding_f16(
    const void* weight,
    const int32_t* input_ids,
    void* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    int hidden_size
) {
    const int threads_per_block = 256;
    const int blocks_per_dim = (hidden_size + threads_per_block - 1) / threads_per_block;

    dim3 grid(batch_size, seq_len, blocks_per_dim);
    dim3 block(threads_per_block);

    embedding_kernel_f16<<<grid, block>>>(
        (const __half*)weight,
        input_ids,
        (__half*)output,
        seq_len,
        hidden_size
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Top-k selection (simple implementation, can be optimized with heap)
__global__ void topk_kernel_f16(
    const __half* __restrict__ logits,
    __half* __restrict__ values,
    int32_t* __restrict__ indices,
    int vocab_size,
    int k
) {
    const int batch_idx = blockIdx.x;
    const __half* row_logits = logits + batch_idx * vocab_size;
    __half* row_values = values + batch_idx * k;
    int32_t* row_indices = indices + batch_idx * k;

    // Simple O(k*n) implementation - good enough for small k
    // For production, use radix select or bitonic sort
    for (int ki = 0; ki < k; ki++) {
        float max_val = -INFINITY;
        int max_idx = 0;

        for (int i = 0; i < vocab_size; i++) {
            float val = __half2float(row_logits[i]);
            // Check if already selected
            bool already_selected = false;
            for (int j = 0; j < ki; j++) {
                if (row_indices[j] == i) {
                    already_selected = true;
                    break;
                }
            }
            if (!already_selected && val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }

        row_values[ki] = __float2half(max_val);
        row_indices[ki] = max_idx;
    }
}

extern "C" BinferError binfer_topk_f16(
    const void* logits,
    void* values,
    int32_t* indices,
    int batch_size,
    int vocab_size,
    int k
) {
    // Use 1 thread per batch item for simplicity
    // This is inefficient but correct; optimize later with parallel reduction
    topk_kernel_f16<<<batch_size, 1>>>(
        (const __half*)logits,
        (__half*)values,
        indices,
        vocab_size,
        k
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}
