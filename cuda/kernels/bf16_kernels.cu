// BF16 (bfloat16) kernel implementations
// These kernels operate on __nv_bfloat16 type for native BF16 support

#include "binfer.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

// Get current stream from gemm.cu (for graph capture)
extern "C" void* binfer_get_current_stream();
#define GET_CURRENT_STREAM() ((cudaStream_t)binfer_get_current_stream())

// Helper: Convert bfloat16 to float and back
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

// ============================================================================
// RMSNorm BF16
// ============================================================================

template<int BLOCK_SIZE>
__global__ void rmsnorm_kernel_bf16(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int hidden_size,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* row_input = input + row * hidden_size;
    __nv_bfloat16* row_output = output + row * hidden_size;

    // Compute sum of squares using thread-local accumulator
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = bf16_to_float(row_input[i]);
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
        float val = bf16_to_float(row_input[i]);
        float w = bf16_to_float(weight[i]);
        row_output[i] = float_to_bf16(val * rsqrt_val * w);
    }
}

extern "C" BinferError binfer_rmsnorm_bf16(
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

    if (hidden_size <= 1024) {
        rmsnorm_kernel_bf16<256><<<total_rows, 256, 0, stream>>>(
            (const __nv_bfloat16*)input,
            (const __nv_bfloat16*)weight,
            (__nv_bfloat16*)output,
            hidden_size,
            eps
        );
    } else if (hidden_size <= 4096) {
        rmsnorm_kernel_bf16<512><<<total_rows, 512, 0, stream>>>(
            (const __nv_bfloat16*)input,
            (const __nv_bfloat16*)weight,
            (__nv_bfloat16*)output,
            hidden_size,
            eps
        );
    } else {
        rmsnorm_kernel_bf16<1024><<<total_rows, 1024, 0, stream>>>(
            (const __nv_bfloat16*)input,
            (const __nv_bfloat16*)weight,
            (__nv_bfloat16*)output,
            hidden_size,
            eps
        );
    }

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// RoPE BF16
// ============================================================================

__global__ void rotary_embedding_kernel_bf16(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int position_offset
) {
    const int batch_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int pair_idx = threadIdx.x;

    if (pair_idx >= head_dim / 2) return;

    const int half_dim = head_dim / 2;
    const int pos = seq_idx + position_offset;

    const float cos_val = bf16_to_float(cos[pos * half_dim + pair_idx]);
    const float sin_val = bf16_to_float(sin[pos * half_dim + pair_idx]);

    // Apply to Q
    if (head_idx < num_heads) {
        const int q_offset = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;

        float q0 = bf16_to_float(q[q_offset + pair_idx]);
        float q1 = bf16_to_float(q[q_offset + pair_idx + half_dim]);

        float q0_new = q0 * cos_val - q1 * sin_val;
        float q1_new = q0 * sin_val + q1 * cos_val;

        q[q_offset + pair_idx] = float_to_bf16(q0_new);
        q[q_offset + pair_idx + half_dim] = float_to_bf16(q1_new);
    }

    // Apply to K
    if (head_idx < num_kv_heads) {
        const int k_offset = ((batch_idx * seq_len + seq_idx) * num_kv_heads + head_idx) * head_dim;

        float k0 = bf16_to_float(k[k_offset + pair_idx]);
        float k1 = bf16_to_float(k[k_offset + pair_idx + half_dim]);

        float k0_new = k0 * cos_val - k1 * sin_val;
        float k1_new = k0 * sin_val + k1 * cos_val;

        k[k_offset + pair_idx] = float_to_bf16(k0_new);
        k[k_offset + pair_idx + half_dim] = float_to_bf16(k1_new);
    }
}

extern "C" BinferError binfer_rotary_embedding_bf16(
    void* q,
    void* k,
    const void* cos,
    const void* sin,
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int position_offset
) {
    const int max_heads = num_heads > num_kv_heads ? num_heads : num_kv_heads;
    cudaStream_t stream = GET_CURRENT_STREAM();

    dim3 grid(batch_size, seq_len, max_heads);
    dim3 block(head_dim / 2);

    rotary_embedding_kernel_bf16<<<grid, block, 0, stream>>>(
        (__nv_bfloat16*)q,
        (__nv_bfloat16*)k,
        (const __nv_bfloat16*)cos,
        (const __nv_bfloat16*)sin,
        batch_size,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        position_offset
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// Batched RoPE with per-sequence position offsets (for decode)
// ============================================================================
// Each sequence can have a different position offset.
// Layout: Q/K are [total_tokens, num_heads, head_dim] where total_tokens = sum of seq_lens
// For decode: each sequence has 1 token, so total_tokens = batch_size

__global__ void rotary_embedding_batched_kernel_bf16(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    const int* __restrict__ token_offsets,    // cumsum of seq_lens: [0, len0, len0+len1, ...]
    const int* __restrict__ position_offsets, // position offset per sequence
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    // Each block handles one (seq, local_pos, head) combination
    const int seq_idx = blockIdx.x;
    const int local_pos = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int pair_idx = threadIdx.x;

    if (seq_idx >= num_seqs) return;
    if (pair_idx >= head_dim / 2) return;

    const int half_dim = head_dim / 2;

    // Get the token range for this sequence
    const int tok_start = token_offsets[seq_idx];
    const int tok_end = token_offsets[seq_idx + 1];
    const int seq_len = tok_end - tok_start;

    if (local_pos >= seq_len) return;

    // Global position in the context
    const int pos = position_offsets[seq_idx] + local_pos;

    // Global token index
    const int token_idx = tok_start + local_pos;

    const float cos_val = bf16_to_float(cos[pos * half_dim + pair_idx]);
    const float sin_val = bf16_to_float(sin[pos * half_dim + pair_idx]);

    // Apply to Q
    if (head_idx < num_heads) {
        const int q_offset = (token_idx * num_heads + head_idx) * head_dim;

        float q0 = bf16_to_float(q[q_offset + pair_idx]);
        float q1 = bf16_to_float(q[q_offset + pair_idx + half_dim]);

        float q0_new = q0 * cos_val - q1 * sin_val;
        float q1_new = q0 * sin_val + q1 * cos_val;

        q[q_offset + pair_idx] = float_to_bf16(q0_new);
        q[q_offset + pair_idx + half_dim] = float_to_bf16(q1_new);
    }

    // Apply to K
    if (head_idx < num_kv_heads) {
        const int k_offset = (token_idx * num_kv_heads + head_idx) * head_dim;

        float k0 = bf16_to_float(k[k_offset + pair_idx]);
        float k1 = bf16_to_float(k[k_offset + pair_idx + half_dim]);

        float k0_new = k0 * cos_val - k1 * sin_val;
        float k1_new = k0 * sin_val + k1 * cos_val;

        k[k_offset + pair_idx] = float_to_bf16(k0_new);
        k[k_offset + pair_idx + half_dim] = float_to_bf16(k1_new);
    }
}

extern "C" BinferError binfer_rotary_embedding_batched_bf16(
    void* q,
    void* k,
    const void* cos,
    const void* sin,
    const int* token_offsets,    // GPU pointer: cumsum [0, len0, len0+len1, ...]
    const int* position_offsets, // GPU pointer: position offset per sequence
    int num_seqs,
    int max_seq_len,  // Max sequence length in batch (for grid sizing)
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    const int max_heads = num_heads > num_kv_heads ? num_heads : num_kv_heads;
    cudaStream_t stream = GET_CURRENT_STREAM();

    dim3 grid(num_seqs, max_seq_len, max_heads);
    dim3 block(head_dim / 2);

    rotary_embedding_batched_kernel_bf16<<<grid, block, 0, stream>>>(
        (__nv_bfloat16*)q,
        (__nv_bfloat16*)k,
        (const __nv_bfloat16*)cos,
        (const __nv_bfloat16*)sin,
        token_offsets,
        position_offsets,
        num_seqs,
        num_heads,
        num_kv_heads,
        head_dim
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Simpler batched RoPE for decode mode: each sequence has exactly 1 token
// This avoids the complexity of token_offsets
__global__ void rotary_embedding_decode_kernel_bf16(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    const int* __restrict__ positions,  // position for each sequence
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int pair_idx = threadIdx.x;

    if (seq_idx >= batch_size) return;
    if (pair_idx >= head_dim / 2) return;

    const int half_dim = head_dim / 2;
    const int pos = positions[seq_idx];

    const float cos_val = bf16_to_float(cos[pos * half_dim + pair_idx]);
    const float sin_val = bf16_to_float(sin[pos * half_dim + pair_idx]);

    // Apply to Q - layout is [batch, num_heads, head_dim]
    if (head_idx < num_heads) {
        const int q_offset = (seq_idx * num_heads + head_idx) * head_dim;

        float q0 = bf16_to_float(q[q_offset + pair_idx]);
        float q1 = bf16_to_float(q[q_offset + pair_idx + half_dim]);

        float q0_new = q0 * cos_val - q1 * sin_val;
        float q1_new = q0 * sin_val + q1 * cos_val;

        q[q_offset + pair_idx] = float_to_bf16(q0_new);
        q[q_offset + pair_idx + half_dim] = float_to_bf16(q1_new);
    }

    // Apply to K - layout is [batch, num_kv_heads, head_dim]
    if (head_idx < num_kv_heads) {
        const int k_offset = (seq_idx * num_kv_heads + head_idx) * head_dim;

        float k0 = bf16_to_float(k[k_offset + pair_idx]);
        float k1 = bf16_to_float(k[k_offset + pair_idx + half_dim]);

        float k0_new = k0 * cos_val - k1 * sin_val;
        float k1_new = k0 * sin_val + k1 * cos_val;

        k[k_offset + pair_idx] = float_to_bf16(k0_new);
        k[k_offset + pair_idx + half_dim] = float_to_bf16(k1_new);
    }
}

extern "C" BinferError binfer_rotary_embedding_decode_bf16(
    void* q,
    void* k,
    const void* cos,
    const void* sin,
    const int* positions,  // GPU pointer: position for each sequence
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    const int max_heads = num_heads > num_kv_heads ? num_heads : num_kv_heads;
    cudaStream_t stream = GET_CURRENT_STREAM();

    dim3 grid(batch_size, max_heads);
    dim3 block(head_dim / 2);

    rotary_embedding_decode_kernel_bf16<<<grid, block, 0, stream>>>(
        (__nv_bfloat16*)q,
        (__nv_bfloat16*)k,
        (const __nv_bfloat16*)cos,
        (const __nv_bfloat16*)sin,
        positions,
        batch_size,
        num_heads,
        num_kv_heads,
        head_dim
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// SiLU BF16
// ============================================================================

__global__ void silu_kernel_bf16(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    size_t numel
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float x = bf16_to_float(input[idx]);
    float sigmoid_x = 1.0f / (1.0f + expf(-x));
    output[idx] = float_to_bf16(x * sigmoid_x);
}

extern "C" BinferError binfer_silu_bf16(
    const void* input,
    void* output,
    size_t numel
) {
    const int block_size = 256;
    const int num_blocks = (numel + block_size - 1) / block_size;
    cudaStream_t stream = GET_CURRENT_STREAM();

    silu_kernel_bf16<<<num_blocks, block_size, 0, stream>>>(
        (const __nv_bfloat16*)input,
        (__nv_bfloat16*)output,
        numel
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// SwiGLU BF16
// ============================================================================

__global__ void swiglu_kernel_bf16(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    size_t numel
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float g = bf16_to_float(gate[idx]);
    float u = bf16_to_float(up[idx]);

    float sigmoid_g = 1.0f / (1.0f + expf(-g));
    float silu_g = g * sigmoid_g;

    output[idx] = float_to_bf16(silu_g * u);
}

extern "C" BinferError binfer_swiglu_bf16(
    const void* gate,
    const void* up,
    void* output,
    size_t numel
) {
    const int block_size = 256;
    const int num_blocks = (numel + block_size - 1) / block_size;
    cudaStream_t stream = GET_CURRENT_STREAM();

    swiglu_kernel_bf16<<<num_blocks, block_size, 0, stream>>>(
        (const __nv_bfloat16*)gate,
        (const __nv_bfloat16*)up,
        (__nv_bfloat16*)output,
        numel
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// Element-wise Add BF16
// ============================================================================

__global__ void add_kernel_bf16(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ output,
    size_t numel
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    // BF16 addition: convert to float, add, convert back
    float fa = __bfloat162float(a[idx]);
    float fb = __bfloat162float(b[idx]);
    output[idx] = __float2bfloat16(fa + fb);
}

extern "C" BinferError binfer_add_bf16(
    const void* a,
    const void* b,
    void* output,
    size_t numel
) {
    const int block_size = 256;
    const int num_blocks = (numel + block_size - 1) / block_size;
    cudaStream_t stream = GET_CURRENT_STREAM();

    add_kernel_bf16<<<num_blocks, block_size, 0, stream>>>(
        (const __nv_bfloat16*)a,
        (const __nv_bfloat16*)b,
        (__nv_bfloat16*)output,
        numel
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// Batched Bias Addition BF16
// ============================================================================
// Adds bias to each row: output[i, j] = input[i, j] + bias[j]
// This is much more efficient than launching a kernel per row!

__global__ void add_bias_kernel_bf16(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    int num_rows,
    int row_size
) {
    const int row = blockIdx.x;
    const int col = threadIdx.x + blockIdx.y * blockDim.x;

    if (row >= num_rows || col >= row_size) return;

    const int idx = row * row_size + col;
    float val = __bfloat162float(input[idx]);
    float b = __bfloat162float(bias[col]);
    output[idx] = __float2bfloat16(val + b);
}

extern "C" BinferError binfer_add_bias_bf16(
    const void* input,
    const void* bias,
    void* output,
    int num_rows,
    int row_size
) {
    const int threads_per_block = 256;
    const int blocks_per_row = (row_size + threads_per_block - 1) / threads_per_block;
    cudaStream_t stream = GET_CURRENT_STREAM();

    dim3 grid(num_rows, blocks_per_row);
    dim3 block(threads_per_block);

    add_bias_kernel_bf16<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)input,
        (const __nv_bfloat16*)bias,
        (__nv_bfloat16*)output,
        num_rows,
        row_size
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// In-place version (output == input)
extern "C" BinferError binfer_add_bias_inplace_bf16(
    void* data,
    const void* bias,
    int num_rows,
    int row_size
) {
    return binfer_add_bias_bf16(data, bias, data, num_rows, row_size);
}

// ============================================================================
// Softmax BF16
// ============================================================================

template<int BLOCK_SIZE>
__global__ void softmax_kernel_bf16(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int vocab_size,
    float temperature
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* row_input = input + row * vocab_size;
    __nv_bfloat16* row_output = output + row * vocab_size;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float val = bf16_to_float(row_input[i]) / temperature;
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
        float val = bf16_to_float(row_input[i]) / temperature - global_max;
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
        float val = bf16_to_float(row_input[i]) / temperature - global_max;
        row_output[i] = float_to_bf16(expf(val) / global_sum);
    }
}

extern "C" BinferError binfer_softmax_bf16(
    const void* input,
    void* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    float temperature
) {
    const int total_rows = batch_size * seq_len;
    cudaStream_t stream = GET_CURRENT_STREAM();

    softmax_kernel_bf16<1024><<<total_rows, 1024, 0, stream>>>(
        (const __nv_bfloat16*)input,
        (__nv_bfloat16*)output,
        vocab_size,
        temperature
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// Embedding Lookup BF16
// ============================================================================

__global__ void embedding_kernel_bf16(
    const __nv_bfloat16* __restrict__ weight,
    const int32_t* __restrict__ input_ids,
    __nv_bfloat16* __restrict__ output,
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

extern "C" BinferError binfer_embedding_bf16(
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
    cudaStream_t stream = GET_CURRENT_STREAM();

    dim3 grid(batch_size, seq_len, blocks_per_dim);
    dim3 block(threads_per_block);

    embedding_kernel_bf16<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)weight,
        input_ids,
        (__nv_bfloat16*)output,
        seq_len,
        hidden_size
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// RoPE Cache Computation BF16
// ============================================================================

__global__ void compute_rope_freqs_kernel_bf16(
    __nv_bfloat16* __restrict__ cos_cache,
    __nv_bfloat16* __restrict__ sin_cache,
    int max_seq_len,
    int head_dim,
    float base,
    float scaling_factor
) {
    const int pos = blockIdx.x;
    const int dim_idx = threadIdx.x;

    if (dim_idx >= head_dim / 2) return;

    const float freq = 1.0f / powf(base, (float)(2 * dim_idx) / head_dim);
    const float angle = (float)pos * freq / scaling_factor;

    const int idx = pos * (head_dim / 2) + dim_idx;
    cos_cache[idx] = float_to_bf16(cosf(angle));
    sin_cache[idx] = float_to_bf16(sinf(angle));
}

extern "C" BinferError binfer_compute_rope_cache_bf16(
    void* cos_cache,
    void* sin_cache,
    int max_seq_len,
    int head_dim,
    float base,
    float scaling_factor
) {
    dim3 grid(max_seq_len);
    dim3 block(head_dim / 2);
    cudaStream_t stream = GET_CURRENT_STREAM();

    compute_rope_freqs_kernel_bf16<<<grid, block, 0, stream>>>(
        (__nv_bfloat16*)cos_cache,
        (__nv_bfloat16*)sin_cache,
        max_seq_len,
        head_dim,
        base,
        scaling_factor
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// BF16 to FP16 Conversion
// ============================================================================

__global__ void convert_bf16_to_fp16_kernel(
    const __nv_bfloat16* __restrict__ input,
    __half* __restrict__ output,
    size_t num_elements
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // BF16 -> float32 -> FP16
    float val = __bfloat162float(input[idx]);
    output[idx] = __float2half(val);
}

extern "C" BinferError binfer_convert_bf16_to_fp16(
    void* input,
    void* output,
    size_t num_elements
) {
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;
    cudaStream_t stream = GET_CURRENT_STREAM();

    convert_bf16_to_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
        (const __nv_bfloat16*)input,
        (__half*)output,
        num_elements
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// Flash Attention BF16
// ============================================================================

#include <cfloat>

// Block sizes for flash attention
constexpr int BLOCK_M_BF16 = 128;
constexpr int BLOCK_N_BF16 = 64;
constexpr int WARP_SIZE_BF16 = 32;

struct SoftmaxStateBF16 {
    float max_val;
    float sum_exp;
};

__device__ __forceinline__ SoftmaxStateBF16 softmax_update_bf16(
    SoftmaxStateBF16 state, float new_max, float new_sum
) {
    float max_new = fmaxf(state.max_val, new_max);
    float scale_old = expf(state.max_val - max_new);
    float scale_new = expf(new_max - max_new);
    return {
        max_new,
        state.sum_exp * scale_old + new_sum * scale_new
    };
}

template<int HEAD_DIM, bool IS_CAUSAL>
__global__ void flash_attention_kernel_bf16(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    int batch_size,
    int seq_q,
    int seq_kv,
    int kv_stride,
    int q_offset,
    int num_heads,
    int num_kv_heads,
    float softmax_scale
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.z;

    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    const int q_start = q_block_idx * BLOCK_M_BF16;
    const int q_end = min(q_start + BLOCK_M_BF16, seq_q);

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE_BF16;
    const int lane_id = tid % WARP_SIZE_BF16;

    float O_acc[HEAD_DIM / WARP_SIZE_BF16] = {0.0f};
    SoftmaxStateBF16 softmax_state = {-FLT_MAX, 0.0f};

    const __nv_bfloat16* Q_ptr = Q + (batch_idx * seq_q * num_heads + head_idx) * HEAD_DIM;
    const __nv_bfloat16* K_ptr = K + (batch_idx * kv_stride * num_kv_heads + kv_head_idx) * HEAD_DIM;
    const __nv_bfloat16* V_ptr = V + (batch_idx * kv_stride * num_kv_heads + kv_head_idx) * HEAD_DIM;
    __nv_bfloat16* O_ptr = O + (batch_idx * seq_q * num_heads + head_idx) * HEAD_DIM;

    for (int q_pos = q_start + warp_id; q_pos < q_end; q_pos += blockDim.x / WARP_SIZE_BF16) {
        float q_local[HEAD_DIM / WARP_SIZE_BF16];
        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE_BF16) {
            q_local[d / WARP_SIZE_BF16] = __bfloat162float(Q_ptr[q_pos * num_heads * HEAD_DIM + d]);
        }

        for (int i = 0; i < HEAD_DIM / WARP_SIZE_BF16; i++) O_acc[i] = 0.0f;
        softmax_state = {-FLT_MAX, 0.0f};

        int abs_q_pos = q_pos + q_offset;
        int kv_end = IS_CAUSAL ? min(abs_q_pos + 1, seq_kv) : seq_kv;

        for (int kv_start = 0; kv_start < kv_end; kv_start += BLOCK_N_BF16) {
            int kv_block_end = min(kv_start + BLOCK_N_BF16, kv_end);

            float scores[BLOCK_N_BF16];
            float block_max = -FLT_MAX;

            for (int kv_pos = kv_start; kv_pos < kv_block_end; kv_pos++) {
                float score = 0.0f;
                for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE_BF16) {
                    float k_val = __bfloat162float(K_ptr[kv_pos * num_kv_heads * HEAD_DIM + d]);
                    score += q_local[d / WARP_SIZE_BF16] * k_val;
                }
                for (int offset = WARP_SIZE_BF16 / 2; offset > 0; offset /= 2) {
                    score += __shfl_down_sync(0xffffffff, score, offset);
                }
                score = __shfl_sync(0xffffffff, score, 0);

                if (IS_CAUSAL && kv_pos > abs_q_pos) {
                    score = -FLT_MAX;
                }

                scores[kv_pos - kv_start] = score * softmax_scale;
                block_max = fmaxf(block_max, scores[kv_pos - kv_start]);
            }

            float block_sum = 0.0f;
            for (int i = 0; i < kv_block_end - kv_start; i++) {
                scores[i] = expf(scores[i] - block_max);
                block_sum += scores[i];
            }

            float old_max = softmax_state.max_val;
            softmax_state = softmax_update_bf16(softmax_state, block_max, block_sum);

            float scale = expf(old_max - softmax_state.max_val);
            for (int i = 0; i < HEAD_DIM / WARP_SIZE_BF16; i++) {
                O_acc[i] *= scale;
            }

            for (int kv_pos = kv_start; kv_pos < kv_block_end; kv_pos++) {
                float attn_weight = scores[kv_pos - kv_start] * expf(block_max - softmax_state.max_val);
                for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE_BF16) {
                    float v_val = __bfloat162float(V_ptr[kv_pos * num_kv_heads * HEAD_DIM + d]);
                    O_acc[d / WARP_SIZE_BF16] += attn_weight * v_val;
                }
            }
        }

        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE_BF16) {
            float o_val = O_acc[d / WARP_SIZE_BF16] / softmax_state.sum_exp;
            O_ptr[q_pos * num_heads * HEAD_DIM + d] = __float2bfloat16(o_val);
        }
    }
}

template<bool IS_CAUSAL>
BinferError flash_attention_dispatch_bf16(
    const void* Q, const void* K, const void* V, void* O,
    int batch_size, int seq_q, int seq_kv, int kv_stride, int q_offset,
    int num_heads, int num_kv_heads, int head_dim,
    float softmax_scale
) {
    dim3 grid(batch_size, num_heads, (seq_q + BLOCK_M_BF16 - 1) / BLOCK_M_BF16);
    dim3 block(256);
    cudaStream_t stream = GET_CURRENT_STREAM();

    switch (head_dim) {
        case 64:
            flash_attention_kernel_bf16<64, IS_CAUSAL><<<grid, block, 0, stream>>>(
                (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V, (__nv_bfloat16*)O,
                batch_size, seq_q, seq_kv, kv_stride, q_offset, num_heads, num_kv_heads, softmax_scale
            );
            break;
        case 128:
            flash_attention_kernel_bf16<128, IS_CAUSAL><<<grid, block, 0, stream>>>(
                (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V, (__nv_bfloat16*)O,
                batch_size, seq_q, seq_kv, kv_stride, q_offset, num_heads, num_kv_heads, softmax_scale
            );
            break;
        default:
            return BINFER_ERROR_INVALID_ARGUMENT;
    }

    return cudaGetLastError() == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_flash_attention_bf16(
    const void* Q, const void* K, const void* V, void* O,
    int batch_size, int seq_q, int seq_kv, int kv_stride, int q_offset,
    int num_heads, int num_kv_heads, int head_dim,
    float softmax_scale, bool is_causal
) {
    if (is_causal) {
        return flash_attention_dispatch_bf16<true>(
            Q, K, V, O, batch_size, seq_q, seq_kv, kv_stride, q_offset,
            num_heads, num_kv_heads, head_dim, softmax_scale
        );
    } else {
        return flash_attention_dispatch_bf16<false>(
            Q, K, V, O, batch_size, seq_q, seq_kv, kv_stride, q_offset,
            num_heads, num_kv_heads, head_dim, softmax_scale
        );
    }
}

extern "C" BinferError binfer_flash_attention_with_cache_bf16(
    const void* Q, const void* K_cache, const void* V_cache, void* O,
    int batch_size, int cache_seq_len, int max_seq_len, int q_offset,
    int num_heads, int num_kv_heads, int head_dim,
    float softmax_scale, bool is_causal
) {
    return binfer_flash_attention_bf16(
        Q, K_cache, V_cache, O,
        batch_size, 1, cache_seq_len, max_seq_len, q_offset,
        num_heads, num_kv_heads, head_dim,
        softmax_scale, is_causal
    );
}

// BF16 KV cache update kernel
__global__ void kv_cache_update_kernel_bf16(
    __nv_bfloat16* __restrict__ K_cache,
    __nv_bfloat16* __restrict__ V_cache,
    const __nv_bfloat16* __restrict__ K_new,
    const __nv_bfloat16* __restrict__ V_new,
    int batch_size,
    int cache_offset,
    int seq_new,
    int num_kv_heads,
    int head_dim,
    int max_seq_len
) {
    const int batch_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int dim_idx = threadIdx.x;

    if (dim_idx >= head_dim) return;
    if (cache_offset + seq_idx >= max_seq_len) return;

    const int cache_pos = cache_offset + seq_idx;
    const int new_idx = ((batch_idx * seq_new + seq_idx) * num_kv_heads + head_idx) * head_dim + dim_idx;
    const int cache_idx = ((batch_idx * max_seq_len + cache_pos) * num_kv_heads + head_idx) * head_dim + dim_idx;

    K_cache[cache_idx] = K_new[new_idx];
    V_cache[cache_idx] = V_new[new_idx];
}

extern "C" BinferError binfer_kv_cache_update_bf16(
    void* K_cache, void* V_cache,
    const void* K_new, const void* V_new,
    int batch_size, int cache_offset, int seq_new,
    int num_kv_heads, int head_dim, int max_seq_len
) {
    dim3 grid(batch_size, seq_new, num_kv_heads);
    dim3 block(head_dim);
    cudaStream_t stream = GET_CURRENT_STREAM();

    kv_cache_update_kernel_bf16<<<grid, block, 0, stream>>>(
        (__nv_bfloat16*)K_cache, (__nv_bfloat16*)V_cache,
        (const __nv_bfloat16*)K_new, (const __nv_bfloat16*)V_new,
        batch_size, cache_offset, seq_new,
        num_kv_heads, head_dim, max_seq_len
    );

    return cudaGetLastError() == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// Eager Attention with Sinks (for GPT-OSS)
// ============================================================================
// Computes attention with extra "sink" logits that absorb some attention mass
// Formula:
//   attn_weights = Q @ K^T * scale
//   combined_logits = concat(attn_weights, sinks)
//   combined_logits = combined_logits - max(combined_logits)
//   probs = softmax(combined_logits)
//   output_probs = probs[:, :-1]  (drop sink)
//   O = output_probs @ V

// Simple attention kernel for small sequences
// Grid: (batch_size, num_heads, seq_q)
// Each thread block handles one query position
__global__ void attention_with_sinks_kernel_bf16(
    const __nv_bfloat16* __restrict__ Q,     // [batch, seq_q, num_heads, head_dim]
    const __nv_bfloat16* __restrict__ K,     // [batch, max_seq, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ V,     // [batch, max_seq, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ sinks, // [num_heads]
    __nv_bfloat16* __restrict__ O,           // [batch, seq_q, num_heads, head_dim]
    int batch_size,
    int seq_q,
    int seq_kv,        // actual KV sequence length
    int kv_stride,     // stride in KV cache (max_seq_len)
    int q_offset,      // offset for Q position (decode mode)
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    bool is_causal
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int q_pos = blockIdx.z;
    const int tid = threadIdx.x;

    // Each thread handles part of head_dim for dot product
    if (batch_idx >= batch_size || head_idx >= num_heads || q_pos >= seq_q) return;

    // GQA: map attention heads to KV heads
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    extern __shared__ float shared_mem[];
    float* logits = shared_mem;                     // [seq_kv + 1]
    float* output_accum = &shared_mem[seq_kv + 1];  // [head_dim]

    // Get Q vector for this position
    const int q_idx = ((batch_idx * seq_q + q_pos) * num_heads + head_idx) * head_dim;
    const __nv_bfloat16* q_vec = Q + q_idx;

    // Compute attention scores: Q @ K^T
    for (int kv_pos = tid; kv_pos < seq_kv; kv_pos += blockDim.x) {
        // Check causal mask
        float mask_val = 0.0f;
        if (is_causal && (q_offset + q_pos) < kv_pos) {
            mask_val = -INFINITY;
        }

        if (mask_val == -INFINITY) {
            logits[kv_pos] = -INFINITY;
        } else {
            // Compute dot product
            const int k_idx = ((batch_idx * kv_stride + kv_pos) * num_kv_heads + kv_head_idx) * head_dim;
            const __nv_bfloat16* k_vec = K + k_idx;

            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += bf16_to_float(q_vec[d]) * bf16_to_float(k_vec[d]);
            }
            logits[kv_pos] = dot * scale;
        }
    }

    // Add sink logit
    if (tid == 0) {
        logits[seq_kv] = bf16_to_float(sinks[head_idx]);
    }
    __syncthreads();

    // Find max (for numerical stability)
    float max_logit = -INFINITY;
    for (int i = tid; i <= seq_kv; i += blockDim.x) {
        max_logit = fmaxf(max_logit, logits[i]);
    }

    // Warp reduction for max
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        max_logit = fmaxf(max_logit, __shfl_down_sync(0xffffffff, max_logit, offset));
    }

    __shared__ float global_max;
    if (tid == 0) global_max = max_logit;
    __syncthreads();

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int i = tid; i <= seq_kv; i += blockDim.x) {
        float val = expf(logits[i] - global_max);
        logits[i] = val;  // Store exp values
        sum_exp += val;
    }

    // Warp reduction for sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }

    __shared__ float global_sum;
    if (tid == 0) global_sum = sum_exp;
    __syncthreads();

    // Normalize to get probabilities (excluding sink)
    for (int i = tid; i < seq_kv; i += blockDim.x) {
        logits[i] = logits[i] / global_sum;
    }
    __syncthreads();

    // Initialize output accumulator
    for (int d = tid; d < head_dim; d += blockDim.x) {
        output_accum[d] = 0.0f;
    }
    __syncthreads();

    // Compute output: probs @ V
    for (int kv_pos = 0; kv_pos < seq_kv; kv_pos++) {
        float prob = logits[kv_pos];
        if (prob > 0.0f) {
            const int v_idx = ((batch_idx * kv_stride + kv_pos) * num_kv_heads + kv_head_idx) * head_dim;
            const __nv_bfloat16* v_vec = V + v_idx;

            for (int d = tid; d < head_dim; d += blockDim.x) {
                atomicAdd(&output_accum[d], prob * bf16_to_float(v_vec[d]));
            }
        }
    }
    __syncthreads();

    // Write output
    const int o_idx = ((batch_idx * seq_q + q_pos) * num_heads + head_idx) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        O[o_idx + d] = float_to_bf16(output_accum[d]);
    }
}

extern "C" BinferError binfer_attention_with_sinks_bf16(
    const void* Q, const void* K, const void* V, const void* sinks,
    void* O,
    int batch_size, int seq_q, int seq_kv, int kv_stride, int q_offset,
    int num_heads, int num_kv_heads, int head_dim,
    float scale, bool is_causal
) {
    dim3 grid(batch_size, num_heads, seq_q);
    dim3 block(32);  // Use 32 threads per block
    cudaStream_t stream = GET_CURRENT_STREAM();

    // Shared memory: logits (seq_kv + 1 floats) + output_accum (head_dim floats)
    size_t shared_mem_size = (seq_kv + 1 + head_dim) * sizeof(float);

    attention_with_sinks_kernel_bf16<<<grid, block, shared_mem_size, stream>>>(
        (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
        (const __nv_bfloat16*)sinks, (__nv_bfloat16*)O,
        batch_size, seq_q, seq_kv, kv_stride, q_offset,
        num_heads, num_kv_heads, head_dim, scale, is_causal
    );

    return cudaGetLastError() == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// Argmax kernel for GPU sampling
// ============================================================================

// Each block handles one row (sequence), finds the argmax of vocab_size elements
template<int BLOCK_SIZE>
__global__ void argmax_kernel_bf16(
    const __nv_bfloat16* __restrict__ logits,  // [batch_size, vocab_size]
    int32_t* __restrict__ output_tokens,        // [batch_size]
    int vocab_size
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* row = logits + batch_idx * vocab_size;

    // Thread-local max tracking
    float local_max = -INFINITY;
    int local_idx = 0;

    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float val = bf16_to_float(row[i]);
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    // Warp reduction to find max within warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (other_max > local_max) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }

    // Block reduction using shared memory
    __shared__ float shared_max[32];
    __shared__ int shared_idx[32];

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        shared_max[warp_id] = local_max;
        shared_idx[warp_id] = local_idx;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        int num_warps = (BLOCK_SIZE + 31) / 32;
        local_max = lane_id < num_warps ? shared_max[lane_id] : -INFINITY;
        local_idx = lane_id < num_warps ? shared_idx[lane_id] : 0;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
            int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
            if (other_max > local_max) {
                local_max = other_max;
                local_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            output_tokens[batch_idx] = local_idx;
        }
    }
}

extern "C" BinferError binfer_argmax_bf16(
    const void* logits,
    void* output_tokens,
    int batch_size,
    int vocab_size
) {
    // Use 256 threads per block for good occupancy
    const int BLOCK_SIZE = 256;
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);
    cudaStream_t stream = GET_CURRENT_STREAM();

    argmax_kernel_bf16<BLOCK_SIZE><<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)logits,
        (int32_t*)output_tokens,
        vocab_size
    );

    return cudaGetLastError() == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}
