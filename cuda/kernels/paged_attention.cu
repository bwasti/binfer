// Paged Attention CUDA kernel
// Gathers K/V from non-contiguous blocks for variable-length batched sequences

#include "attention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cfloat>

// Get current stream from gemm.cu (for graph capture)
extern "C" void* binfer_get_current_stream();
#define GET_CURRENT_STREAM() ((cudaStream_t)binfer_get_current_stream())

// Block size must match TypeScript BLOCK_SIZE
constexpr int BLOCK_SIZE = 16;

// Warp size
constexpr int WARP_SIZE = 32;

// Convert bf16 to float
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

// Convert float to bf16
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float val) {
    return __float2bfloat16(val);
}

// Paged attention kernel for decode phase
// Each sequence can have different length and different block mapping
template<int HEAD_DIM>
__global__ void paged_attention_kernel_f16(
    const __half* __restrict__ Q,           // [num_seqs, num_heads, head_dim]
    const __half* __restrict__ K_cache,     // [num_blocks, block_size, num_kv_heads, head_dim]
    const __half* __restrict__ V_cache,     // [num_blocks, block_size, num_kv_heads, head_dim]
    __half* __restrict__ O,                 // [num_seqs, num_heads, head_dim]
    const int* __restrict__ block_tables,   // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,   // [num_seqs]
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int max_blocks_per_seq,
    float softmax_scale
) {
    // Grid: (num_seqs, num_heads)
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    // GQA: map query head to KV head
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Thread position
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Get context length for this sequence
    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    // Get block table for this sequence
    const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q for this sequence/head into registers
    float q_local[HEAD_DIM / WARP_SIZE];
    const __half* q_ptr = Q + (seq_idx * num_heads + head_idx) * HEAD_DIM;
    for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
        q_local[d / WARP_SIZE] = __half2float(q_ptr[d]);
    }

    // Online softmax state
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    float acc[HEAD_DIM / WARP_SIZE] = {0.0f};

    // Process each token in context
    // Each warp processes different tokens in parallel
    for (int token_idx = warp_id; token_idx < context_len; token_idx += num_warps) {
        // Find which block and slot this token is in
        const int block_idx = token_idx / BLOCK_SIZE;
        const int slot_idx = token_idx % BLOCK_SIZE;
        const int physical_block = seq_block_table[block_idx];

        // Calculate K/V addresses
        // K_cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
        const int kv_offset = ((physical_block * BLOCK_SIZE + slot_idx) * num_kv_heads + kv_head_idx) * HEAD_DIM;
        const __half* k_ptr = K_cache + kv_offset;
        const __half* v_ptr = V_cache + kv_offset;

        // Compute Q @ K^T
        float score = 0.0f;
        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
            float k_val = __half2float(k_ptr[d]);
            score += q_local[d / WARP_SIZE] * k_val;
        }

        // Warp reduce for score
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            score += __shfl_down_sync(0xffffffff, score, offset);
        }
        score = __shfl_sync(0xffffffff, score, 0);
        score *= softmax_scale;

        // Update online softmax
        float old_max = max_score;
        max_score = fmaxf(max_score, score);
        float scale = expf(old_max - max_score);
        sum_exp = sum_exp * scale + expf(score - max_score);

        // Scale previous accumulator
        for (int i = 0; i < HEAD_DIM / WARP_SIZE; i++) {
            acc[i] *= scale;
        }

        // Accumulate V weighted by attention
        float attn_weight = expf(score - max_score);
        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
            float v_val = __half2float(v_ptr[d]);
            acc[d / WARP_SIZE] += attn_weight * v_val;
        }
    }

    // Reduce across warps using shared memory
    extern __shared__ char smem[];
    float* warp_max = (float*)smem;
    float* warp_sum = warp_max + num_warps;
    float* warp_acc = warp_sum + num_warps;  // [num_warps, HEAD_DIM]

    if (lane_id == 0) {
        warp_max[warp_id] = max_score;
        warp_sum[warp_id] = sum_exp;
    }
    for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
        warp_acc[warp_id * HEAD_DIM + d] = acc[d / WARP_SIZE];
    }
    __syncthreads();

    // Final reduction (first warp only)
    if (warp_id == 0) {
        // Calculate how many warps actually processed tokens
        // Warp w handles tokens at positions w, w+num_warps, w+2*num_warps, ...
        // So warp w processed tokens if w < context_len
        const int active_warps = min(num_warps, context_len);

        float final_max = -FLT_MAX;
        for (int w = 0; w < active_warps; w++) {
            final_max = fmaxf(final_max, warp_max[w]);
        }

        float final_sum = 0.0f;
        for (int w = 0; w < active_warps; w++) {
            final_sum += warp_sum[w] * expf(warp_max[w] - final_max);
        }

        // Combine accumulators
        float final_acc[HEAD_DIM / WARP_SIZE] = {0.0f};
        for (int w = 0; w < active_warps; w++) {
            float scale = expf(warp_max[w] - final_max);
            for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                final_acc[d / WARP_SIZE] += warp_acc[w * HEAD_DIM + d] * scale;
            }
        }

        // Write output
        __half* o_ptr = O + (seq_idx * num_heads + head_idx) * HEAD_DIM;
        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
            float o_val = final_acc[d / WARP_SIZE] / final_sum;
            o_ptr[d] = __float2half(o_val);
        }
    }
}

// Dispatch based on head dimension
extern "C" BinferError binfer_paged_attention_f16(
    const void* Q,
    const void* K_cache,
    const void* V_cache,
    void* O,
    const int* block_tables,  // host pointer
    const int* context_lens,  // host pointer
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,   // stride of block_tables (passed from caller)
    float softmax_scale
) {
    if (block_size != BLOCK_SIZE) {
        return BINFER_ERROR_INVALID_ARGUMENT;
    }

    // Allocate device memory for block_tables and context_lens
    int* d_block_tables;
    int* d_context_lens;
    size_t block_tables_size = batch_size * max_blocks_per_seq * sizeof(int);
    size_t context_lens_size = batch_size * sizeof(int);

    cudaError_t err = cudaMalloc(&d_block_tables, block_tables_size);
    if (err != cudaSuccess) return BINFER_ERROR_CUDA;

    err = cudaMalloc(&d_context_lens, context_lens_size);
    if (err != cudaSuccess) {
        cudaFree(d_block_tables);
        return BINFER_ERROR_CUDA;
    }

    // Copy from host to device
    err = cudaMemcpy(d_block_tables, block_tables, block_tables_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_block_tables);
        cudaFree(d_context_lens);
        return BINFER_ERROR_CUDA;
    }

    err = cudaMemcpy(d_context_lens, context_lens, context_lens_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_block_tables);
        cudaFree(d_context_lens);
        return BINFER_ERROR_CUDA;
    }

    dim3 grid(batch_size, num_heads);
    dim3 block(128);  // 4 warps
    const int num_warps = 4;
    size_t smem_size = num_warps * sizeof(float) * (2 + head_dim);
    cudaStream_t stream = GET_CURRENT_STREAM();

    switch (head_dim) {
        case 64:
            paged_attention_kernel_f16<64><<<grid, block, smem_size, stream>>>(
                (const __half*)Q, (const __half*)K_cache, (const __half*)V_cache,
                (__half*)O, d_block_tables, d_context_lens,
                batch_size, num_heads, num_kv_heads, max_blocks_per_seq, softmax_scale
            );
            break;
        case 128:
            paged_attention_kernel_f16<128><<<grid, block, smem_size, stream>>>(
                (const __half*)Q, (const __half*)K_cache, (const __half*)V_cache,
                (__half*)O, d_block_tables, d_context_lens,
                batch_size, num_heads, num_kv_heads, max_blocks_per_seq, softmax_scale
            );
            break;
        default:
            cudaFree(d_block_tables);
            cudaFree(d_context_lens);
            return BINFER_ERROR_INVALID_ARGUMENT;
    }

    err = cudaGetLastError();

    // Free temporary device memory
    cudaFree(d_block_tables);
    cudaFree(d_context_lens);

    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Kernel to copy new K/V values into paged cache blocks
__global__ void paged_kv_cache_update_kernel_f16(
    __half* __restrict__ K_cache,           // [num_blocks, block_size, num_kv_heads, head_dim]
    __half* __restrict__ V_cache,           // [num_blocks, block_size, num_kv_heads, head_dim]
    const __half* __restrict__ K_new,       // [num_seqs, num_new_tokens, num_kv_heads, head_dim]
    const __half* __restrict__ V_new,       // [num_seqs, num_new_tokens, num_kv_heads, head_dim]
    const int* __restrict__ block_tables,   // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,   // [num_seqs] - length BEFORE adding new tokens
    int num_seqs,
    int num_new_tokens,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq
) {
    // Grid: (num_seqs, num_new_tokens, num_kv_heads)
    const int seq_idx = blockIdx.x;
    const int token_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int dim_idx = threadIdx.x;

    if (dim_idx >= head_dim) return;
    if (token_idx >= num_new_tokens) return;

    // Calculate destination position in cache
    const int context_len = context_lens[seq_idx];
    const int dest_pos = context_len + token_idx;
    const int block_idx = dest_pos / BLOCK_SIZE;
    const int slot_idx = dest_pos % BLOCK_SIZE;

    // Get physical block ID
    const int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];

    // Calculate indices
    const int src_idx = ((seq_idx * num_new_tokens + token_idx) * num_kv_heads + head_idx) * head_dim + dim_idx;
    const int dst_idx = ((physical_block * BLOCK_SIZE + slot_idx) * num_kv_heads + head_idx) * head_dim + dim_idx;

    K_cache[dst_idx] = K_new[src_idx];
    V_cache[dst_idx] = V_new[src_idx];
}

extern "C" BinferError binfer_paged_kv_cache_update_f16(
    void* K_cache,
    void* V_cache,
    const void* K_new,
    const void* V_new,
    const int* block_tables,  // host pointer
    const int* context_lens,  // host pointer
    int num_seqs,
    int num_new_tokens,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq
) {
    // Allocate device memory for block_tables and context_lens
    int* d_block_tables;
    int* d_context_lens;
    size_t block_tables_size = num_seqs * max_blocks_per_seq * sizeof(int);
    size_t context_lens_size = num_seqs * sizeof(int);

    cudaError_t err = cudaMalloc(&d_block_tables, block_tables_size);
    if (err != cudaSuccess) return BINFER_ERROR_CUDA;

    err = cudaMalloc(&d_context_lens, context_lens_size);
    if (err != cudaSuccess) {
        cudaFree(d_block_tables);
        return BINFER_ERROR_CUDA;
    }

    // Copy from host to device
    err = cudaMemcpy(d_block_tables, block_tables, block_tables_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_block_tables);
        cudaFree(d_context_lens);
        return BINFER_ERROR_CUDA;
    }

    err = cudaMemcpy(d_context_lens, context_lens, context_lens_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_block_tables);
        cudaFree(d_context_lens);
        return BINFER_ERROR_CUDA;
    }

    dim3 grid(num_seqs, num_new_tokens, num_kv_heads);
    dim3 block(head_dim);
    cudaStream_t stream = GET_CURRENT_STREAM();

    paged_kv_cache_update_kernel_f16<<<grid, block, 0, stream>>>(
        (__half*)K_cache, (__half*)V_cache,
        (const __half*)K_new, (const __half*)V_new,
        d_block_tables, d_context_lens,
        num_seqs, num_new_tokens, num_kv_heads, head_dim, max_blocks_per_seq
    );

    err = cudaGetLastError();

    // Free temporary device memory
    cudaFree(d_block_tables);
    cudaFree(d_context_lens);

    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// BF16 Versions of Paged Attention Kernels
// ============================================================================

// Paged attention kernel for decode phase (BF16)
template<int HEAD_DIM>
__global__ void paged_attention_kernel_bf16(
    const __nv_bfloat16* __restrict__ Q,           // [num_seqs, num_heads, head_dim]
    const __nv_bfloat16* __restrict__ K_cache,     // [num_blocks, block_size, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ V_cache,     // [num_blocks, block_size, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ O,                 // [num_seqs, num_heads, head_dim]
    const int* __restrict__ block_tables,   // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,   // [num_seqs]
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int max_blocks_per_seq,
    float softmax_scale
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q
    float q_local[HEAD_DIM / WARP_SIZE];
    const __nv_bfloat16* q_ptr = Q + (seq_idx * num_heads + head_idx) * HEAD_DIM;
    for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
        q_local[d / WARP_SIZE] = bf16_to_float(q_ptr[d]);
    }

    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    float acc[HEAD_DIM / WARP_SIZE] = {0.0f};

    for (int token_idx = warp_id; token_idx < context_len; token_idx += num_warps) {
        const int block_idx = token_idx / BLOCK_SIZE;
        const int slot_idx = token_idx % BLOCK_SIZE;
        const int physical_block = seq_block_table[block_idx];

        const int kv_offset = ((physical_block * BLOCK_SIZE + slot_idx) * num_kv_heads + kv_head_idx) * HEAD_DIM;
        const __nv_bfloat16* k_ptr = K_cache + kv_offset;
        const __nv_bfloat16* v_ptr = V_cache + kv_offset;

        float score = 0.0f;
        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
            float k_val = bf16_to_float(k_ptr[d]);
            score += q_local[d / WARP_SIZE] * k_val;
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            score += __shfl_down_sync(0xffffffff, score, offset);
        }
        score = __shfl_sync(0xffffffff, score, 0);
        score *= softmax_scale;

        float old_max = max_score;
        max_score = fmaxf(max_score, score);
        float scale = expf(old_max - max_score);
        sum_exp = sum_exp * scale + expf(score - max_score);

        for (int i = 0; i < HEAD_DIM / WARP_SIZE; i++) {
            acc[i] *= scale;
        }

        float attn_weight = expf(score - max_score);
        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
            float v_val = bf16_to_float(v_ptr[d]);
            acc[d / WARP_SIZE] += attn_weight * v_val;
        }
    }

    extern __shared__ char smem[];
    float* warp_max = (float*)smem;
    float* warp_sum = warp_max + num_warps;
    float* warp_acc = warp_sum + num_warps;

    if (lane_id == 0) {
        warp_max[warp_id] = max_score;
        warp_sum[warp_id] = sum_exp;
    }
    for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
        warp_acc[warp_id * HEAD_DIM + d] = acc[d / WARP_SIZE];
    }
    __syncthreads();

    if (warp_id == 0) {
        // Calculate how many warps actually processed tokens
        // Warp w handles tokens at positions w, w+num_warps, w+2*num_warps, ...
        // So warp w processed tokens if w < context_len
        const int active_warps = min(num_warps, context_len);

        float final_max = -FLT_MAX;
        for (int w = 0; w < active_warps; w++) {
            final_max = fmaxf(final_max, warp_max[w]);
        }

        float final_sum = 0.0f;
        for (int w = 0; w < active_warps; w++) {
            final_sum += warp_sum[w] * expf(warp_max[w] - final_max);
        }

        float final_acc[HEAD_DIM / WARP_SIZE] = {0.0f};
        for (int w = 0; w < active_warps; w++) {
            float scale = expf(warp_max[w] - final_max);
            for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                final_acc[d / WARP_SIZE] += warp_acc[w * HEAD_DIM + d] * scale;
            }
        }

        __nv_bfloat16* o_ptr = O + (seq_idx * num_heads + head_idx) * HEAD_DIM;
        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
            float o_val = final_acc[d / WARP_SIZE] / final_sum;
            o_ptr[d] = float_to_bf16(o_val);
        }
    }
}

extern "C" BinferError binfer_paged_attention_bf16(
    const void* Q,
    const void* K_cache,
    const void* V_cache,
    void* O,
    const int* block_tables,  // host pointer
    const int* context_lens,  // host pointer
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,   // stride of block_tables (passed from caller)
    float softmax_scale
) {
    if (block_size != BLOCK_SIZE) {
        return BINFER_ERROR_INVALID_ARGUMENT;
    }

    // Allocate device memory for block_tables and context_lens
    int* d_block_tables;
    int* d_context_lens;
    size_t block_tables_size = batch_size * max_blocks_per_seq * sizeof(int);
    size_t context_lens_size = batch_size * sizeof(int);

    cudaError_t err = cudaMalloc(&d_block_tables, block_tables_size);
    if (err != cudaSuccess) return BINFER_ERROR_CUDA;

    err = cudaMalloc(&d_context_lens, context_lens_size);
    if (err != cudaSuccess) {
        cudaFree(d_block_tables);
        return BINFER_ERROR_CUDA;
    }

    // Copy from host to device
    err = cudaMemcpy(d_block_tables, block_tables, block_tables_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_block_tables);
        cudaFree(d_context_lens);
        return BINFER_ERROR_CUDA;
    }

    err = cudaMemcpy(d_context_lens, context_lens, context_lens_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_block_tables);
        cudaFree(d_context_lens);
        return BINFER_ERROR_CUDA;
    }

    dim3 grid(batch_size, num_heads);
    dim3 block(128);
    const int num_warps = 4;
    size_t smem_size = num_warps * sizeof(float) * (2 + head_dim);
    cudaStream_t stream = GET_CURRENT_STREAM();

    switch (head_dim) {
        case 64:
            paged_attention_kernel_bf16<64><<<grid, block, smem_size, stream>>>(
                (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K_cache, (const __nv_bfloat16*)V_cache,
                (__nv_bfloat16*)O, d_block_tables, d_context_lens,
                batch_size, num_heads, num_kv_heads, max_blocks_per_seq, softmax_scale
            );
            break;
        case 128:
            paged_attention_kernel_bf16<128><<<grid, block, smem_size, stream>>>(
                (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K_cache, (const __nv_bfloat16*)V_cache,
                (__nv_bfloat16*)O, d_block_tables, d_context_lens,
                batch_size, num_heads, num_kv_heads, max_blocks_per_seq, softmax_scale
            );
            break;
        default:
            cudaFree(d_block_tables);
            cudaFree(d_context_lens);
            return BINFER_ERROR_INVALID_ARGUMENT;
    }

    err = cudaGetLastError();

    // Free temporary device memory
    cudaFree(d_block_tables);
    cudaFree(d_context_lens);

    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Kernel to copy new K/V values into paged cache blocks (BF16)
__global__ void paged_kv_cache_update_kernel_bf16(
    __nv_bfloat16* __restrict__ K_cache,
    __nv_bfloat16* __restrict__ V_cache,
    const __nv_bfloat16* __restrict__ K_new,
    const __nv_bfloat16* __restrict__ V_new,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int num_seqs,
    int num_new_tokens,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq
) {
    const int seq_idx = blockIdx.x;
    const int token_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int dim_idx = threadIdx.x;

    if (dim_idx >= head_dim) return;
    if (token_idx >= num_new_tokens) return;

    const int context_len = context_lens[seq_idx];
    const int dest_pos = context_len + token_idx;
    const int block_idx = dest_pos / BLOCK_SIZE;
    const int slot_idx = dest_pos % BLOCK_SIZE;

    const int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];

    const int src_idx = ((seq_idx * num_new_tokens + token_idx) * num_kv_heads + head_idx) * head_dim + dim_idx;
    const int dst_idx = ((physical_block * BLOCK_SIZE + slot_idx) * num_kv_heads + head_idx) * head_dim + dim_idx;

    K_cache[dst_idx] = K_new[src_idx];
    V_cache[dst_idx] = V_new[src_idx];
}

extern "C" BinferError binfer_paged_kv_cache_update_bf16(
    void* K_cache,
    void* V_cache,
    const void* K_new,
    const void* V_new,
    const int* block_tables,  // host pointer
    const int* context_lens,  // host pointer
    int num_seqs,
    int num_new_tokens,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq
) {
    // Allocate device memory for block_tables and context_lens
    int* d_block_tables;
    int* d_context_lens;
    size_t block_tables_size = num_seqs * max_blocks_per_seq * sizeof(int);
    size_t context_lens_size = num_seqs * sizeof(int);

    cudaError_t err = cudaMalloc(&d_block_tables, block_tables_size);
    if (err != cudaSuccess) return BINFER_ERROR_CUDA;

    err = cudaMalloc(&d_context_lens, context_lens_size);
    if (err != cudaSuccess) {
        cudaFree(d_block_tables);
        return BINFER_ERROR_CUDA;
    }

    // Copy from host to device
    err = cudaMemcpy(d_block_tables, block_tables, block_tables_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_block_tables);
        cudaFree(d_context_lens);
        return BINFER_ERROR_CUDA;
    }

    err = cudaMemcpy(d_context_lens, context_lens, context_lens_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_block_tables);
        cudaFree(d_context_lens);
        return BINFER_ERROR_CUDA;
    }

    dim3 grid(num_seqs, num_new_tokens, num_kv_heads);
    dim3 block(head_dim);
    cudaStream_t stream = GET_CURRENT_STREAM();

    paged_kv_cache_update_kernel_bf16<<<grid, block, 0, stream>>>(
        (__nv_bfloat16*)K_cache, (__nv_bfloat16*)V_cache,
        (const __nv_bfloat16*)K_new, (const __nv_bfloat16*)V_new,
        d_block_tables, d_context_lens,
        num_seqs, num_new_tokens, num_kv_heads, head_dim, max_blocks_per_seq
    );

    err = cudaGetLastError();

    // Free temporary device memory
    cudaFree(d_block_tables);
    cudaFree(d_context_lens);

    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// Paged Attention with Sinks (for GPT-OSS)
// ============================================================================

// Paged attention kernel with sinks for decode phase (BF16)
// Sinks add an extra "attention sink" position that absorbs unused attention
template<int HEAD_DIM>
__global__ void paged_attention_with_sinks_kernel_bf16(
    const __nv_bfloat16* __restrict__ Q,           // [num_seqs, num_heads, head_dim]
    const __nv_bfloat16* __restrict__ K_cache,     // [num_blocks, block_size, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ V_cache,     // [num_blocks, block_size, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ sinks,       // [num_heads] - per-head learnable sink values
    __nv_bfloat16* __restrict__ O,                 // [num_seqs, num_heads, head_dim]
    const int* __restrict__ block_tables,   // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,   // [num_seqs]
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int max_blocks_per_seq,
    float softmax_scale
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q
    float q_local[HEAD_DIM / WARP_SIZE];
    const __nv_bfloat16* q_ptr = Q + (seq_idx * num_heads + head_idx) * HEAD_DIM;
    for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
        q_local[d / WARP_SIZE] = bf16_to_float(q_ptr[d]);
    }

    // Load sink value for this head
    float sink_logit = 0.0f;
    if (tid == 0) {
        sink_logit = bf16_to_float(sinks[head_idx]);
    }
    sink_logit = __shfl_sync(0xffffffff, sink_logit, 0);

    float max_score = sink_logit;  // Initialize with sink
    float sum_exp = 0.0f;
    float acc[HEAD_DIM / WARP_SIZE] = {0.0f};

    // Process each token in context
    for (int token_idx = warp_id; token_idx < context_len; token_idx += num_warps) {
        const int block_idx = token_idx / BLOCK_SIZE;
        const int slot_idx = token_idx % BLOCK_SIZE;
        const int physical_block = seq_block_table[block_idx];

        const int kv_offset = ((physical_block * BLOCK_SIZE + slot_idx) * num_kv_heads + kv_head_idx) * HEAD_DIM;
        const __nv_bfloat16* k_ptr = K_cache + kv_offset;
        const __nv_bfloat16* v_ptr = V_cache + kv_offset;

        float score = 0.0f;
        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
            float k_val = bf16_to_float(k_ptr[d]);
            score += q_local[d / WARP_SIZE] * k_val;
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            score += __shfl_down_sync(0xffffffff, score, offset);
        }
        score = __shfl_sync(0xffffffff, score, 0);
        score *= softmax_scale;

        float old_max = max_score;
        max_score = fmaxf(max_score, score);
        float scale = expf(old_max - max_score);
        sum_exp = sum_exp * scale + expf(score - max_score);

        for (int i = 0; i < HEAD_DIM / WARP_SIZE; i++) {
            acc[i] *= scale;
        }

        float attn_weight = expf(score - max_score);
        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
            float v_val = bf16_to_float(v_ptr[d]);
            acc[d / WARP_SIZE] += attn_weight * v_val;
        }
    }

    // Add sink contribution to sum_exp (but not to acc - sink has no V)
    // The sink probability mass is computed but doesn't contribute to output
    {
        float old_max = max_score;
        // max_score already includes sink_logit from initialization
        float scale = expf(old_max - max_score);  // This is 1.0 since max_score didn't change
        sum_exp = sum_exp * scale + expf(sink_logit - max_score);
        // acc already scaled correctly
    }

    // Reduce across warps using shared memory
    extern __shared__ char smem[];
    float* warp_max = (float*)smem;
    float* warp_sum = warp_max + num_warps;
    float* warp_acc = warp_sum + num_warps;

    if (lane_id == 0) {
        warp_max[warp_id] = max_score;
        warp_sum[warp_id] = sum_exp;
    }
    for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
        warp_acc[warp_id * HEAD_DIM + d] = acc[d / WARP_SIZE];
    }
    __syncthreads();

    if (warp_id == 0) {
        const int active_warps = min(num_warps, context_len);

        float final_max = -FLT_MAX;
        for (int w = 0; w < active_warps; w++) {
            final_max = fmaxf(final_max, warp_max[w]);
        }
        // Also consider sink in final max (for correctness when context_len is small)
        final_max = fmaxf(final_max, sink_logit);

        float final_sum = 0.0f;
        for (int w = 0; w < active_warps; w++) {
            final_sum += warp_sum[w] * expf(warp_max[w] - final_max);
        }
        // Add sink contribution to final sum
        final_sum += expf(sink_logit - final_max);

        float final_acc[HEAD_DIM / WARP_SIZE] = {0.0f};
        for (int w = 0; w < active_warps; w++) {
            float scale = expf(warp_max[w] - final_max);
            for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                final_acc[d / WARP_SIZE] += warp_acc[w * HEAD_DIM + d] * scale;
            }
        }

        __nv_bfloat16* o_ptr = O + (seq_idx * num_heads + head_idx) * HEAD_DIM;
        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
            float o_val = final_acc[d / WARP_SIZE] / final_sum;
            o_ptr[d] = float_to_bf16(o_val);
        }
    }
}

extern "C" BinferError binfer_paged_attention_with_sinks_bf16(
    const void* Q,
    const void* K_cache,
    const void* V_cache,
    const void* sinks,
    void* O,
    const int* block_tables,
    const int* context_lens,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    float softmax_scale
) {
    if (block_size != BLOCK_SIZE) {
        return BINFER_ERROR_INVALID_ARGUMENT;
    }

    int* d_block_tables;
    int* d_context_lens;
    size_t block_tables_size = batch_size * max_blocks_per_seq * sizeof(int);
    size_t context_lens_size = batch_size * sizeof(int);

    cudaError_t err = cudaMalloc(&d_block_tables, block_tables_size);
    if (err != cudaSuccess) return BINFER_ERROR_CUDA;

    err = cudaMalloc(&d_context_lens, context_lens_size);
    if (err != cudaSuccess) {
        cudaFree(d_block_tables);
        return BINFER_ERROR_CUDA;
    }

    err = cudaMemcpy(d_block_tables, block_tables, block_tables_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_block_tables);
        cudaFree(d_context_lens);
        return BINFER_ERROR_CUDA;
    }

    err = cudaMemcpy(d_context_lens, context_lens, context_lens_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_block_tables);
        cudaFree(d_context_lens);
        return BINFER_ERROR_CUDA;
    }

    dim3 grid(batch_size, num_heads);
    dim3 block(128);
    const int num_warps = 4;
    size_t smem_size = num_warps * sizeof(float) * (2 + head_dim);
    cudaStream_t stream = GET_CURRENT_STREAM();

    switch (head_dim) {
        case 64:
            paged_attention_with_sinks_kernel_bf16<64><<<grid, block, smem_size, stream>>>(
                (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K_cache, (const __nv_bfloat16*)V_cache,
                (const __nv_bfloat16*)sinks, (__nv_bfloat16*)O, d_block_tables, d_context_lens,
                batch_size, num_heads, num_kv_heads, max_blocks_per_seq, softmax_scale
            );
            break;
        case 128:
            paged_attention_with_sinks_kernel_bf16<128><<<grid, block, smem_size, stream>>>(
                (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K_cache, (const __nv_bfloat16*)V_cache,
                (const __nv_bfloat16*)sinks, (__nv_bfloat16*)O, d_block_tables, d_context_lens,
                batch_size, num_heads, num_kv_heads, max_blocks_per_seq, softmax_scale
            );
            break;
        default:
            cudaFree(d_block_tables);
            cudaFree(d_context_lens);
            return BINFER_ERROR_INVALID_ARGUMENT;
    }

    err = cudaGetLastError();

    cudaFree(d_block_tables);
    cudaFree(d_context_lens);

    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// Device-pointer variants (no internal malloc/free for pre-uploaded buffers)
// ============================================================================

// KV cache update with device pointers - avoids per-call malloc/free
extern "C" BinferError binfer_paged_kv_cache_update_bf16_device(
    void* K_cache,
    void* V_cache,
    const void* K_new,
    const void* V_new,
    const void* d_block_tables,  // device pointer (already uploaded)
    const void* d_context_lens,  // device pointer (already uploaded)
    int num_seqs,
    int num_new_tokens,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq
) {
    dim3 grid(num_seqs, num_new_tokens, num_kv_heads);
    dim3 block(head_dim);
    cudaStream_t stream = GET_CURRENT_STREAM();

    paged_kv_cache_update_kernel_bf16<<<grid, block, 0, stream>>>(
        (__nv_bfloat16*)K_cache, (__nv_bfloat16*)V_cache,
        (const __nv_bfloat16*)K_new, (const __nv_bfloat16*)V_new,
        (const int*)d_block_tables, (const int*)d_context_lens,
        num_seqs, num_new_tokens, num_kv_heads, head_dim, max_blocks_per_seq
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Paged attention with sinks using device pointers
extern "C" BinferError binfer_paged_attention_with_sinks_bf16_device(
    const void* Q,
    const void* K_cache,
    const void* V_cache,
    const void* sinks,
    void* O,
    const void* d_block_tables,  // device pointer (already uploaded)
    const void* d_context_lens,  // device pointer (already uploaded)
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    float softmax_scale
) {
    dim3 grid(batch_size, num_heads);
    dim3 block(128);
    const int num_warps = 4;
    size_t smem_size = num_warps * sizeof(float) * (2 + head_dim);
    cudaStream_t stream = GET_CURRENT_STREAM();

    cudaError_t err;
    switch (head_dim) {
        case 64:
            paged_attention_with_sinks_kernel_bf16<64><<<grid, block, smem_size, stream>>>(
                (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K_cache, (const __nv_bfloat16*)V_cache,
                (const __nv_bfloat16*)sinks, (__nv_bfloat16*)O,
                (const int*)d_block_tables, (const int*)d_context_lens,
                batch_size, num_heads, num_kv_heads, max_blocks_per_seq, softmax_scale
            );
            break;
        case 128:
            paged_attention_with_sinks_kernel_bf16<128><<<grid, block, smem_size, stream>>>(
                (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K_cache, (const __nv_bfloat16*)V_cache,
                (const __nv_bfloat16*)sinks, (__nv_bfloat16*)O,
                (const int*)d_block_tables, (const int*)d_context_lens,
                batch_size, num_heads, num_kv_heads, max_blocks_per_seq, softmax_scale
            );
            break;
        default:
            return BINFER_ERROR_INVALID_ARGUMENT;
    }

    err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Regular paged attention using device pointers
extern "C" BinferError binfer_paged_attention_bf16_device(
    const void* Q,
    const void* K_cache,
    const void* V_cache,
    void* O,
    const void* d_block_tables,  // device pointer (already uploaded)
    const void* d_context_lens,  // device pointer (already uploaded)
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    float softmax_scale
) {
    dim3 grid(batch_size, num_heads);
    dim3 block(128);
    const int num_warps = 4;
    size_t smem_size = num_warps * sizeof(float) * (2 + head_dim);
    cudaStream_t stream = GET_CURRENT_STREAM();

    cudaError_t err;
    switch (head_dim) {
        case 64:
            paged_attention_kernel_bf16<64><<<grid, block, smem_size, stream>>>(
                (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K_cache, (const __nv_bfloat16*)V_cache,
                (__nv_bfloat16*)O, (const int*)d_block_tables, (const int*)d_context_lens,
                batch_size, num_heads, num_kv_heads, max_blocks_per_seq, softmax_scale
            );
            break;
        case 128:
            paged_attention_kernel_bf16<128><<<grid, block, smem_size, stream>>>(
                (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K_cache, (const __nv_bfloat16*)V_cache,
                (__nv_bfloat16*)O, (const int*)d_block_tables, (const int*)d_context_lens,
                batch_size, num_heads, num_kv_heads, max_blocks_per_seq, softmax_scale
            );
            break;
        default:
            return BINFER_ERROR_INVALID_ARGUMENT;
    }

    err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}


