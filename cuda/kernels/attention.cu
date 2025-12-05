// Flash Attention implementation
// Based on FlashAttention-2 algorithm by Tri Dao
// Reference: https://github.com/Dao-AILab/flash-attention

#include "attention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>

// Block sizes for flash attention
// Tuned for H100 - adjust for other GPUs
constexpr int BLOCK_M = 128;  // Query block size
constexpr int BLOCK_N = 64;   // Key/Value block size
constexpr int BLOCK_K = 64;   // Head dimension block

// Warp size
constexpr int WARP_SIZE = 32;

// Helper: online softmax state
struct SoftmaxState {
    float max_val;
    float sum_exp;
};

__device__ __forceinline__ SoftmaxState softmax_update(
    SoftmaxState state, float new_max, float new_sum
) {
    float max_new = fmaxf(state.max_val, new_max);
    float scale_old = expf(state.max_val - max_new);
    float scale_new = expf(new_max - max_new);
    return {
        max_new,
        state.sum_exp * scale_old + new_sum * scale_new
    };
}

// Simplified flash attention kernel (not fully optimized but functional)
// Full optimization would use tensor cores and more sophisticated tiling
template<int HEAD_DIM, bool IS_CAUSAL>
__global__ void flash_attention_kernel_f16(
    const __half* __restrict__ Q,      // [batch, seq_q, num_heads, head_dim]
    const __half* __restrict__ K,      // [batch, kv_stride, num_kv_heads, head_dim]
    const __half* __restrict__ V,      // [batch, kv_stride, num_kv_heads, head_dim]
    __half* __restrict__ O,            // [batch, seq_q, num_heads, head_dim]
    int batch_size,
    int seq_q,
    int seq_kv,         // Valid KV length
    int kv_stride,      // Stride in KV cache (max_seq_len for cached, seq_kv for fresh)
    int q_offset,       // Offset for absolute query position (for causal mask in decode)
    int num_heads,
    int num_kv_heads,
    float softmax_scale
) {
    // Grid: (batch, num_heads, ceil(seq_q / BLOCK_M))
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.z;

    // GQA: map query head to KV head
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_q);

    // Thread position within block
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Shared memory for K, V tiles
    extern __shared__ char smem[];
    __half* K_tile = (__half*)smem;
    __half* V_tile = K_tile + BLOCK_N * HEAD_DIM;

    // Per-thread accumulators
    float O_acc[HEAD_DIM / WARP_SIZE] = {0.0f};
    SoftmaxState softmax_state = {-FLT_MAX, 0.0f};

    // Pointers to this batch/head - use kv_stride for K,V indexing
    const __half* Q_ptr = Q + (batch_idx * seq_q * num_heads + head_idx) * HEAD_DIM;
    const __half* K_ptr = K + (batch_idx * kv_stride * num_kv_heads + kv_head_idx) * HEAD_DIM;
    const __half* V_ptr = V + (batch_idx * kv_stride * num_kv_heads + kv_head_idx) * HEAD_DIM;
    __half* O_ptr = O + (batch_idx * seq_q * num_heads + head_idx) * HEAD_DIM;

    // Process each query position in this block
    for (int q_pos = q_start + warp_id; q_pos < q_end; q_pos += blockDim.x / WARP_SIZE) {
        // Load Q for this position
        float q_local[HEAD_DIM / WARP_SIZE];
        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
            q_local[d / WARP_SIZE] = __half2float(Q_ptr[q_pos * num_heads * HEAD_DIM + d]);
        }

        // Reset accumulators for this query
        for (int i = 0; i < HEAD_DIM / WARP_SIZE; i++) O_acc[i] = 0.0f;
        softmax_state = {-FLT_MAX, 0.0f};

        // Determine KV range (causal masking uses absolute position)
        // Absolute query position = q_pos + q_offset
        int abs_q_pos = q_pos + q_offset;
        int kv_end = IS_CAUSAL ? min(abs_q_pos + 1, seq_kv) : seq_kv;

        // Iterate over KV blocks
        for (int kv_start = 0; kv_start < kv_end; kv_start += BLOCK_N) {
            int kv_block_end = min(kv_start + BLOCK_N, kv_end);

            // Compute attention scores for this block
            float scores[BLOCK_N];
            float block_max = -FLT_MAX;

            for (int kv_pos = kv_start; kv_pos < kv_block_end; kv_pos++) {
                // Dot product Q @ K^T
                float score = 0.0f;
                for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                    float k_val = __half2float(K_ptr[kv_pos * num_kv_heads * HEAD_DIM + d]);
                    score += q_local[d / WARP_SIZE] * k_val;
                }
                // Warp reduce
                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                    score += __shfl_down_sync(0xffffffff, score, offset);
                }
                score = __shfl_sync(0xffffffff, score, 0);

                // Apply causal mask (use absolute position)
                if (IS_CAUSAL && kv_pos > abs_q_pos) {
                    score = -FLT_MAX;
                }

                scores[kv_pos - kv_start] = score * softmax_scale;
                block_max = fmaxf(block_max, scores[kv_pos - kv_start]);
            }

            // Compute softmax for this block
            float block_sum = 0.0f;
            for (int i = 0; i < kv_block_end - kv_start; i++) {
                scores[i] = expf(scores[i] - block_max);
                block_sum += scores[i];
            }

            // Update running softmax state
            float old_max = softmax_state.max_val;
            softmax_state = softmax_update(softmax_state, block_max, block_sum);

            // Rescale previous output accumulator
            float scale = expf(old_max - softmax_state.max_val);
            for (int i = 0; i < HEAD_DIM / WARP_SIZE; i++) {
                O_acc[i] *= scale;
            }

            // Accumulate V weighted by attention
            for (int kv_pos = kv_start; kv_pos < kv_block_end; kv_pos++) {
                float attn_weight = scores[kv_pos - kv_start] * expf(block_max - softmax_state.max_val);
                for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                    float v_val = __half2float(V_ptr[kv_pos * num_kv_heads * HEAD_DIM + d]);
                    O_acc[d / WARP_SIZE] += attn_weight * v_val;
                }
            }
        }

        // Normalize by softmax sum and write output
        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
            float o_val = O_acc[d / WARP_SIZE] / softmax_state.sum_exp;
            O_ptr[q_pos * num_heads * HEAD_DIM + d] = __float2half(o_val);
        }
    }
}

// Dispatch based on head dimension
template<bool IS_CAUSAL>
BinferError flash_attention_dispatch(
    const void* Q, const void* K, const void* V, void* O,
    int batch_size, int seq_q, int seq_kv, int kv_stride, int q_offset,
    int num_heads, int num_kv_heads, int head_dim,
    float softmax_scale
) {
    dim3 grid(batch_size, num_heads, (seq_q + BLOCK_M - 1) / BLOCK_M);
    dim3 block(256);
    size_t smem_size = 2 * BLOCK_N * head_dim * sizeof(__half);

    switch (head_dim) {
        case 64:
            flash_attention_kernel_f16<64, IS_CAUSAL><<<grid, block, smem_size>>>(
                (const __half*)Q, (const __half*)K, (const __half*)V, (__half*)O,
                batch_size, seq_q, seq_kv, kv_stride, q_offset, num_heads, num_kv_heads, softmax_scale
            );
            break;
        case 128:
            flash_attention_kernel_f16<128, IS_CAUSAL><<<grid, block, smem_size>>>(
                (const __half*)Q, (const __half*)K, (const __half*)V, (__half*)O,
                batch_size, seq_q, seq_kv, kv_stride, q_offset, num_heads, num_kv_heads, softmax_scale
            );
            break;
        default:
            return BINFER_ERROR_INVALID_ARGUMENT;
    }

    return cudaGetLastError() == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_flash_attention_f16(
    const void* Q, const void* K, const void* V, void* O,
    int batch_size, int seq_q, int seq_kv, int kv_stride, int q_offset,
    int num_heads, int num_kv_heads, int head_dim,
    float softmax_scale, bool is_causal
) {
    if (is_causal) {
        return flash_attention_dispatch<true>(
            Q, K, V, O, batch_size, seq_q, seq_kv, kv_stride, q_offset,
            num_heads, num_kv_heads, head_dim, softmax_scale
        );
    } else {
        return flash_attention_dispatch<false>(
            Q, K, V, O, batch_size, seq_q, seq_kv, kv_stride, q_offset,
            num_heads, num_kv_heads, head_dim, softmax_scale
        );
    }
}

// KV cache update kernel
__global__ void kv_cache_update_kernel_f16(
    __half* __restrict__ K_cache,
    __half* __restrict__ V_cache,
    const __half* __restrict__ K_new,
    const __half* __restrict__ V_new,
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

    // Calculate indices
    const int new_idx = ((batch_idx * seq_new + seq_idx) * num_kv_heads + head_idx) * head_dim + dim_idx;
    const int cache_idx = ((batch_idx * max_seq_len + cache_pos) * num_kv_heads + head_idx) * head_dim + dim_idx;

    K_cache[cache_idx] = K_new[new_idx];
    V_cache[cache_idx] = V_new[new_idx];
}

extern "C" BinferError binfer_kv_cache_update_f16(
    void* K_cache, void* V_cache,
    const void* K_new, const void* V_new,
    int batch_size, int cache_offset, int seq_new,
    int num_kv_heads, int head_dim, int max_seq_len
) {
    dim3 grid(batch_size, seq_new, num_kv_heads);
    dim3 block(head_dim);

    kv_cache_update_kernel_f16<<<grid, block>>>(
        (__half*)K_cache, (__half*)V_cache,
        (const __half*)K_new, (const __half*)V_new,
        batch_size, cache_offset, seq_new,
        num_kv_heads, head_dim, max_seq_len
    );

    return cudaGetLastError() == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Flash attention with KV cache (for decoding)
extern "C" BinferError binfer_flash_attention_with_cache_f16(
    const void* Q, const void* K_cache, const void* V_cache, void* O,
    int batch_size, int cache_seq_len, int max_seq_len, int q_offset,
    int num_heads, int num_kv_heads, int head_dim,
    float softmax_scale, bool is_causal
) {
    // For decode, seq_q = 1, kv_stride = max_seq_len, q_offset = absolute position
    return binfer_flash_attention_f16(
        Q, K_cache, V_cache, O,
        batch_size, 1, cache_seq_len, max_seq_len, q_offset,
        num_heads, num_kv_heads, head_dim,
        softmax_scale, is_causal
    );
}
