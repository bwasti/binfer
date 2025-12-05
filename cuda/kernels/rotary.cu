#include "binfer.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Rotary Position Embedding kernel
// Applies RoPE to Q and K tensors in-place
// Uses the "rotary" style: split head_dim in half, apply rotation
__global__ void rotary_embedding_kernel_f16(
    __half* __restrict__ q,
    __half* __restrict__ k,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int position_offset
) {
    // Grid: (batch, seq, head)
    // Each thread handles one pair of elements
    const int batch_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int pair_idx = threadIdx.x;  // 0 to head_dim/2 - 1

    if (pair_idx >= head_dim / 2) return;

    const int half_dim = head_dim / 2;
    const int pos = seq_idx + position_offset;

    // Get cos/sin for this position and dimension
    // cos/sin shape: [max_seq_len, head_dim/2]
    const float cos_val = __half2float(cos[pos * half_dim + pair_idx]);
    const float sin_val = __half2float(sin[pos * half_dim + pair_idx]);

    // Apply to Q
    if (head_idx < num_heads) {
        // Q layout: [batch, seq, num_heads, head_dim]
        const int q_offset = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;

        float q0 = __half2float(q[q_offset + pair_idx]);
        float q1 = __half2float(q[q_offset + pair_idx + half_dim]);

        // Rotation: [cos, -sin; sin, cos] @ [q0, q1]
        float q0_new = q0 * cos_val - q1 * sin_val;
        float q1_new = q0 * sin_val + q1 * cos_val;

        q[q_offset + pair_idx] = __float2half(q0_new);
        q[q_offset + pair_idx + half_dim] = __float2half(q1_new);
    }

    // Apply to K (only for KV heads)
    if (head_idx < num_kv_heads) {
        // K layout: [batch, seq, num_kv_heads, head_dim]
        const int k_offset = ((batch_idx * seq_len + seq_idx) * num_kv_heads + head_idx) * head_dim;

        float k0 = __half2float(k[k_offset + pair_idx]);
        float k1 = __half2float(k[k_offset + pair_idx + half_dim]);

        float k0_new = k0 * cos_val - k1 * sin_val;
        float k1_new = k0 * sin_val + k1 * cos_val;

        k[k_offset + pair_idx] = __float2half(k0_new);
        k[k_offset + pair_idx + half_dim] = __float2half(k1_new);
    }
}

extern "C" BinferError binfer_rotary_embedding_f16(
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
    // Use max of num_heads and num_kv_heads for grid z dimension
    const int max_heads = num_heads > num_kv_heads ? num_heads : num_kv_heads;

    dim3 grid(batch_size, seq_len, max_heads);
    dim3 block(head_dim / 2);

    rotary_embedding_kernel_f16<<<grid, block>>>(
        (__half*)q,
        (__half*)k,
        (const __half*)cos,
        (const __half*)sin,
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

// Precompute RoPE frequencies (called once at model load)
__global__ void compute_rope_freqs_kernel(
    __half* __restrict__ cos_cache,
    __half* __restrict__ sin_cache,
    int max_seq_len,
    int head_dim,
    float base,
    float scaling_factor
) {
    const int pos = blockIdx.x;
    const int dim_idx = threadIdx.x;

    if (dim_idx >= head_dim / 2) return;

    // Compute frequency for this dimension
    const float freq = 1.0f / powf(base, (float)(2 * dim_idx) / head_dim);
    const float angle = (float)pos * freq / scaling_factor;

    const int idx = pos * (head_dim / 2) + dim_idx;
    cos_cache[idx] = __float2half(cosf(angle));
    sin_cache[idx] = __float2half(sinf(angle));
}

// Helper function to precompute RoPE cache (export via header if needed)
extern "C" BinferError binfer_compute_rope_cache(
    void* cos_cache,
    void* sin_cache,
    int max_seq_len,
    int head_dim,
    float base,
    float scaling_factor
) {
    dim3 grid(max_seq_len);
    dim3 block(head_dim / 2);

    compute_rope_freqs_kernel<<<grid, block>>>(
        (__half*)cos_cache,
        (__half*)sin_cache,
        max_seq_len,
        head_dim,
        base,
        scaling_factor
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}
