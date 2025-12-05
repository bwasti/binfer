// FlashAttention-3 C API Shim for Binfer
// Provides a simplified C interface to FA3's Hopper kernels

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

// Include CUTLASS types (needed for bfloat16_t and half_t)
#include <cutlass/numeric_types.h>

// Include FA3 headers
#include "flash.h"
#include "static_switch.h"

// Forward declarations for the instantiated kernels we need
template <int Arch, typename T, int kHeadDim, int kHeadDimV, bool Split, bool PagedKVNonTMA, bool Has_softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);

extern "C" {

// Error codes
#define FA3_SUCCESS 0
#define FA3_ERROR_INVALID_ARGUMENT 1
#define FA3_ERROR_CUDA 2
#define FA3_ERROR_UNSUPPORTED 3

/**
 * FlashAttention-3 forward pass for BF16.
 *
 * @param Q Query tensor [batch, seq_q, num_heads, head_dim]
 * @param K Key tensor [batch, seq_kv, num_kv_heads, head_dim]
 * @param V Value tensor [batch, seq_kv, num_kv_heads, head_dim]
 * @param O Output tensor [batch, seq_q, num_heads, head_dim]
 * @param softmax_lse Log-sum-exp output [batch, num_heads, seq_q] (can be nullptr)
 * @param batch_size Batch size
 * @param seq_q Query sequence length
 * @param seq_kv Key/value sequence length
 * @param num_heads Number of query heads
 * @param num_kv_heads Number of KV heads (for GQA)
 * @param head_dim Head dimension (must be 64, 96, 128, 192, or 256)
 * @param softmax_scale Softmax scale factor (typically 1/sqrt(head_dim))
 * @param is_causal Whether to use causal masking
 * @param stream CUDA stream
 */
int fa3_fwd_bf16(
    const void* Q,
    const void* K,
    const void* V,
    void* O,
    void* softmax_lse,
    int batch_size,
    int seq_q,
    int seq_kv,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream
) {
    // Validate head_dim
    if (head_dim != 64 && head_dim != 96 && head_dim != 128 && head_dim != 192 && head_dim != 256) {
        return FA3_ERROR_INVALID_ARGUMENT;
    }

    // Initialize params
    Flash_fwd_params params = {};

    // Set data type
    params.is_bf16 = true;
    params.is_fp32 = false;
    params.is_e4m3 = false;

    // Set pointers
    params.q_ptr = const_cast<void*>(Q);
    params.k_ptr = const_cast<void*>(K);
    params.v_ptr = const_cast<void*>(V);
    params.o_ptr = O;
    params.softmax_lse_ptr = softmax_lse;

    // Set dimensions
    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_kv_heads;
    params.seqlen_q = seq_q;
    params.seqlen_k = seq_kv;
    params.d = head_dim;
    params.dv = head_dim;
    params.d_rounded = ((head_dim + 31) / 32) * 32;
    params.seqlen_q_rounded = ((seq_q + 127) / 128) * 128;
    params.seqlen_k_rounded = ((seq_kv + 127) / 128) * 128;

    // Set strides (assumes contiguous layout: [batch, seq, heads, head_dim])
    params.q_batch_stride = seq_q * num_heads * head_dim;
    params.k_batch_stride = seq_kv * num_kv_heads * head_dim;
    params.v_batch_stride = seq_kv * num_kv_heads * head_dim;
    params.o_batch_stride = seq_q * num_heads * head_dim;

    params.q_row_stride = num_heads * head_dim;
    params.k_row_stride = num_kv_heads * head_dim;
    params.v_row_stride = num_kv_heads * head_dim;
    params.o_row_stride = num_heads * head_dim;

    params.q_head_stride = head_dim;
    params.k_head_stride = head_dim;
    params.v_head_stride = head_dim;
    params.o_head_stride = head_dim;

    params.v_dim_stride = 1;

    // Set softmax scale
    params.scale_softmax = softmax_scale;
    params.softcap = 0.0f;

    // Set causal
    params.is_causal = is_causal;
    params.is_local = false;
    params.window_size_left = -1;
    params.window_size_right = is_causal ? 0 : -1;

    // No dropout
    params.p_dropout = 1.0f;
    params.rp_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;

    // No paging
    params.page_table = nullptr;
    params.pagedkv_tma = false;

    // No variable length
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.seqused_q = nullptr;
    params.seqused_k = nullptr;

    // Single split (no split-KV)
    params.num_splits = 1;
    params.pack_gqa = false;

    // Architecture
    params.arch = 90;  // Hopper

    // Get SM count
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    params.num_sm = props.multiProcessorCount;

    // Allocate and zero the tile_count_semaphore (needed for Hopper persistent scheduler)
    int* tile_count_semaphore = nullptr;
    cudaMalloc(&tile_count_semaphore, sizeof(int));
    cudaMemset(tile_count_semaphore, 0, sizeof(int));
    params.tile_count_semaphore = tile_count_semaphore;
    params.tile_count_semaphore_offset = 0;

    // Other scheduler params
    params.num_m_blocks_ptr = nullptr;
    params.num_splits_dynamic_ptr = nullptr;
    params.varlen_batch_idx_ptr = nullptr;
    params.num_nheads_in_l2_ptr = nullptr;
    params.skip_scheduler_metadata_computation = false;
    params.varlen_sort_batches = false;
    params.head_swizzle = false;
    params.prepare_varlen_pdl = false;

    // Dispatch based on head_dim
    try {
        switch (head_dim) {
            case 64:
                run_mha_fwd_<90, cutlass::bfloat16_t, 64, 64, false, false, false, false>(params, stream);
                break;
            case 96:
                run_mha_fwd_<90, cutlass::bfloat16_t, 96, 96, false, false, false, false>(params, stream);
                break;
            case 128:
                run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, false, false, false, false>(params, stream);
                break;
            case 192:
                run_mha_fwd_<90, cutlass::bfloat16_t, 192, 192, false, false, false, false>(params, stream);
                break;
            case 256:
                run_mha_fwd_<90, cutlass::bfloat16_t, 256, 256, false, false, false, false>(params, stream);
                break;
            default:
                cudaFree(tile_count_semaphore);
                return FA3_ERROR_UNSUPPORTED;
        }
    } catch (...) {
        cudaFree(tile_count_semaphore);
        return FA3_ERROR_CUDA;
    }

    // Sync and cleanup
    cudaError_t syncErr = cudaStreamSynchronize(stream);
    cudaFree(tile_count_semaphore);

    if (syncErr != cudaSuccess) {
        fprintf(stderr, "FA3 BF16 sync error: %s\n", cudaGetErrorString(syncErr));
        return FA3_ERROR_CUDA;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "FA3 BF16 error: %s\n", cudaGetErrorString(err));
    }
    return err == cudaSuccess ? FA3_SUCCESS : FA3_ERROR_CUDA;
}

/**
 * FlashAttention-3 forward pass for FP16.
 */
int fa3_fwd_fp16(
    const void* Q,
    const void* K,
    const void* V,
    void* O,
    void* softmax_lse,
    int batch_size,
    int seq_q,
    int seq_kv,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream
) {
    // Validate head_dim
    if (head_dim != 64 && head_dim != 96 && head_dim != 128 && head_dim != 192 && head_dim != 256) {
        return FA3_ERROR_INVALID_ARGUMENT;
    }

    // Initialize params
    Flash_fwd_params params = {};

    // Set data type
    params.is_bf16 = false;
    params.is_fp32 = false;
    params.is_e4m3 = false;

    // Set pointers
    params.q_ptr = const_cast<void*>(Q);
    params.k_ptr = const_cast<void*>(K);
    params.v_ptr = const_cast<void*>(V);
    params.o_ptr = O;
    params.softmax_lse_ptr = softmax_lse;

    // Set dimensions
    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_kv_heads;
    params.seqlen_q = seq_q;
    params.seqlen_k = seq_kv;
    params.d = head_dim;
    params.dv = head_dim;
    params.d_rounded = ((head_dim + 31) / 32) * 32;
    params.seqlen_q_rounded = ((seq_q + 127) / 128) * 128;
    params.seqlen_k_rounded = ((seq_kv + 127) / 128) * 128;

    // Set strides
    params.q_batch_stride = seq_q * num_heads * head_dim;
    params.k_batch_stride = seq_kv * num_kv_heads * head_dim;
    params.v_batch_stride = seq_kv * num_kv_heads * head_dim;
    params.o_batch_stride = seq_q * num_heads * head_dim;

    params.q_row_stride = num_heads * head_dim;
    params.k_row_stride = num_kv_heads * head_dim;
    params.v_row_stride = num_kv_heads * head_dim;
    params.o_row_stride = num_heads * head_dim;

    params.q_head_stride = head_dim;
    params.k_head_stride = head_dim;
    params.v_head_stride = head_dim;
    params.o_head_stride = head_dim;

    params.v_dim_stride = 1;

    // Set softmax scale
    params.scale_softmax = softmax_scale;
    params.softcap = 0.0f;

    // Set causal
    params.is_causal = is_causal;
    params.is_local = false;
    params.window_size_left = -1;
    params.window_size_right = is_causal ? 0 : -1;

    // No dropout
    params.p_dropout = 1.0f;
    params.rp_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;

    // No paging
    params.page_table = nullptr;
    params.pagedkv_tma = false;

    // No variable length
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.seqused_q = nullptr;
    params.seqused_k = nullptr;

    // Single split
    params.num_splits = 1;
    params.pack_gqa = false;

    // Architecture
    params.arch = 90;

    // Get SM count
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    params.num_sm = props.multiProcessorCount;

    // Allocate and zero the tile_count_semaphore (needed for Hopper persistent scheduler)
    int* tile_count_semaphore = nullptr;
    cudaMalloc(&tile_count_semaphore, sizeof(int));
    cudaMemset(tile_count_semaphore, 0, sizeof(int));
    params.tile_count_semaphore = tile_count_semaphore;
    params.tile_count_semaphore_offset = 0;

    // Other scheduler params
    params.num_m_blocks_ptr = nullptr;
    params.num_splits_dynamic_ptr = nullptr;
    params.varlen_batch_idx_ptr = nullptr;
    params.num_nheads_in_l2_ptr = nullptr;
    params.skip_scheduler_metadata_computation = false;
    params.varlen_sort_batches = false;
    params.head_swizzle = false;
    params.prepare_varlen_pdl = false;

    // Dispatch based on head_dim
    try {
        switch (head_dim) {
            case 64:
                run_mha_fwd_<90, cutlass::half_t, 64, 64, false, false, false, false>(params, stream);
                break;
            case 96:
                run_mha_fwd_<90, cutlass::half_t, 96, 96, false, false, false, false>(params, stream);
                break;
            case 128:
                run_mha_fwd_<90, cutlass::half_t, 128, 128, false, false, false, false>(params, stream);
                break;
            case 192:
                run_mha_fwd_<90, cutlass::half_t, 192, 192, false, false, false, false>(params, stream);
                break;
            case 256:
                run_mha_fwd_<90, cutlass::half_t, 256, 256, false, false, false, false>(params, stream);
                break;
            default:
                cudaFree(tile_count_semaphore);
                return FA3_ERROR_UNSUPPORTED;
        }
    } catch (...) {
        cudaFree(tile_count_semaphore);
        return FA3_ERROR_CUDA;
    }

    // Sync and cleanup
    cudaStreamSynchronize(stream);
    cudaFree(tile_count_semaphore);

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? FA3_SUCCESS : FA3_ERROR_CUDA;
}

/**
 * FlashAttention-3 forward pass for BF16 with variable-length sequences and paged KV cache.
 * This is used for continuous batching with vLLM-style paged attention.
 *
 * @param Q Query tensor [total_q_tokens, num_heads, head_dim] - concatenated queries
 * @param K_cache Paged K cache [num_blocks, block_size, num_kv_heads, head_dim]
 * @param V_cache Paged V cache [num_blocks, block_size, num_kv_heads, head_dim]
 * @param O Output tensor [total_q_tokens, num_heads, head_dim]
 * @param softmax_lse Log-sum-exp output [num_heads, total_q_tokens] (can be nullptr)
 * @param cu_seqlens_q Cumulative query lengths [batch_size + 1] (device pointer)
 * @param seqused_k Actual K sequence lengths [batch_size] (device pointer)
 * @param block_table Block table mapping [batch_size, max_blocks_per_seq] (device pointer)
 * @param batch_size Number of sequences in batch
 * @param max_seqlen_q Maximum query sequence length
 * @param max_seqlen_k Maximum KV sequence length
 * @param num_heads Number of query heads
 * @param num_kv_heads Number of KV heads (for GQA)
 * @param head_dim Head dimension (must be 64, 96, 128, 192, or 256)
 * @param block_size Tokens per block (typically 16)
 * @param max_blocks_per_seq Maximum blocks per sequence
 * @param softmax_scale Softmax scale factor
 * @param is_causal Whether to use causal masking
 * @param stream CUDA stream
 */
int fa3_fwd_varlen_paged_bf16(
    const void* Q,
    const void* K_cache,
    const void* V_cache,
    void* O,
    void* softmax_lse,
    const int* cu_seqlens_q,
    const int* seqused_k,
    const int* block_table,
    int batch_size,
    int max_seqlen_q,
    int max_seqlen_k,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream
) {
    // Validate head_dim
    if (head_dim != 64 && head_dim != 96 && head_dim != 128 && head_dim != 192 && head_dim != 256) {
        return FA3_ERROR_INVALID_ARGUMENT;
    }

    // Initialize params
    Flash_fwd_params params = {};

    // Set data type
    params.is_bf16 = true;
    params.is_fp32 = false;
    params.is_e4m3 = false;

    // Set pointers
    params.q_ptr = const_cast<void*>(Q);
    params.k_ptr = const_cast<void*>(K_cache);
    params.v_ptr = const_cast<void*>(V_cache);
    params.o_ptr = O;
    params.softmax_lse_ptr = softmax_lse;

    // Set dimensions
    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_kv_heads;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.d = head_dim;
    params.dv = head_dim;
    params.d_rounded = ((head_dim + 31) / 32) * 32;
    params.seqlen_q_rounded = ((max_seqlen_q + 127) / 128) * 128;
    params.seqlen_k_rounded = ((max_seqlen_k + 127) / 128) * 128;

    // Q strides: [total_q_tokens, num_heads, head_dim]
    // For varlen, batch_stride is not used (we use cu_seqlens instead)
    params.q_batch_stride = 0;  // Not used for varlen
    params.q_row_stride = num_heads * head_dim;
    params.q_head_stride = head_dim;

    // K/V strides for PAGED layout: [num_blocks, block_size, num_kv_heads, head_dim]
    // The page_table handles block lookup, so batch_stride is page table stride
    params.k_batch_stride = block_size * num_kv_heads * head_dim;  // stride between blocks
    params.v_batch_stride = block_size * num_kv_heads * head_dim;
    params.k_row_stride = num_kv_heads * head_dim;  // stride within block (between tokens)
    params.v_row_stride = num_kv_heads * head_dim;
    params.k_head_stride = head_dim;
    params.v_head_stride = head_dim;

    // Output strides: [total_q_tokens, num_heads, head_dim]
    params.o_batch_stride = 0;  // Not used for varlen
    params.o_row_stride = num_heads * head_dim;
    params.o_head_stride = head_dim;

    params.v_dim_stride = 1;

    // Set softmax scale
    params.scale_softmax = softmax_scale;
    params.softcap = 0.0f;

    // Set causal
    params.is_causal = is_causal;
    params.is_local = false;
    params.window_size_left = -1;
    params.window_size_right = is_causal ? 0 : -1;

    // No dropout
    params.p_dropout = 1.0f;
    params.rp_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;

    // PAGED KV CACHE - the key difference!
    params.page_table = const_cast<int*>(block_table);
    params.page_table_batch_stride = max_blocks_per_seq;
    params.page_size = block_size;
    params.pagedkv_tma = false;  // Non-TMA paged attention

    // Variable length sequences
    params.cu_seqlens_q = const_cast<int*>(cu_seqlens_q);
    params.cu_seqlens_k = nullptr;  // Not used with paged KV
    params.seqused_q = nullptr;
    params.seqused_k = const_cast<int*>(seqused_k);

    // Single split (no split-KV)
    params.num_splits = 1;
    params.pack_gqa = false;

    // Architecture
    params.arch = 90;  // Hopper

    // Get SM count
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    params.num_sm = props.multiProcessorCount;

    // Allocate and zero the tile_count_semaphore
    int* tile_count_semaphore = nullptr;
    cudaMalloc(&tile_count_semaphore, sizeof(int));
    cudaMemset(tile_count_semaphore, 0, sizeof(int));
    params.tile_count_semaphore = tile_count_semaphore;
    params.tile_count_semaphore_offset = 0;

    // Other scheduler params
    params.num_m_blocks_ptr = nullptr;
    params.num_splits_dynamic_ptr = nullptr;
    params.varlen_batch_idx_ptr = nullptr;
    params.num_nheads_in_l2_ptr = nullptr;
    params.skip_scheduler_metadata_computation = false;
    params.varlen_sort_batches = false;
    params.head_swizzle = false;
    params.prepare_varlen_pdl = false;

    // Dispatch based on head_dim - PagedKVNonTMA = true!
    try {
        switch (head_dim) {
            case 64:
                run_mha_fwd_<90, cutlass::bfloat16_t, 64, 64, false, true, false, false>(params, stream);
                break;
            case 96:
                run_mha_fwd_<90, cutlass::bfloat16_t, 96, 96, false, true, false, false>(params, stream);
                break;
            case 128:
                run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, false, true, false, false>(params, stream);
                break;
            case 192:
                run_mha_fwd_<90, cutlass::bfloat16_t, 192, 192, false, true, false, false>(params, stream);
                break;
            case 256:
                run_mha_fwd_<90, cutlass::bfloat16_t, 256, 256, false, true, false, false>(params, stream);
                break;
            default:
                cudaFree(tile_count_semaphore);
                return FA3_ERROR_UNSUPPORTED;
        }
    } catch (...) {
        cudaFree(tile_count_semaphore);
        return FA3_ERROR_CUDA;
    }

    // Sync and cleanup
    cudaError_t syncErr = cudaStreamSynchronize(stream);
    cudaFree(tile_count_semaphore);

    if (syncErr != cudaSuccess) {
        fprintf(stderr, "FA3 varlen paged BF16 sync error: %s\n", cudaGetErrorString(syncErr));
        return FA3_ERROR_CUDA;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "FA3 varlen paged BF16 error: %s\n", cudaGetErrorString(err));
    }
    return err == cudaSuccess ? FA3_SUCCESS : FA3_ERROR_CUDA;
}

} // extern "C"
