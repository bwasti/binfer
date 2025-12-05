#ifndef BINFER_ATTENTION_H
#define BINFER_ATTENTION_H

#include "binfer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Flash Attention forward pass
// Computes: softmax(Q @ K^T / sqrt(head_dim)) @ V
//
// Q: [batch, seq_q, num_heads, head_dim]
// K: [batch, kv_stride, num_kv_heads, head_dim]
// V: [batch, kv_stride, num_kv_heads, head_dim]
// O: [batch, seq_q, num_heads, head_dim]
//
// Supports GQA (grouped query attention) where num_heads > num_kv_heads
BinferError binfer_flash_attention_f16(
    const void* Q,
    const void* K,
    const void* V,
    void* O,
    int batch_size,
    int seq_q,
    int seq_kv,               // Valid KV length
    int kv_stride,            // Stride in K,V (max_seq_len for cached)
    int q_offset,             // Offset for absolute Q position (for causal mask in decode)
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float softmax_scale,      // Usually 1/sqrt(head_dim)
    bool is_causal            // Apply causal mask
);

// Flash Attention with KV cache
// For decoding: Q has seq_len=1, K/V are the full cache
BinferError binfer_flash_attention_with_cache_f16(
    const void* Q,            // [batch, 1, num_heads, head_dim]
    const void* K_cache,      // [batch, max_seq, num_kv_heads, head_dim]
    const void* V_cache,      // [batch, max_seq, num_kv_heads, head_dim]
    void* O,                  // [batch, 1, num_heads, head_dim]
    int batch_size,
    int cache_seq_len,        // Current valid length of KV cache
    int max_seq_len,          // Maximum cache capacity (stride)
    int q_offset,             // Absolute position of query (for causal mask)
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float softmax_scale,
    bool is_causal
);

// Paged attention for variable-length sequences
// Used with block-based KV cache (vLLM-style)
BinferError binfer_paged_attention_f16(
    const void* Q,            // [num_tokens, num_heads, head_dim]
    const void* K_cache,      // [num_blocks, block_size, num_kv_heads, head_dim]
    const void* V_cache,      // [num_blocks, block_size, num_kv_heads, head_dim]
    void* O,                  // [num_tokens, num_heads, head_dim]
    const int* block_tables,  // [batch, max_blocks_per_seq]
    const int* context_lens,  // [batch] - actual sequence length for each
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_context_len,
    float softmax_scale
);

// Update KV cache with new K, V values
BinferError binfer_kv_cache_update_f16(
    void* K_cache,            // [batch, max_seq, num_kv_heads, head_dim]
    void* V_cache,            // [batch, max_seq, num_kv_heads, head_dim]
    const void* K_new,        // [batch, seq_new, num_kv_heads, head_dim]
    const void* V_new,        // [batch, seq_new, num_kv_heads, head_dim]
    int batch_size,
    int cache_offset,         // Position to start writing
    int seq_new,              // Number of new tokens
    int num_kv_heads,
    int head_dim,
    int max_seq_len           // Maximum cache capacity
);

// Update paged KV cache with new K, V values
BinferError binfer_paged_kv_cache_update_f16(
    void* K_cache,            // [num_blocks, block_size, num_kv_heads, head_dim]
    void* V_cache,            // [num_blocks, block_size, num_kv_heads, head_dim]
    const void* K_new,        // [num_seqs, num_new_tokens, num_kv_heads, head_dim]
    const void* V_new,        // [num_seqs, num_new_tokens, num_kv_heads, head_dim]
    const int* block_tables,  // [num_seqs, max_blocks_per_seq]
    const int* context_lens,  // [num_seqs] - length BEFORE adding new tokens
    int num_seqs,
    int num_new_tokens,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq
);

#ifdef __cplusplus
}
#endif

#endif // BINFER_ATTENTION_H
