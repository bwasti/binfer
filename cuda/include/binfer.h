#ifndef BINFER_H
#define BINFER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
typedef enum {
    BINFER_SUCCESS = 0,
    BINFER_ERROR_CUDA = 1,
    BINFER_ERROR_INVALID_ARGUMENT = 2,
    BINFER_ERROR_OUT_OF_MEMORY = 3,
} BinferError;

// Device management
int binfer_get_device_count();
BinferError binfer_get_device_properties(int device, char* name, size_t name_len, size_t* total_memory);
BinferError binfer_set_device(int device);
BinferError binfer_mem_info(size_t* free_bytes, size_t* total_bytes);

// Initialize cuBLAS handles for all devices (call once at startup)
BinferError binfer_init_cublas(int num_devices);

// Memory management
BinferError binfer_malloc(void** ptr, size_t size);
BinferError binfer_free(void* ptr);
BinferError binfer_memcpy_h2d(void* dst, const void* src, size_t size);
BinferError binfer_memcpy_d2h(void* dst, const void* src, size_t size);
BinferError binfer_memcpy_d2d(void* dst, const void* src, size_t size);

// Pinned (page-locked) host memory for faster H2D transfers
BinferError binfer_malloc_host(void** ptr, size_t size);
BinferError binfer_free_host(void* ptr);
// Async memcpy using pinned memory (returns immediately, use synchronize to wait)
BinferError binfer_memcpy_h2d_async(void* dst, const void* src, size_t size);
// Copy from regular host memory to pinned memory
BinferError binfer_memcpy_host_to_pinned(void* pinned_dst, const void* src, size_t size);
// Synchronize all pending operations on current device
BinferError binfer_synchronize();

// GEMM operations (cuBLAS wrapper)
// C = alpha * A @ B + beta * C
// A: [M, K], B: [K, N], C: [M, N]
BinferError binfer_gemm_f16(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta
);

BinferError binfer_gemm_bf16(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta
);

// Batched GEMM for multi-head attention
BinferError binfer_gemm_batched_f16(
    const void** A, const void** B, void** C,
    int M, int N, int K,
    float alpha, float beta,
    int batch_size
);

// GEMM with B transposed: C = A @ B^T
BinferError binfer_gemm_f16_transb(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta
);

BinferError binfer_gemm_bf16_transb(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta
);

// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
BinferError binfer_rmsnorm_f16(
    const void* input,    // [batch, seq_len, hidden_size]
    const void* weight,   // [hidden_size]
    void* output,         // [batch, seq_len, hidden_size]
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps
);

BinferError binfer_rmsnorm_bf16(
    const void* input,
    const void* weight,
    void* output,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps
);

// RoPE: Apply rotary position embeddings
BinferError binfer_rotary_embedding_f16(
    void* q,              // [batch, seq_len, num_heads, head_dim]
    void* k,              // [batch, seq_len, num_kv_heads, head_dim]
    const void* cos,      // [seq_len, head_dim/2] or [1, seq_len, 1, head_dim]
    const void* sin,      // [seq_len, head_dim/2] or [1, seq_len, 1, head_dim]
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int position_offset   // For KV cache continuation
);

BinferError binfer_rotary_embedding_bf16(
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
);

// Activations
BinferError binfer_silu_f16(
    const void* input,
    void* output,
    size_t numel
);

BinferError binfer_silu_bf16(
    const void* input,
    void* output,
    size_t numel
);

BinferError binfer_gelu_f16(
    const void* input,
    void* output,
    size_t numel
);

// SwiGLU: silu(gate) * up
BinferError binfer_swiglu_f16(
    const void* gate,     // [batch, seq_len, intermediate_size]
    const void* up,       // [batch, seq_len, intermediate_size]
    void* output,         // [batch, seq_len, intermediate_size]
    size_t numel
);

BinferError binfer_swiglu_bf16(
    const void* gate,
    const void* up,
    void* output,
    size_t numel
);

// Element-wise operations
BinferError binfer_add_f16(
    const void* a,
    const void* b,
    void* output,
    size_t numel
);

BinferError binfer_add_bf16(
    const void* a,
    const void* b,
    void* output,
    size_t numel
);

BinferError binfer_mul_f16(
    const void* a,
    const void* b,
    void* output,
    size_t numel
);

// Softmax
BinferError binfer_softmax_f16(
    const void* input,    // [batch, seq_len, vocab_size]
    void* output,         // [batch, seq_len, vocab_size]
    int batch_size,
    int seq_len,
    int vocab_size,
    float temperature
);

BinferError binfer_softmax_bf16(
    const void* input,
    void* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    float temperature
);

// Top-k sampling (returns indices)
BinferError binfer_topk_f16(
    const void* logits,   // [batch, vocab_size]
    void* values,         // [batch, k]
    int32_t* indices,     // [batch, k]
    int batch_size,
    int vocab_size,
    int k
);

// Embedding lookup
BinferError binfer_embedding_f16(
    const void* weight,   // [vocab_size, hidden_size]
    const int32_t* input_ids, // [batch, seq_len]
    void* output,         // [batch, seq_len, hidden_size]
    int batch_size,
    int seq_len,
    int vocab_size,
    int hidden_size
);

BinferError binfer_embedding_bf16(
    const void* weight,
    const int32_t* input_ids,
    void* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    int hidden_size
);

// RoPE cache precomputation
BinferError binfer_compute_rope_cache(
    void* cos_cache,
    void* sin_cache,
    int max_seq_len,
    int head_dim,
    float base,
    float scaling_factor
);

BinferError binfer_compute_rope_cache_bf16(
    void* cos_cache,
    void* sin_cache,
    int max_seq_len,
    int head_dim,
    float base,
    float scaling_factor
);

// BF16 to FP16 conversion (for GPU-based dtype conversion)
BinferError binfer_convert_bf16_to_fp16(
    void* input,
    void* output,
    size_t num_elements
);

// BF16 Flash Attention
BinferError binfer_flash_attention_bf16(
    const void* Q,
    const void* K,
    const void* V,
    void* O,
    int batch_size,
    int seq_q,
    int seq_kv,
    int kv_stride,
    int q_offset,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float softmax_scale,
    bool is_causal
);

BinferError binfer_flash_attention_with_cache_bf16(
    const void* Q,
    const void* K_cache,
    const void* V_cache,
    void* O,
    int batch_size,
    int cache_seq_len,
    int max_seq_len,
    int q_offset,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float softmax_scale,
    bool is_causal
);

BinferError binfer_kv_cache_update_bf16(
    void* K_cache,
    void* V_cache,
    const void* K_new,
    const void* V_new,
    int batch_size,
    int cache_offset,
    int seq_new,
    int num_kv_heads,
    int head_dim,
    int max_seq_len
);

// Synchronization
BinferError binfer_synchronize();

// Profiling - CUDA events for timing
BinferError binfer_event_create(void** event);
BinferError binfer_event_destroy(void* event);
BinferError binfer_event_record(void* event);  // Records on default stream
BinferError binfer_event_synchronize(void* event);
BinferError binfer_event_elapsed_time(float* ms, void* start, void* end);

// ============================================================================
// Mixture of Experts (MoE) kernels
// ============================================================================

// Initialize MXFP4 dequantization lookup tables (call once at startup)
BinferError binfer_init_mxfp4_tables();

// MXFP4 dequantization: Convert quantized weights to BF16
// blocks: [num_experts, out_features, num_blocks, 16] as uint8
// scales: [num_experts, out_features, num_blocks] as uint8 (E5M2)
// output: [num_experts, out_features, in_features] as bf16
BinferError binfer_mxfp4_dequant(
    const void* blocks,
    const void* scales,
    const void* bias,
    void* output,
    int num_experts,
    int out_features,
    int num_blocks,
    int in_features
);

// MXFP4 single expert dequantization: Convert one expert's weights to BF16
// More efficient for on-demand dequant of selected experts only
// blocks: [num_experts, out_features, num_blocks, 16] as uint8
// scales: [num_experts, out_features, num_blocks] as uint8 (E8M0)
// output: [out_features, in_features] as bf16 (single expert)
BinferError binfer_mxfp4_dequant_single_expert(
    const void* blocks,
    const void* scales,
    void* output,
    int expert_idx,
    int num_experts,
    int out_features,
    int num_blocks,
    int in_features
);

// Expert routing: compute router logits and select top-k experts
// hidden: [batch_seq, hidden_size] as bf16
// router_weight: [num_experts, hidden_size] as bf16
// router_bias: [num_experts] as bf16
// expert_indices: [batch_seq, top_k] as int32 (output)
// expert_weights: [batch_seq, top_k] as bf16 (output, softmax normalized)
BinferError binfer_moe_router_topk(
    const void* hidden,
    const void* router_weight,
    const void* router_bias,
    int32_t* expert_indices,
    void* expert_weights,
    int batch_size,
    int seq_len,
    int hidden_size,
    int num_experts,
    int top_k
);

// MoE SwiGLU activation for fused gate/up projection
// gate_up: [batch, intermediate * 2] -> output: [batch, intermediate]
BinferError binfer_moe_swiglu(
    const void* gate_up,
    void* output,
    int batch,
    int intermediate_size
);

// GPT-OSS custom activation (interleaved gate/up with modified gating)
// gate_up: [batch, intermediate * 2] interleaved -> output: [batch, intermediate]
// Activation: (up + 1) * gate * sigmoid(gate * alpha), with clamping
BinferError binfer_gpt_oss_activation(
    const void* gate_up,
    void* output,
    int batch,
    int intermediate_size,
    float alpha,
    float limit
);

// Full MoE forward pass with MXFP4 weights
BinferError binfer_moe_forward(
    const void* input,
    const void* gate_up_blocks,
    const void* gate_up_scales,
    const void* gate_up_bias,
    const void* down_blocks,
    const void* down_scales,
    const void* down_bias,
    const void* router_weight,
    const void* router_bias,
    void* output,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size,
    int num_experts,
    int top_k
);

// Scale and add: output = output + scale * input
BinferError binfer_scale_add_bf16(
    const void* input,
    void* output,
    float scale,
    int numel
);

// Fused MoE forward: dequant + GEMM on-the-fly for single token
// This avoids the need to sync and read expert indices to CPU
BinferError binfer_moe_fused_forward(
    const void* hidden,           // [hidden_size] bf16
    const void* gate_up_blocks,   // [num_experts, intermediate*2, num_blocks_in, 16] uint8
    const void* gate_up_scales,   // [num_experts, intermediate*2, num_blocks_in] uint8
    const void* gate_up_bias,     // [num_experts, intermediate*2] bf16 or nullptr
    const void* down_blocks,      // [num_experts, hidden, num_blocks_inter, 16] uint8
    const void* down_scales,      // [num_experts, hidden, num_blocks_inter] uint8
    const void* down_bias,        // [num_experts, hidden] bf16 or nullptr
    const void* expert_indices,   // [top_k] int32 - on GPU
    const void* expert_weights,   // [top_k] bf16 - on GPU
    void* output,                 // [hidden_size] bf16
    int hidden_size,
    int intermediate_size,
    int num_experts,
    int top_k
);

// Debug: just run gate_up kernel
BinferError binfer_moe_gate_up_debug(
    const void* hidden,
    const void* gate_up_blocks,
    const void* gate_up_scales,
    const void* expert_indices,
    void* gate_up_out,
    int hidden_size,
    int intermediate_size,
    int num_blocks,
    int num_experts,
    int top_k
);

// Memory set (zero memory)
BinferError binfer_memset(void* ptr, int value, size_t size);

// =============================================================================
// MoE with cuBLAS - expert-grouped batching for maximum GPU utilization
// =============================================================================

// Initialize lookup tables for cuBLAS MoE
BinferError binfer_init_moe_cublas_tables();

// Get required scratch buffer size for cuBLAS MoE
size_t binfer_moe_cublas_scratch_size(
    int numTokens,
    int hiddenSize,
    int intermediateSize,
    int numExperts,
    int topK
);

// MoE forward pass using cuBLAS GEMMs for maximum GPU utilization
// Strategy: for each expert, gather tokens, dequant weights, cuBLAS GEMM, scatter results
BinferError binfer_moe_cublas_forward(
    const void* hidden,              // [numTokens, hiddenSize]
    const void* gate_up_blocks,      // [numExperts, intermediate*2, numBlocksIn, 16]
    const void* gate_up_scales,      // [numExperts, intermediate*2, numBlocksIn]
    const void* gate_up_bias,        // [numExperts, intermediate*2] or nullptr
    const void* down_blocks,         // [numExperts, hiddenSize, numBlocksInter, 16]
    const void* down_scales,         // [numExperts, hiddenSize, numBlocksInter]
    const void* down_bias,           // [numExperts, hiddenSize] or nullptr
    const void* expert_indices,      // [numTokens, topK] int32
    const void* expert_weights,      // [numTokens, topK] bf16
    void* output,                    // [numTokens, hiddenSize]
    void* scratch,                   // Scratch buffer for intermediates
    size_t scratch_size,
    int numTokens,
    int hiddenSize,
    int intermediateSize,
    int numExperts,
    int topK
);

// =============================================================================
// MoE with cuBLAS Grouped GEMM - optimal for parallel expert execution
// =============================================================================

// Initialize lookup tables for grouped GEMM MoE
BinferError binfer_init_moe_grouped_tables();

// Get required scratch buffer size for grouped GEMM MoE
size_t binfer_moe_grouped_scratch_size(
    int numTokens,
    int hiddenSize,
    int intermediateSize,
    int numExperts,
    int topK
);

// MoE forward pass using cuBLAS Grouped GEMM
BinferError binfer_moe_grouped_forward(
    const void* hidden,              // [numTokens, hiddenSize]
    const void* gate_up_blocks,      // [numExperts, intermediate*2, numBlocksIn, 16]
    const void* gate_up_scales,      // [numExperts, intermediate*2, numBlocksIn]
    const void* gate_up_bias,        // [numExperts, intermediate*2] or nullptr
    const void* down_blocks,         // [numExperts, hiddenSize, numBlocksInter, 16]
    const void* down_scales,         // [numExperts, hiddenSize, numBlocksInter]
    const void* down_bias,           // [numExperts, hiddenSize] or nullptr
    const void* expert_indices,      // [numTokens, topK] int32
    const void* expert_weights,      // [numTokens, topK] bf16
    void* output,                    // [numTokens, hiddenSize]
    void* scratch,                   // Scratch buffer for intermediates
    size_t scratch_size,
    int numTokens,
    int hiddenSize,
    int intermediateSize,
    int numExperts,
    int topK
);

// =============================================================================
// MoE with CUTLASS Grouped GEMM - optimal for H100 with BF16 support
// =============================================================================

// Initialize lookup tables for CUTLASS MoE
BinferError binfer_init_moe_cutlass_tables();

// Get required scratch buffer size for CUTLASS MoE
size_t binfer_moe_cutlass_scratch_size(
    int numTokens,
    int hiddenSize,
    int intermediateSize,
    int numExperts,
    int topK
);

// MoE forward pass using CUTLASS Grouped GEMM
BinferError binfer_moe_cutlass_forward(
    const void* hidden,
    const void* gate_up_blocks,
    const void* gate_up_scales,
    const void* gate_up_bias,
    const void* down_blocks,
    const void* down_scales,
    const void* down_bias,
    const void* expert_indices,
    const void* expert_weights,
    void* output,
    void* scratch,
    size_t scratch_size,
    int numTokens,
    int hiddenSize,
    int intermediateSize,
    int numExperts,
    int topK
);

// =============================================================================
// GPU Direct Storage (GDS) / cuFile support
// =============================================================================

// Check if GDS is available on this system
// Returns 1 if available, 0 if not
int binfer_gds_available();

// Initialize GDS driver (call once at startup)
// Returns BINFER_SUCCESS if GDS is available and initialized
BinferError binfer_gds_init();

// Close GDS driver
BinferError binfer_gds_close();

// Register GPU buffer for GDS DMA
// buffer must be GPU memory allocated with binfer_malloc
BinferError binfer_gds_register_buffer(void* buffer, size_t size);

// Deregister GPU buffer
BinferError binfer_gds_deregister_buffer(void* buffer);

// Open a file for GDS (returns opaque handle)
// File is opened with O_RDONLY | O_DIRECT
// Returns handle in *handle, or 0 on failure
BinferError binfer_gds_open(const char* path, void** handle);

// Close GDS file handle
BinferError binfer_gds_close_file(void* handle);

// Read from file directly to GPU memory
// Returns number of bytes read, or negative error code
ssize_t binfer_gds_read(
    void* handle,           // File handle from binfer_gds_open
    void* gpu_buffer,       // GPU memory (must be registered)
    size_t size,            // Number of bytes to read
    size_t file_offset,     // Offset in file
    size_t buffer_offset    // Offset in GPU buffer
);

// =============================================================================
// CUDA Graph capture and replay
// =============================================================================

// Set the current stream for all CUDA operations (for graph capture)
// Pass NULL/0 to reset to the default stream
BinferError binfer_set_current_stream(void* stream);

// Get the current stream
void* binfer_get_current_stream();

// Begin capturing operations on a stream into a graph
// mode: 0 = global (all streams), 1 = thread-local, 2 = relaxed
BinferError binfer_stream_begin_capture(void* stream, int mode);

// End capture and return the graph
BinferError binfer_stream_end_capture(void* stream, void** graph);

// Instantiate a graph into an executable
BinferError binfer_graph_instantiate(void** graph_exec, void* graph);

// Destroy graph and graph exec
BinferError binfer_graph_destroy(void* graph);
BinferError binfer_graph_exec_destroy(void* graph_exec);

// Launch an instantiated graph on a stream
BinferError binfer_graph_launch(void* graph_exec, void* stream);

// Update graph exec with new graph (for parameter changes)
// Returns BINFER_SUCCESS if update succeeded, error if graph structure changed
BinferError binfer_graph_exec_update(void* graph_exec, void* graph);

#ifdef __cplusplus
}
#endif

#endif // BINFER_H
