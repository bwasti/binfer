// CUDA FFI Bindings for Bun
import { dlopen, FFIType, ptr, suffix, CString } from "bun:ffi";
import { existsSync } from "fs";

// Error codes matching binfer.h
export enum BinferError {
  SUCCESS = 0,
  ERROR_CUDA = 1,
  ERROR_INVALID_ARGUMENT = 2,
  ERROR_OUT_OF_MEMORY = 3,
}

// Find the CUDA library
function findCudaLibrary(): string {
  const possiblePaths = [
    `./cuda/build/libbinfer_cuda.${suffix}`,
    `../cuda/build/libbinfer_cuda.${suffix}`,
    `/usr/local/lib/libbinfer_cuda.${suffix}`,
  ];

  for (const path of possiblePaths) {
    if (existsSync(path)) {
      return path;
    }
  }

  throw new Error(
    `Could not find libbinfer_cuda.${suffix}. Run 'bun run build:cuda' first.`
  );
}

// FFI symbols definition
const cudaSymbols = {
  // Device management
  binfer_get_device_count: {
    args: [] as const,
    returns: FFIType.i32,
  },
  binfer_get_device_properties: {
    args: [FFIType.i32, FFIType.ptr, FFIType.u64, FFIType.ptr] as const,
    returns: FFIType.i32,
  },
  binfer_set_device: {
    args: [FFIType.i32] as const,
    returns: FFIType.i32,
  },
  binfer_mem_info: {
    args: [FFIType.ptr, FFIType.ptr] as const,  // free_bytes, total_bytes
    returns: FFIType.i32,
  },
  binfer_init_cublas: {
    args: [FFIType.i32] as const,  // num_devices
    returns: FFIType.i32,
  },

  // Memory management
  binfer_malloc: {
    args: [FFIType.ptr, FFIType.u64] as const,
    returns: FFIType.i32,
  },
  binfer_free: {
    args: [FFIType.u64] as const,  // device ptr
    returns: FFIType.i32,
  },
  binfer_memcpy_h2d: {
    args: [FFIType.u64, FFIType.ptr, FFIType.u64] as const,  // dst is device ptr (u64), src is host ptr
    returns: FFIType.i32,
  },
  binfer_memcpy_d2h: {
    args: [FFIType.ptr, FFIType.u64, FFIType.u64] as const,  // dst is host ptr, src is device ptr (u64)
    returns: FFIType.i32,
  },
  binfer_memcpy_d2d: {
    args: [FFIType.u64, FFIType.u64, FFIType.u64] as const,  // both device ptrs
    returns: FFIType.i32,
  },
  // Pinned host memory
  binfer_malloc_host: {
    args: [FFIType.ptr, FFIType.u64] as const,
    returns: FFIType.i32,
  },
  binfer_free_host: {
    args: [FFIType.u64] as const,  // host ptr passed as u64
    returns: FFIType.i32,
  },
  binfer_memcpy_h2d_async: {
    args: [FFIType.u64, FFIType.u64, FFIType.u64] as const,  // dst (device), src (host pinned), size
    returns: FFIType.i32,
  },
  binfer_memcpy_host_to_pinned: {
    args: [FFIType.u64, FFIType.ptr, FFIType.u64] as const,  // pinned_dst, src (regular host), size
    returns: FFIType.i32,
  },
  binfer_synchronize: {
    args: [] as const,
    returns: FFIType.i32,
  },

  // CUDA Streams
  binfer_stream_create: {
    args: [FFIType.ptr] as const,  // cudaStream_t*
    returns: FFIType.i32,
  },
  binfer_stream_destroy: {
    args: [FFIType.u64] as const,  // cudaStream_t (opaque pointer as u64)
    returns: FFIType.i32,
  },
  binfer_stream_synchronize: {
    args: [FFIType.u64] as const,  // cudaStream_t
    returns: FFIType.i32,
  },
  // Current stream for graph capture (all kernels use this stream)
  binfer_set_current_stream: {
    args: [FFIType.u64] as const,  // stream (0 = default)
    returns: FFIType.i32,
  },
  binfer_get_current_stream: {
    args: [] as const,
    returns: FFIType.u64,  // returns stream pointer
  },
  // CUDA Graph capture
  binfer_stream_begin_capture: {
    args: [FFIType.u64, FFIType.i32] as const,  // stream, mode
    returns: FFIType.i32,
  },
  binfer_stream_end_capture: {
    args: [FFIType.u64, FFIType.ptr] as const,  // stream, graph*
    returns: FFIType.i32,
  },
  binfer_graph_instantiate: {
    args: [FFIType.ptr, FFIType.u64] as const,  // graph_exec*, graph
    returns: FFIType.i32,
  },
  binfer_graph_destroy: {
    args: [FFIType.u64] as const,  // graph
    returns: FFIType.i32,
  },
  binfer_graph_exec_destroy: {
    args: [FFIType.u64] as const,  // graph_exec
    returns: FFIType.i32,
  },
  binfer_graph_launch: {
    args: [FFIType.u64, FFIType.u64] as const,  // graph_exec, stream
    returns: FFIType.i32,
  },
  binfer_graph_exec_update: {
    args: [FFIType.u64, FFIType.u64] as const,  // graph_exec, graph
    returns: FFIType.i32,
  },
  binfer_memcpy_h2d_async_stream: {
    args: [FFIType.u64, FFIType.u64, FFIType.u64, FFIType.u64] as const,  // dst, src (pinned), size, stream
    returns: FFIType.i32,
  },

  // GEMM
  binfer_gemm_f16: {
    args: [
      FFIType.u64,  // A - device ptr
      FFIType.u64,  // B - device ptr
      FFIType.u64,  // C - device ptr
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
      FFIType.f32,
      FFIType.f32,
    ] as const,
    returns: FFIType.i32,
  },
  // GEMM with B transposed: C = A @ B^T
  binfer_gemm_f16_transb: {
    args: [
      FFIType.u64,  // A - device ptr
      FFIType.u64,  // B - device ptr
      FFIType.u64,  // C - device ptr
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
      FFIType.f32,
      FFIType.f32,
    ] as const,
    returns: FFIType.i32,
  },
  binfer_gemm_bf16: {
    args: [
      FFIType.u64,  // A - device ptr
      FFIType.u64,  // B - device ptr
      FFIType.u64,  // C - device ptr
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
      FFIType.f32,
      FFIType.f32,
    ] as const,
    returns: FFIType.i32,
  },

  // RMSNorm
  binfer_rmsnorm_f16: {
    args: [
      FFIType.u64,  // output - device ptr
      FFIType.u64,  // input - device ptr
      FFIType.u64,  // weight - device ptr
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
      FFIType.f32,
    ] as const,
    returns: FFIType.i32,
  },

  // RoPE
  binfer_rotary_embedding_f16: {
    args: [
      FFIType.u64,  // q - device ptr
      FFIType.u64,  // k - device ptr
      FFIType.u64,  // cos - device ptr
      FFIType.u64,  // sin - device ptr
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
    ] as const,
    returns: FFIType.i32,
  },

  // Activations
  binfer_silu_f16: {
    args: [FFIType.u64, FFIType.u64, FFIType.u64] as const,
    returns: FFIType.i32,
  },
  binfer_gelu_f16: {
    args: [FFIType.u64, FFIType.u64, FFIType.u64] as const,
    returns: FFIType.i32,
  },
  binfer_swiglu_f16: {
    args: [FFIType.u64, FFIType.u64, FFIType.u64, FFIType.u64] as const,
    returns: FFIType.i32,
  },

  // Element-wise
  binfer_add_f16: {
    args: [FFIType.u64, FFIType.u64, FFIType.u64, FFIType.u64] as const,
    returns: FFIType.i32,
  },
  binfer_mul_f16: {
    args: [FFIType.u64, FFIType.u64, FFIType.u64, FFIType.u64] as const,
    returns: FFIType.i32,
  },

  // Softmax
  binfer_softmax_f16: {
    args: [
      FFIType.u64,  // output - device ptr
      FFIType.u64,  // input - device ptr
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
      FFIType.f32,
    ] as const,
    returns: FFIType.i32,
  },

  // Top-k
  binfer_topk_f16: {
    args: [
      FFIType.u64,  // values - device ptr
      FFIType.u64,  // indices - device ptr
      FFIType.u64,  // input - device ptr
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
    ] as const,
    returns: FFIType.i32,
  },

  // Embedding
  binfer_embedding_f16: {
    args: [
      FFIType.u64,  // output - device ptr
      FFIType.u64,  // weight - device ptr
      FFIType.u64,  // indices - device ptr
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
    ] as const,
    returns: FFIType.i32,
  },

  // Flash Attention
  binfer_flash_attention_f16: {
    args: [
      FFIType.u64,  // Q - device ptr
      FFIType.u64,  // K - device ptr
      FFIType.u64,  // V - device ptr
      FFIType.u64,  // O - device ptr
      FFIType.i32,  // batch_size
      FFIType.i32,  // seq_q
      FFIType.i32,  // seq_kv
      FFIType.i32,  // kv_stride (max_seq_len for cached tensors)
      FFIType.i32,  // q_offset (absolute position for causal mask)
      FFIType.i32,  // num_heads
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.f32,  // softmax_scale
      FFIType.bool, // is_causal
    ] as const,
    returns: FFIType.i32,
  },

  binfer_flash_attention_with_cache_f16: {
    args: [
      FFIType.u64,  // Q - device ptr
      FFIType.u64,  // K_cache - device ptr
      FFIType.u64,  // V_cache - device ptr
      FFIType.u64,  // O - device ptr
      FFIType.i32,  // batch_size
      FFIType.i32,  // cache_seq_len
      FFIType.i32,  // num_heads
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.f32,  // softmax_scale
      FFIType.bool, // is_causal
    ] as const,
    returns: FFIType.i32,
  },

  binfer_kv_cache_update_f16: {
    args: [
      FFIType.u64,  // K_cache - device ptr
      FFIType.u64,  // V_cache - device ptr
      FFIType.u64,  // K_new - device ptr
      FFIType.u64,  // V_new - device ptr
      FFIType.i32,  // batch_size
      FFIType.i32,  // cache_offset
      FFIType.i32,  // seq_new
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.i32,  // max_seq_len
    ] as const,
    returns: FFIType.i32,
  },

  // BF16 Flash Attention
  binfer_flash_attention_bf16: {
    args: [
      FFIType.u64,  // Q - device ptr
      FFIType.u64,  // K - device ptr
      FFIType.u64,  // V - device ptr
      FFIType.u64,  // O - device ptr
      FFIType.i32,  // batch_size
      FFIType.i32,  // seq_q
      FFIType.i32,  // seq_kv
      FFIType.i32,  // kv_stride
      FFIType.i32,  // q_offset
      FFIType.i32,  // num_heads
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.f32,  // softmax_scale
      FFIType.bool, // is_causal
    ] as const,
    returns: FFIType.i32,
  },

  binfer_flash_attention_with_cache_bf16: {
    args: [
      FFIType.u64,  // Q - device ptr
      FFIType.u64,  // K_cache - device ptr
      FFIType.u64,  // V_cache - device ptr
      FFIType.u64,  // O - device ptr
      FFIType.i32,  // batch_size
      FFIType.i32,  // cache_seq_len
      FFIType.i32,  // max_seq_len
      FFIType.i32,  // q_offset
      FFIType.i32,  // num_heads
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.f32,  // softmax_scale
      FFIType.bool, // is_causal
    ] as const,
    returns: FFIType.i32,
  },

  binfer_kv_cache_update_bf16: {
    args: [
      FFIType.u64,  // K_cache - device ptr
      FFIType.u64,  // V_cache - device ptr
      FFIType.u64,  // K_new - device ptr
      FFIType.u64,  // V_new - device ptr
      FFIType.i32,  // batch_size
      FFIType.i32,  // cache_offset
      FFIType.i32,  // seq_new
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.i32,  // max_seq_len
    ] as const,
    returns: FFIType.i32,
  },

  // Paged Attention
  binfer_paged_attention_f16: {
    args: [
      FFIType.u64,  // Q - device ptr
      FFIType.u64,  // K_cache - device ptr (paged blocks)
      FFIType.u64,  // V_cache - device ptr (paged blocks)
      FFIType.u64,  // O - device ptr
      FFIType.ptr,  // block_tables - host ptr to int array
      FFIType.ptr,  // context_lens - host ptr to int array
      FFIType.i32,  // batch_size
      FFIType.i32,  // num_heads
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.i32,  // block_size
      FFIType.i32,  // max_blocks_per_seq (stride of block_tables)
      FFIType.f32,  // softmax_scale
    ] as const,
    returns: FFIType.i32,
  },

  binfer_paged_kv_cache_update_f16: {
    args: [
      FFIType.u64,  // K_cache - device ptr (paged blocks)
      FFIType.u64,  // V_cache - device ptr (paged blocks)
      FFIType.u64,  // K_new - device ptr
      FFIType.u64,  // V_new - device ptr
      FFIType.ptr,  // block_tables - host ptr to int array
      FFIType.ptr,  // context_lens - host ptr to int array
      FFIType.i32,  // num_seqs
      FFIType.i32,  // num_new_tokens
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.i32,  // max_blocks_per_seq
    ] as const,
    returns: FFIType.i32,
  },

  // Paged Attention BF16
  binfer_paged_attention_bf16: {
    args: [
      FFIType.u64,  // Q - device ptr
      FFIType.u64,  // K_cache - device ptr (paged blocks)
      FFIType.u64,  // V_cache - device ptr (paged blocks)
      FFIType.u64,  // O - device ptr
      FFIType.ptr,  // block_tables - host ptr to int array
      FFIType.ptr,  // context_lens - host ptr to int array
      FFIType.i32,  // batch_size
      FFIType.i32,  // num_heads
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.i32,  // block_size
      FFIType.i32,  // max_blocks_per_seq (stride of block_tables)
      FFIType.f32,  // softmax_scale
    ] as const,
    returns: FFIType.i32,
  },

  // Paged Attention with Sinks BF16 (for GPT-OSS)
  binfer_paged_attention_with_sinks_bf16: {
    args: [
      FFIType.u64,  // Q - device ptr
      FFIType.u64,  // K_cache - device ptr (paged blocks)
      FFIType.u64,  // V_cache - device ptr (paged blocks)
      FFIType.u64,  // sinks - device ptr [num_heads]
      FFIType.u64,  // O - device ptr
      FFIType.ptr,  // block_tables - host ptr to int array
      FFIType.ptr,  // context_lens - host ptr to int array
      FFIType.i32,  // batch_size
      FFIType.i32,  // num_heads
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.i32,  // block_size
      FFIType.i32,  // max_blocks_per_seq (stride of block_tables)
      FFIType.f32,  // softmax_scale
    ] as const,
    returns: FFIType.i32,
  },

  binfer_paged_kv_cache_update_bf16: {
    args: [
      FFIType.u64,  // K_cache - device ptr (paged blocks)
      FFIType.u64,  // V_cache - device ptr (paged blocks)
      FFIType.u64,  // K_new - device ptr
      FFIType.u64,  // V_new - device ptr
      FFIType.ptr,  // block_tables - host ptr to int array
      FFIType.ptr,  // context_lens - host ptr to int array
      FFIType.i32,  // num_seqs
      FFIType.i32,  // num_new_tokens
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.i32,  // max_blocks_per_seq
    ] as const,
    returns: FFIType.i32,
  },

  // Device-pointer variants (for pre-uploaded block tables)
  binfer_paged_kv_cache_update_bf16_device: {
    args: [
      FFIType.u64,  // K_cache - device ptr (paged blocks)
      FFIType.u64,  // V_cache - device ptr (paged blocks)
      FFIType.u64,  // K_new - device ptr
      FFIType.u64,  // V_new - device ptr
      FFIType.u64,  // d_block_tables - device ptr (already uploaded)
      FFIType.u64,  // d_context_lens - device ptr (already uploaded)
      FFIType.i32,  // num_seqs
      FFIType.i32,  // num_new_tokens
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.i32,  // max_blocks_per_seq
    ] as const,
    returns: FFIType.i32,
  },

  binfer_paged_attention_bf16_device: {
    args: [
      FFIType.u64,  // Q - device ptr
      FFIType.u64,  // K_cache - device ptr
      FFIType.u64,  // V_cache - device ptr
      FFIType.u64,  // O - device ptr
      FFIType.u64,  // d_block_tables - device ptr (already uploaded)
      FFIType.u64,  // d_context_lens - device ptr (already uploaded)
      FFIType.i32,  // batch_size
      FFIType.i32,  // num_heads
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.i32,  // block_size
      FFIType.i32,  // max_blocks_per_seq
      FFIType.f32,  // softmax_scale
    ] as const,
    returns: FFIType.i32,
  },

  binfer_paged_attention_with_sinks_bf16_device: {
    args: [
      FFIType.u64,  // Q - device ptr
      FFIType.u64,  // K_cache - device ptr
      FFIType.u64,  // V_cache - device ptr
      FFIType.u64,  // sinks - device ptr
      FFIType.u64,  // O - device ptr
      FFIType.u64,  // d_block_tables - device ptr (already uploaded)
      FFIType.u64,  // d_context_lens - device ptr (already uploaded)
      FFIType.i32,  // batch_size
      FFIType.i32,  // num_heads
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.i32,  // block_size
      FFIType.i32,  // max_blocks_per_seq
      FFIType.f32,  // softmax_scale
    ] as const,
    returns: FFIType.i32,
  },

  // Profiling - CUDA events
  binfer_event_create: {
    args: [FFIType.ptr] as const,  // void** event
    returns: FFIType.i32,
  },
  binfer_event_destroy: {
    args: [FFIType.u64] as const,  // void* event (as ptr)
    returns: FFIType.i32,
  },
  binfer_event_record: {
    args: [FFIType.u64] as const,  // void* event (as ptr)
    returns: FFIType.i32,
  },
  binfer_event_synchronize: {
    args: [FFIType.u64] as const,  // void* event (as ptr)
    returns: FFIType.i32,
  },
  binfer_event_elapsed_time: {
    args: [FFIType.ptr, FFIType.u64, FFIType.u64] as const,  // float* ms, void* start, void* end
    returns: FFIType.i32,
  },
  binfer_event_record_stream: {
    args: [FFIType.u64, FFIType.u64] as const,  // void* event, void* stream
    returns: FFIType.i32,
  },
  binfer_stream_wait_event: {
    args: [FFIType.u64, FFIType.u64] as const,  // void* stream, void* event
    returns: FFIType.i32,
  },

  // BF16 kernels
  binfer_gemm_bf16_transb: {
    args: [
      FFIType.u64,  // A - device ptr
      FFIType.u64,  // B - device ptr
      FFIType.u64,  // C - device ptr
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
      FFIType.f32,
      FFIType.f32,
    ] as const,
    returns: FFIType.i32,
  },
  binfer_rmsnorm_bf16: {
    args: [
      FFIType.u64,  // output - device ptr
      FFIType.u64,  // input - device ptr
      FFIType.u64,  // weight - device ptr
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
      FFIType.f32,
    ] as const,
    returns: FFIType.i32,
  },
  binfer_rotary_embedding_bf16: {
    args: [
      FFIType.u64,  // q - device ptr
      FFIType.u64,  // k - device ptr
      FFIType.u64,  // cos - device ptr
      FFIType.u64,  // sin - device ptr
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
    ] as const,
    returns: FFIType.i32,
  },
  binfer_rotary_embedding_decode_bf16: {
    args: [
      FFIType.u64,  // q - device ptr
      FFIType.u64,  // k - device ptr
      FFIType.u64,  // cos - device ptr
      FFIType.u64,  // sin - device ptr
      FFIType.u64,  // positions - device ptr (int32 array)
      FFIType.i32,  // batch_size
      FFIType.i32,  // num_heads
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
    ] as const,
    returns: FFIType.i32,
  },
  binfer_silu_bf16: {
    args: [FFIType.u64, FFIType.u64, FFIType.u64] as const,
    returns: FFIType.i32,
  },
  binfer_swiglu_bf16: {
    args: [FFIType.u64, FFIType.u64, FFIType.u64, FFIType.u64] as const,
    returns: FFIType.i32,
  },
  binfer_add_bf16: {
    args: [FFIType.u64, FFIType.u64, FFIType.u64, FFIType.u64] as const,
    returns: FFIType.i32,
  },
  binfer_add_bias_bf16: {
    args: [FFIType.u64, FFIType.u64, FFIType.u64, FFIType.i32, FFIType.i32] as const,
    returns: FFIType.i32,
  },
  binfer_add_bias_inplace_bf16: {
    args: [FFIType.u64, FFIType.u64, FFIType.i32, FFIType.i32] as const,
    returns: FFIType.i32,
  },
  binfer_argmax_bf16: {
    args: [
      FFIType.u64,  // logits - device ptr [batch_size, vocab_size]
      FFIType.u64,  // output_tokens - device ptr [batch_size] (int32)
      FFIType.i32,  // batch_size
      FFIType.i32,  // vocab_size
    ] as const,
    returns: FFIType.i32,
  },
  binfer_softmax_bf16: {
    args: [
      FFIType.u64,  // output - device ptr
      FFIType.u64,  // input - device ptr
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
      FFIType.f32,
    ] as const,
    returns: FFIType.i32,
  },
  binfer_embedding_bf16: {
    args: [
      FFIType.u64,  // output - device ptr
      FFIType.u64,  // weight - device ptr
      FFIType.u64,  // indices - device ptr
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
      FFIType.i32,
    ] as const,
    returns: FFIType.i32,
  },
  binfer_compute_rope_cache_bf16: {
    args: [
      FFIType.u64,  // cos_cache - device ptr
      FFIType.u64,  // sin_cache - device ptr
      FFIType.i32,
      FFIType.i32,
      FFIType.f32,
      FFIType.f32,
    ] as const,
    returns: FFIType.i32,
  },

  // BF16 to FP16 conversion
  binfer_convert_bf16_to_fp16: {
    args: [
      FFIType.u64,  // input - device ptr
      FFIType.u64,  // output - device ptr
      FFIType.u64,  // num_elements
    ] as const,
    returns: FFIType.i32,
  },

  // ======================================================================
  // MoE (Mixture of Experts) kernels
  // ======================================================================

  // Initialize MXFP4 lookup tables
  binfer_init_mxfp4_tables: {
    args: [] as const,
    returns: FFIType.i32,
  },

  // MXFP4 dequantization
  binfer_mxfp4_dequant: {
    args: [
      FFIType.u64,  // blocks - device ptr (uint8)
      FFIType.u64,  // scales - device ptr (uint8)
      FFIType.u64,  // bias - device ptr (bf16)
      FFIType.u64,  // output - device ptr (bf16)
      FFIType.i32,  // num_experts
      FFIType.i32,  // out_features
      FFIType.i32,  // num_blocks
      FFIType.i32,  // in_features
    ] as const,
    returns: FFIType.i32,
  },

  // MXFP4 single expert dequantization (on-demand for selected experts)
  binfer_mxfp4_dequant_single_expert: {
    args: [
      FFIType.u64,  // blocks - device ptr (uint8)
      FFIType.u64,  // scales - device ptr (uint8)
      FFIType.u64,  // output - device ptr (bf16)
      FFIType.i32,  // expert_idx
      FFIType.i32,  // num_experts
      FFIType.i32,  // out_features
      FFIType.i32,  // num_blocks
      FFIType.i32,  // in_features
    ] as const,
    returns: FFIType.i32,
  },

  // MoE router top-k selection
  binfer_moe_router_topk: {
    args: [
      FFIType.u64,  // hidden - device ptr (bf16)
      FFIType.u64,  // router_weight - device ptr (bf16)
      FFIType.u64,  // router_bias - device ptr (bf16)
      FFIType.u64,  // expert_indices - device ptr (int32)
      FFIType.u64,  // expert_weights - device ptr (bf16)
      FFIType.i32,  // batch_size
      FFIType.i32,  // seq_len
      FFIType.i32,  // hidden_size
      FFIType.i32,  // num_experts
      FFIType.i32,  // top_k
    ] as const,
    returns: FFIType.i32,
  },

  // MoE SwiGLU for fused gate/up
  binfer_moe_swiglu: {
    args: [
      FFIType.u64,  // gate_up - device ptr (bf16)
      FFIType.u64,  // output - device ptr (bf16)
      FFIType.i32,  // batch
      FFIType.i32,  // intermediate_size
    ] as const,
    returns: FFIType.i32,
  },

  // GPT-OSS custom activation (interleaved gate/up with modified gating)
  binfer_gpt_oss_activation: {
    args: [
      FFIType.u64,  // gate_up - device ptr (bf16)
      FFIType.u64,  // output - device ptr (bf16)
      FFIType.i32,  // batch
      FFIType.i32,  // intermediate_size
      FFIType.f32,  // alpha (1.702 for GPT-OSS)
      FFIType.f32,  // limit (7.0 for GPT-OSS)
    ] as const,
    returns: FFIType.i32,
  },

  binfer_scale_add_bf16: {
    args: [
      FFIType.u64,  // input - device ptr (bf16)
      FFIType.u64,  // output - device ptr (bf16, in/out)
      FFIType.f32,  // scale
      FFIType.i32,  // numel
    ] as const,
    returns: FFIType.i32,
  },
  binfer_attention_with_sinks_bf16: {
    args: [
      FFIType.u64,  // Q - device ptr (bf16)
      FFIType.u64,  // K - device ptr (bf16)
      FFIType.u64,  // V - device ptr (bf16)
      FFIType.u64,  // sinks - device ptr (bf16)
      FFIType.u64,  // O - device ptr (bf16)
      FFIType.i32,  // batch_size
      FFIType.i32,  // seq_q
      FFIType.i32,  // seq_kv
      FFIType.i32,  // kv_stride
      FFIType.i32,  // q_offset
      FFIType.i32,  // num_heads
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.f32,  // scale
      FFIType.i32,  // is_causal (bool as i32)
    ] as const,
    returns: FFIType.i32,
  },
  binfer_memset: {
    args: [
      FFIType.u64,  // ptr - device ptr
      FFIType.i32,  // value
      FFIType.u64,  // size in bytes
    ] as const,
    returns: FFIType.i32,
  },

  // Fused MoE forward: dequant + GEMM on-the-fly for single token
  binfer_moe_fused_forward: {
    args: [
      FFIType.u64,  // hidden - device ptr (bf16)
      FFIType.u64,  // gate_up_blocks - device ptr (uint8)
      FFIType.u64,  // gate_up_scales - device ptr (uint8)
      FFIType.u64,  // gate_up_bias - device ptr (bf16) or 0 for nullptr
      FFIType.u64,  // down_blocks - device ptr (uint8)
      FFIType.u64,  // down_scales - device ptr (uint8)
      FFIType.u64,  // down_bias - device ptr (bf16) or 0 for nullptr
      FFIType.u64,  // expert_indices - device ptr (int32)
      FFIType.u64,  // expert_weights - device ptr (float32)
      FFIType.u64,  // output - device ptr (bf16)
      FFIType.i32,  // hidden_size
      FFIType.i32,  // intermediate_size
      FFIType.i32,  // num_experts
      FFIType.i32,  // top_k
    ] as const,
    returns: FFIType.i32,
  },

  // Debug: just run gate_up kernel
  binfer_moe_gate_up_debug: {
    args: [
      FFIType.u64,  // hidden - device ptr (bf16)
      FFIType.u64,  // gate_up_blocks - device ptr (uint8)
      FFIType.u64,  // gate_up_scales - device ptr (uint8)
      FFIType.u64,  // expert_indices - device ptr (int32)
      FFIType.u64,  // gate_up_out - device ptr (bf16)
      FFIType.i32,  // hidden_size
      FFIType.i32,  // intermediate_size
      FFIType.i32,  // num_blocks
      FFIType.i32,  // num_experts
      FFIType.i32,  // top_k
    ] as const,
    returns: FFIType.i32,
  },

  // Expert-parallel MoE forward: for TP>1 where experts are sharded across GPUs
  binfer_moe_fused_forward_ep: {
    args: [
      FFIType.u64,  // hidden - device ptr (bf16)
      FFIType.u64,  // gate_up_blocks - device ptr (uint8) [experts_per_rank, ...]
      FFIType.u64,  // gate_up_scales - device ptr (uint8)
      FFIType.u64,  // gate_up_bias - device ptr (bf16) or 0 for nullptr
      FFIType.u64,  // down_blocks - device ptr (uint8) [experts_per_rank, ...]
      FFIType.u64,  // down_scales - device ptr (uint8)
      FFIType.u64,  // down_bias - device ptr (bf16) or 0 for nullptr
      FFIType.u64,  // expert_indices - device ptr (int32) - global indices
      FFIType.u64,  // expert_weights - device ptr (bf16)
      FFIType.u64,  // output - device ptr (bf16) - partial result
      FFIType.i32,  // hidden_size
      FFIType.i32,  // intermediate_size
      FFIType.i32,  // experts_per_rank
      FFIType.i32,  // top_k
      FFIType.i32,  // rank
      FFIType.i32,  // world_size
    ] as const,
    returns: FFIType.i32,
  },

  // Optimized MoE forward: vectorized loads, larger tiles
  binfer_moe_optimized_forward: {
    args: [
      FFIType.u64,  // hidden - device ptr [numTokens, hiddenSize]
      FFIType.u64,  // gate_up_blocks
      FFIType.u64,  // gate_up_scales
      FFIType.u64,  // gate_up_bias or 0
      FFIType.u64,  // down_blocks
      FFIType.u64,  // down_scales
      FFIType.u64,  // down_bias or 0
      FFIType.u64,  // expert_indices [numTokens, topK]
      FFIType.u64,  // expert_weights [numTokens, topK] bf16
      FFIType.u64,  // output [numTokens, hiddenSize]
      FFIType.u64,  // gate_up_buffer (pre-allocated)
      FFIType.u64,  // activated_buffer (pre-allocated)
      FFIType.i32,  // num_tokens
      FFIType.i32,  // hidden_size
      FFIType.i32,  // intermediate_size
      FFIType.i32,  // num_experts
      FFIType.i32,  // top_k
    ] as const,
    returns: FFIType.i32,
  },

  // Optimized EP MoE forward: for TP>1
  binfer_moe_optimized_forward_ep: {
    args: [
      FFIType.u64,  // hidden
      FFIType.u64,  // gate_up_blocks
      FFIType.u64,  // gate_up_scales
      FFIType.u64,  // gate_up_bias or 0
      FFIType.u64,  // down_blocks
      FFIType.u64,  // down_scales
      FFIType.u64,  // down_bias or 0
      FFIType.u64,  // expert_indices [numTokens, topK]
      FFIType.u64,  // expert_weights [numTokens, topK] bf16
      FFIType.u64,  // output [numTokens, hiddenSize]
      FFIType.u64,  // gate_up_buffer
      FFIType.u64,  // activated_buffer
      FFIType.i32,  // num_tokens
      FFIType.i32,  // hidden_size
      FFIType.i32,  // intermediate_size
      FFIType.i32,  // experts_per_rank
      FFIType.i32,  // top_k
      FFIType.i32,  // rank
      FFIType.i32,  // world_size
    ] as const,
    returns: FFIType.i32,
  },

  // MoE sub-operation profiling
  binfer_moe_enable_profiling: {
    args: [FFIType.bool] as const,  // enable
    returns: FFIType.void,
  },
  binfer_moe_get_profiling: {
    args: [
      FFIType.ptr,  // gate_up_ms (float*)
      FFIType.ptr,  // activation_ms (float*)
      FFIType.ptr,  // down_ms (float*)
      FFIType.ptr,  // call_count (int*)
    ] as const,
    returns: FFIType.void,
  },
  binfer_moe_enable_tensor_cores: {
    args: [FFIType.bool] as const,  // enable
    returns: FFIType.void,
  },
  binfer_moe_preallocate_buffers: {
    args: [
      FFIType.i32,  // num_tokens
      FFIType.i32,  // hidden_size
    ] as const,
    returns: FFIType.i32,
  },
  binfer_moe_tensor_cores_enabled: {
    args: [] as const,
    returns: FFIType.bool,
  },

  // CUTLASS Grouped GEMM MoE
  binfer_init_moe_cutlass_tables: {
    args: [] as const,
    returns: FFIType.i32,
  },
  binfer_moe_cutlass_scratch_size: {
    args: [
      FFIType.i32,  // numTokens
      FFIType.i32,  // hiddenSize
      FFIType.i32,  // intermediateSize
      FFIType.i32,  // numExperts
      FFIType.i32,  // topK
    ] as const,
    returns: FFIType.u64,  // size_t
  },
  binfer_moe_cutlass_forward: {
    args: [
      FFIType.u64,  // hidden
      FFIType.u64,  // gate_up_blocks
      FFIType.u64,  // gate_up_scales
      FFIType.u64,  // gate_up_bias (or 0 for nullptr)
      FFIType.u64,  // down_blocks
      FFIType.u64,  // down_scales
      FFIType.u64,  // down_bias (or 0 for nullptr)
      FFIType.u64,  // expert_indices
      FFIType.u64,  // expert_weights
      FFIType.u64,  // output
      FFIType.u64,  // scratch
      FFIType.u64,  // scratch_size
      FFIType.i32,  // numTokens
      FFIType.i32,  // hiddenSize
      FFIType.i32,  // intermediateSize
      FFIType.i32,  // numExperts
      FFIType.i32,  // topK
    ] as const,
    returns: FFIType.i32,
  },
  binfer_moe_cutlass_forward_async: {
    args: [
      FFIType.u64,  // hidden
      FFIType.u64,  // gate_up_blocks
      FFIType.u64,  // gate_up_scales
      FFIType.u64,  // gate_up_bias (or 0 for nullptr)
      FFIType.u64,  // down_blocks
      FFIType.u64,  // down_scales
      FFIType.u64,  // down_bias (or 0 for nullptr)
      FFIType.u64,  // expert_indices
      FFIType.u64,  // expert_weights
      FFIType.u64,  // output
      FFIType.u64,  // scratch
      FFIType.u64,  // scratch_size
      FFIType.i32,  // numTokens
      FFIType.i32,  // hiddenSize
      FFIType.i32,  // intermediateSize
      FFIType.i32,  // numExperts
      FFIType.i32,  // topK
    ] as const,
    returns: FFIType.i32,
  },
  binfer_moe_cutlass_scratch_size_ep: {
    args: [
      FFIType.i32,  // numTokens
      FFIType.i32,  // hiddenSize
      FFIType.i32,  // intermediateSize
      FFIType.i32,  // expertsPerRank
      FFIType.i32,  // topK
    ] as const,
    returns: FFIType.u64,  // size_t
  },
  binfer_moe_cutlass_forward_ep: {
    args: [
      FFIType.u64,  // hidden
      FFIType.u64,  // gate_up_blocks
      FFIType.u64,  // gate_up_scales
      FFIType.u64,  // gate_up_bias (or 0 for nullptr)
      FFIType.u64,  // down_blocks
      FFIType.u64,  // down_scales
      FFIType.u64,  // down_bias (or 0 for nullptr)
      FFIType.u64,  // expert_indices
      FFIType.u64,  // expert_weights
      FFIType.u64,  // output
      FFIType.u64,  // scratch
      FFIType.u64,  // scratch_size
      FFIType.i32,  // numTokens
      FFIType.i32,  // hiddenSize
      FFIType.i32,  // intermediateSize
      FFIType.i32,  // expertsPerRank
      FFIType.i32,  // topK
      FFIType.i32,  // rank
      FFIType.i32,  // worldSize
    ] as const,
    returns: FFIType.i32,
  },

  // Marlin-style MoE kernel (single kernel launch with work-stealing)
  binfer_moe_marlin_scratch_size_ep: {
    args: [
      FFIType.i32,  // numTokens
      FFIType.i32,  // numExperts
      FFIType.i32,  // hiddenSize
      FFIType.i32,  // intermediateSize
      FFIType.i32,  // topK
    ] as const,
    returns: FFIType.u64,  // size_t
  },
  binfer_moe_marlin_forward_ep: {
    args: [
      FFIType.u64,  // hidden
      FFIType.u64,  // gate_up_blocks
      FFIType.u64,  // gate_up_scales
      FFIType.u64,  // gate_up_bias (or 0 for nullptr)
      FFIType.u64,  // down_blocks
      FFIType.u64,  // down_scales
      FFIType.u64,  // down_bias (or 0 for nullptr)
      FFIType.u64,  // expert_indices
      FFIType.u64,  // expert_weights
      FFIType.u64,  // output
      FFIType.u64,  // scratch
      FFIType.u64,  // scratch_size
      FFIType.i32,  // numTokens
      FFIType.i32,  // hiddenSize
      FFIType.i32,  // intermediateSize
      FFIType.i32,  // expertsPerRank
      FFIType.i32,  // topK
      FFIType.i32,  // rank
      FFIType.i32,  // worldSize
    ] as const,
    returns: FFIType.void,
  },

  // GPU Direct Storage (GDS) / cuFile
  binfer_gds_available: {
    args: [] as const,
    returns: FFIType.i32,
  },
  binfer_gds_init: {
    args: [] as const,
    returns: FFIType.i32,
  },
  binfer_gds_close: {
    args: [] as const,
    returns: FFIType.i32,
  },
  binfer_gds_register_buffer: {
    args: [FFIType.u64, FFIType.u64] as const,  // buffer (device ptr), size
    returns: FFIType.i32,
  },
  binfer_gds_deregister_buffer: {
    args: [FFIType.u64] as const,  // buffer (device ptr)
    returns: FFIType.i32,
  },
  binfer_gds_open: {
    args: [FFIType.ptr, FFIType.ptr] as const,  // path (cstring), handle (out)
    returns: FFIType.i32,
  },
  binfer_gds_close_file: {
    args: [FFIType.u64] as const,  // handle
    returns: FFIType.i32,
  },
  binfer_gds_read: {
    args: [
      FFIType.u64,  // handle
      FFIType.u64,  // gpu_buffer (device ptr)
      FFIType.u64,  // size
      FFIType.u64,  // file_offset
      FFIType.u64,  // buffer_offset
    ] as const,
    returns: FFIType.i64,  // ssize_t (bytes read or error)
  },
} as const;

export interface DeviceProperties {
  name: string;
  totalMemory: number;
}

export class CudaBackend {
  private lib: ReturnType<typeof dlopen<typeof cudaSymbols>> | null = null;
  private _available: boolean = false;

  constructor() {
    try {
      const libPath = findCudaLibrary();
      this.lib = dlopen(libPath, cudaSymbols);
      this._available = true;
    } catch (e) {
      console.warn("CUDA library not found:", e);
      this._available = false;
    }
  }

  async isAvailable(): Promise<boolean> {
    return this._available && this.getDeviceCount() > 0;
  }

  getDeviceCount(): number {
    if (!this.lib) return 0;
    return this.lib.symbols.binfer_get_device_count();
  }

  getDeviceProperties(device: number): DeviceProperties {
    if (!this.lib) {
      throw new Error("CUDA not available");
    }

    const nameBuffer = new Uint8Array(256);
    const memoryBuffer = new BigUint64Array(1);

    const err = this.lib.symbols.binfer_get_device_properties(
      device,
      ptr(nameBuffer),
      256,
      ptr(memoryBuffer)
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to get device properties: error ${err}`);
    }

    // Find null terminator
    let nameEnd = nameBuffer.indexOf(0);
    if (nameEnd === -1) nameEnd = 256;
    const name = new TextDecoder().decode(nameBuffer.subarray(0, nameEnd));

    return {
      name,
      totalMemory: Number(memoryBuffer[0]),
    };
  }

  setDevice(device: number): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_set_device(device);
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to set device: error ${err}`);
    }
  }

  /**
   * Initialize cuBLAS handles for all devices upfront.
   * Call this once at startup to avoid lazy initialization issues with multi-GPU.
   */
  initCublas(numDevices: number): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_init_cublas(numDevices);
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to initialize cuBLAS: error ${err}`);
    }
  }

  /**
   * Get GPU memory info (free and total bytes) for the current device
   */
  getMemInfo(): { free: number; total: number } {
    if (!this.lib) throw new Error("CUDA not available");

    const freeBuffer = new BigUint64Array(1);
    const totalBuffer = new BigUint64Array(1);
    const err = this.lib.symbols.binfer_mem_info(ptr(freeBuffer), ptr(totalBuffer));

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to get memory info: error ${err}`);
    }

    return {
      free: Number(freeBuffer[0]),
      total: Number(totalBuffer[0]),
    };
  }

  // Memory allocation - returns pointer as bigint
  malloc(size: number): bigint {
    if (!this.lib) throw new Error("CUDA not available");

    const ptrBuffer = new BigUint64Array(1);
    const err = this.lib.symbols.binfer_malloc(ptr(ptrBuffer), size);

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to allocate ${size} bytes: error ${err}`);
    }

    return ptrBuffer[0];
  }

  free(devicePtr: bigint): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_free(devicePtr);
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to free memory: error ${err}`);
    }
  }

  memcpyH2D(dst: bigint, src: ArrayBuffer | Uint8Array | Buffer, size: number): void {
    if (!this.lib) throw new Error("CUDA not available");

    // Create a view and keep it referenced to prevent GC during FFI call
    const srcView = src instanceof Uint8Array ? src : new Uint8Array(src);

    const err = this.lib.symbols.binfer_memcpy_h2d(
      dst,
      ptr(srcView),
      size
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to copy H2D: error ${err}`);
    }
  }

  memcpyD2H(dst: ArrayBuffer | Uint8Array | Buffer, src: bigint, size: number): void {
    if (!this.lib) throw new Error("CUDA not available");

    // Create a view and keep it referenced to prevent GC during FFI call
    const dstView = dst instanceof Uint8Array ? dst : new Uint8Array(dst);

    const err = this.lib.symbols.binfer_memcpy_d2h(
      ptr(dstView),
      src,
      size
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to copy D2H: error ${err}`);
    }
  }

  memcpyD2D(dst: bigint, src: bigint, size: number): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_memcpy_d2d(dst, src, size);
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to copy D2D: error ${err}`);
    }
  }

  // Pinned (page-locked) host memory for faster H2D transfers
  mallocHost(size: number): bigint {
    if (!this.lib) throw new Error("CUDA not available");

    const ptrBuffer = new BigUint64Array(1);
    const err = this.lib.symbols.binfer_malloc_host(ptr(ptrBuffer), size);
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to allocate pinned memory: error ${err}`);
    }
    return ptrBuffer[0];
  }

  freeHost(hostPtr: bigint): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_free_host(hostPtr);
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to free pinned memory: error ${err}`);
    }
  }

  // Async H2D copy - source must be pinned memory
  memcpyH2DAsync(dst: bigint, srcPinned: bigint, size: number): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_memcpy_h2d_async(dst, srcPinned, size);
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to async copy H2D: error ${err}`);
    }
  }

  // Copy from regular host memory (e.g., ArrayBuffer or Uint8Array) to pinned memory
  memcpyHostToPinned(pinnedDst: bigint, src: ArrayBuffer | Uint8Array, size: number): void {
    if (!this.lib) throw new Error("CUDA not available");

    const srcArray = src instanceof Uint8Array ? src : new Uint8Array(src);
    const err = this.lib.symbols.binfer_memcpy_host_to_pinned(pinnedDst, ptr(srcArray), size);
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to copy to pinned memory: error ${err}`);
    }
  }

  synchronize(): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_synchronize();
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to synchronize: error ${err}`);
    }
  }

  // CUDA Streams for async overlap
  streamCreate(): bigint {
    if (!this.lib) throw new Error("CUDA not available");

    const streamPtr = new BigUint64Array(1);
    const err = this.lib.symbols.binfer_stream_create(ptr(streamPtr));
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to create stream: error ${err}`);
    }
    return streamPtr[0];
  }

  streamDestroy(stream: bigint): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_stream_destroy(stream);
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to destroy stream: error ${err}`);
    }
  }

  streamSynchronize(stream: bigint): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_stream_synchronize(stream);
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to synchronize stream: error ${err}`);
    }
  }

  /**
   * Set the current stream for all CUDA operations.
   * Used during graph capture - set to capture stream before, reset to 0n after.
   */
  setCurrentStream(stream: bigint): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_set_current_stream(stream);
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to set current stream: error ${err}`);
    }
  }

  /**
   * Get the current stream for CUDA operations.
   */
  getCurrentStream(): bigint {
    if (!this.lib) throw new Error("CUDA not available");
    return this.lib.symbols.binfer_get_current_stream() as bigint;
  }

  memcpyH2DAsyncStream(dst: bigint, srcPinned: bigint, size: number, stream: bigint): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_memcpy_h2d_async_stream(dst, srcPinned, size, stream);
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed async H2D copy: error ${err}`);
    }
  }

  // CUDA Graph capture and replay
  // Capture modes: 0=global, 1=thread-local, 2=relaxed
  streamBeginCapture(stream: bigint, mode: number = 0): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_stream_begin_capture(stream, mode);
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to begin capture: error ${err}`);
    }
  }

  streamEndCapture(stream: bigint): bigint {
    if (!this.lib) throw new Error("CUDA not available");

    const graphPtr = new BigUint64Array(1);
    const err = this.lib.symbols.binfer_stream_end_capture(stream, ptr(graphPtr));
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to end capture: error ${err}`);
    }
    return graphPtr[0];
  }

  graphInstantiate(graph: bigint): bigint {
    if (!this.lib) throw new Error("CUDA not available");

    const execPtr = new BigUint64Array(1);
    const err = this.lib.symbols.binfer_graph_instantiate(ptr(execPtr), graph);
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to instantiate graph: error ${err}`);
    }
    return execPtr[0];
  }

  graphDestroy(graph: bigint): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_graph_destroy(graph);
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to destroy graph: error ${err}`);
    }
  }

  graphExecDestroy(graphExec: bigint): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_graph_exec_destroy(graphExec);
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to destroy graph exec: error ${err}`);
    }
  }

  graphLaunch(graphExec: bigint, stream: bigint): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_graph_launch(graphExec, stream);
    if (err !== BinferError.SUCCESS) {
      throw new Error(`Failed to launch graph: error ${err}`);
    }
  }

  graphExecUpdate(graphExec: bigint, graph: bigint): boolean {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_graph_exec_update(graphExec, graph);
    return err === BinferError.SUCCESS;
  }

  // GEMM operations
  gemmF16(
    A: bigint,
    B: bigint,
    C: bigint,
    M: number,
    N: number,
    K: number,
    alpha: number = 1.0,
    beta: number = 0.0
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_gemm_f16(
      A,
      B,
      C,
      M,
      N,
      K,
      alpha,
      beta
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`GEMM failed: error ${err}`);
    }
  }

  // GEMM with B transposed: C[M,N] = A[M,K] @ B^T where B is [N,K]
  gemmF16TransB(
    A: bigint,
    B: bigint,
    C: bigint,
    M: number,
    N: number,
    K: number,
    alpha: number = 1.0,
    beta: number = 0.0
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_gemm_f16_transb(
      A,
      B,
      C,
      M,
      N,
      K,
      alpha,
      beta
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`GEMM TransB failed: error ${err}`);
    }
  }

  // RMSNorm
  rmsnormF16(
    input: bigint,
    weight: bigint,
    output: bigint,
    batchSize: number,
    seqLen: number,
    hiddenSize: number,
    eps: number = 1e-5
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_rmsnorm_f16(
      input,
      weight,
      output,
      batchSize,
      seqLen,
      hiddenSize,
      eps
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`RMSNorm failed: error ${err}`);
    }
  }

  // SiLU activation
  siluF16(input: bigint, output: bigint, numel: number): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_silu_f16(
      input,
      output,
      numel
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`SiLU failed: error ${err}`);
    }
  }

  // SwiGLU
  swigluF16(
    gate: bigint,
    up: bigint,
    output: bigint,
    numel: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_swiglu_f16(
      gate,
      up,
      output,
      numel
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`SwiGLU failed: error ${err}`);
    }
  }

  // Element-wise add
  addF16(a: bigint, b: bigint, output: bigint, numel: number): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_add_f16(
      a,
      b,
      output,
      numel
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Add failed: error ${err}`);
    }
  }

  // Rotary Position Embedding
  rotaryEmbeddingF16(
    q: bigint,
    k: bigint,
    cos: bigint,
    sin: bigint,
    batchSize: number,
    seqLen: number,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    positionOffset: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_rotary_embedding_f16(
      q,
      k,
      cos,
      sin,
      batchSize,
      seqLen,
      numHeads,
      numKvHeads,
      headDim,
      positionOffset
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`RoPE failed: error ${err}`);
    }
  }

  // Softmax
  softmaxF16(
    input: bigint,
    output: bigint,
    batchSize: number,
    seqLen: number,
    vocabSize: number,
    temperature: number = 1.0
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_softmax_f16(
      input,
      output,
      batchSize,
      seqLen,
      vocabSize,
      temperature
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Softmax failed: error ${err}`);
    }
  }

  // Embedding lookup
  embeddingF16(
    weight: bigint,
    inputIds: bigint,
    output: bigint,
    batchSize: number,
    seqLen: number,
    vocabSize: number,
    hiddenSize: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_embedding_f16(
      weight,
      inputIds,
      output,
      batchSize,
      seqLen,
      vocabSize,
      hiddenSize
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Embedding failed: error ${err}`);
    }
  }

  // Flash Attention
  flashAttentionF16(
    Q: bigint,
    K: bigint,
    V: bigint,
    O: bigint,
    batchSize: number,
    seqQ: number,
    seqKV: number,
    kvStride: number,  // max_seq_len for cached tensors, seqKV for fresh
    qOffset: number,   // absolute position offset for Q (for causal mask in decode)
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    softmaxScale: number,
    isCausal: boolean
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_flash_attention_f16(
      Q,
      K,
      V,
      O,
      batchSize,
      seqQ,
      seqKV,
      kvStride,
      qOffset,
      numHeads,
      numKvHeads,
      headDim,
      softmaxScale,
      isCausal
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Flash attention failed: error ${err}`);
    }
  }

  // Flash Attention with KV cache (for decoding)
  flashAttentionWithCacheF16(
    Q: bigint,
    KCache: bigint,
    VCache: bigint,
    O: bigint,
    batchSize: number,
    cacheSeqLen: number,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    softmaxScale: number,
    isCausal: boolean
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_flash_attention_with_cache_f16(
      Q,
      KCache,
      VCache,
      O,
      batchSize,
      cacheSeqLen,
      numHeads,
      numKvHeads,
      headDim,
      softmaxScale,
      isCausal
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Flash attention with cache failed: error ${err}`);
    }
  }

  // Update KV cache
  kvCacheUpdateF16(
    KCache: bigint,
    VCache: bigint,
    KNew: bigint,
    VNew: bigint,
    batchSize: number,
    cacheOffset: number,
    seqNew: number,
    numKvHeads: number,
    headDim: number,
    maxSeqLen: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_kv_cache_update_f16(
      KCache,
      VCache,
      KNew,
      VNew,
      batchSize,
      cacheOffset,
      seqNew,
      numKvHeads,
      headDim,
      maxSeqLen
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`KV cache update failed: error ${err}`);
    }
  }

  // BF16 Flash Attention
  flashAttentionBf16(
    Q: bigint,
    K: bigint,
    V: bigint,
    O: bigint,
    batchSize: number,
    seqQ: number,
    seqKV: number,
    kvStride: number,
    qOffset: number,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    softmaxScale: number,
    isCausal: boolean
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_flash_attention_bf16(
      Q,
      K,
      V,
      O,
      batchSize,
      seqQ,
      seqKV,
      kvStride,
      qOffset,
      numHeads,
      numKvHeads,
      headDim,
      softmaxScale,
      isCausal
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Flash attention BF16 failed: error ${err}`);
    }
  }

  flashAttentionWithCacheBf16(
    Q: bigint,
    KCache: bigint,
    VCache: bigint,
    O: bigint,
    batchSize: number,
    cacheSeqLen: number,
    maxSeqLen: number,
    qOffset: number,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    softmaxScale: number,
    isCausal: boolean
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_flash_attention_with_cache_bf16(
      Q,
      KCache,
      VCache,
      O,
      batchSize,
      cacheSeqLen,
      maxSeqLen,
      qOffset,
      numHeads,
      numKvHeads,
      headDim,
      softmaxScale,
      isCausal
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Flash attention with cache BF16 failed: error ${err}`);
    }
  }

  kvCacheUpdateBf16(
    KCache: bigint,
    VCache: bigint,
    KNew: bigint,
    VNew: bigint,
    batchSize: number,
    cacheOffset: number,
    seqNew: number,
    numKvHeads: number,
    headDim: number,
    maxSeqLen: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_kv_cache_update_bf16(
      KCache,
      VCache,
      KNew,
      VNew,
      batchSize,
      cacheOffset,
      seqNew,
      numKvHeads,
      headDim,
      maxSeqLen
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`KV cache update BF16 failed: error ${err}`);
    }
  }

  // Paged Attention for variable-length batched sequences
  pagedAttentionF16(
    Q: bigint,
    KCache: bigint,
    VCache: bigint,
    O: bigint,
    blockTables: Int32Array,
    contextLens: Int32Array,
    batchSize: number,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    blockSize: number,
    maxContextLen: number,
    softmaxScale: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_paged_attention_f16(
      Q,
      KCache,
      VCache,
      O,
      ptr(blockTables),
      ptr(contextLens),
      batchSize,
      numHeads,
      numKvHeads,
      headDim,
      blockSize,
      maxContextLen,
      softmaxScale
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Paged attention failed: error ${err}`);
    }
  }

  // Update paged KV cache with new K/V values
  pagedKvCacheUpdateF16(
    KCache: bigint,
    VCache: bigint,
    KNew: bigint,
    VNew: bigint,
    blockTables: Int32Array,
    contextLens: Int32Array,
    numSeqs: number,
    numNewTokens: number,
    numKvHeads: number,
    headDim: number,
    maxBlocksPerSeq: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_paged_kv_cache_update_f16(
      KCache,
      VCache,
      KNew,
      VNew,
      ptr(blockTables),
      ptr(contextLens),
      numSeqs,
      numNewTokens,
      numKvHeads,
      headDim,
      maxBlocksPerSeq
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Paged KV cache update failed: error ${err}`);
    }
  }

  // Paged Attention BF16 for variable-length batched sequences
  pagedAttentionBf16(
    Q: bigint,
    KCache: bigint,
    VCache: bigint,
    O: bigint,
    blockTables: Int32Array,
    contextLens: Int32Array,
    batchSize: number,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    blockSize: number,
    maxBlocksPerSeq: number,
    softmaxScale: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_paged_attention_bf16(
      Q,
      KCache,
      VCache,
      O,
      ptr(blockTables),
      ptr(contextLens),
      batchSize,
      numHeads,
      numKvHeads,
      headDim,
      blockSize,
      maxBlocksPerSeq,
      softmaxScale
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Paged attention BF16 failed: error ${err}`);
    }
  }

  // Paged Attention with Sinks (for GPT-OSS)
  pagedAttentionWithSinksBf16(
    Q: bigint,
    KCache: bigint,
    VCache: bigint,
    sinks: bigint,
    O: bigint,
    blockTables: Int32Array,
    contextLens: Int32Array,
    batchSize: number,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    blockSize: number,
    maxBlocksPerSeq: number,
    softmaxScale: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_paged_attention_with_sinks_bf16(
      Q,
      KCache,
      VCache,
      sinks,
      O,
      ptr(blockTables),
      ptr(contextLens),
      batchSize,
      numHeads,
      numKvHeads,
      headDim,
      blockSize,
      maxBlocksPerSeq,
      softmaxScale
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Paged attention with sinks BF16 failed: error ${err}`);
    }
  }

  // Update paged KV cache with new K/V values (BF16)
  pagedKvCacheUpdateBf16(
    KCache: bigint,
    VCache: bigint,
    KNew: bigint,
    VNew: bigint,
    blockTables: Int32Array,
    contextLens: Int32Array,
    numSeqs: number,
    numNewTokens: number,
    numKvHeads: number,
    headDim: number,
    maxBlocksPerSeq: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_paged_kv_cache_update_bf16(
      KCache,
      VCache,
      KNew,
      VNew,
      ptr(blockTables),
      ptr(contextLens),
      numSeqs,
      numNewTokens,
      numKvHeads,
      headDim,
      maxBlocksPerSeq
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Paged KV cache update BF16 failed: error ${err}`);
    }
  }

  // Device-pointer variants - use when block tables are pre-uploaded to avoid per-call malloc
  pagedKvCacheUpdateBf16Device(
    KCache: bigint,
    VCache: bigint,
    KNew: bigint,
    VNew: bigint,
    dBlockTables: bigint,  // device pointer (already uploaded)
    dContextLens: bigint,  // device pointer (already uploaded)
    numSeqs: number,
    numNewTokens: number,
    numKvHeads: number,
    headDim: number,
    maxBlocksPerSeq: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_paged_kv_cache_update_bf16_device(
      KCache,
      VCache,
      KNew,
      VNew,
      dBlockTables,
      dContextLens,
      numSeqs,
      numNewTokens,
      numKvHeads,
      headDim,
      maxBlocksPerSeq
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Paged KV cache update BF16 (device) failed: error ${err}`);
    }
  }

  pagedAttentionBf16Device(
    Q: bigint,
    KCache: bigint,
    VCache: bigint,
    O: bigint,
    dBlockTables: bigint,  // device pointer (already uploaded)
    dContextLens: bigint,  // device pointer (already uploaded)
    batchSize: number,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    blockSize: number,
    maxBlocksPerSeq: number,
    softmaxScale: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_paged_attention_bf16_device(
      Q,
      KCache,
      VCache,
      O,
      dBlockTables,
      dContextLens,
      batchSize,
      numHeads,
      numKvHeads,
      headDim,
      blockSize,
      maxBlocksPerSeq,
      softmaxScale
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Paged attention BF16 (device) failed: error ${err}`);
    }
  }

  pagedAttentionWithSinksBf16Device(
    Q: bigint,
    KCache: bigint,
    VCache: bigint,
    sinks: bigint,
    O: bigint,
    dBlockTables: bigint,  // device pointer (already uploaded)
    dContextLens: bigint,  // device pointer (already uploaded)
    batchSize: number,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    blockSize: number,
    maxBlocksPerSeq: number,
    softmaxScale: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_paged_attention_with_sinks_bf16_device(
      Q,
      KCache,
      VCache,
      sinks,
      O,
      dBlockTables,
      dContextLens,
      batchSize,
      numHeads,
      numKvHeads,
      headDim,
      blockSize,
      maxBlocksPerSeq,
      softmaxScale
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Paged attention with sinks BF16 (device) failed: error ${err}`);
    }
  }

  // Profiling - CUDA Events
  eventCreate(): bigint {
    if (!this.lib) throw new Error("CUDA not available");

    const ptrBuffer = new BigUint64Array(1);
    const err = this.lib.symbols.binfer_event_create(ptr(ptrBuffer));

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Event create failed: error ${err}`);
    }

    return ptrBuffer[0];
  }

  eventDestroy(event: bigint): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_event_destroy(event);

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Event destroy failed: error ${err}`);
    }
  }

  eventRecord(event: bigint): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_event_record(event);

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Event record failed: error ${err}`);
    }
  }

  eventSynchronize(event: bigint): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_event_synchronize(event);

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Event synchronize failed: error ${err}`);
    }
  }

  eventElapsedTime(start: bigint, end: bigint): number {
    if (!this.lib) throw new Error("CUDA not available");

    const msBuffer = new Float32Array(1);
    const err = this.lib.symbols.binfer_event_elapsed_time(ptr(msBuffer), start, end);

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Event elapsed time failed: error ${err}`);
    }

    return msBuffer[0];
  }

  // Record an event on a specific stream (for multi-device graph capture)
  eventRecordStream(event: bigint, stream: bigint): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_event_record_stream(event, stream);

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Event record stream failed: error ${err}`);
    }
  }

  // Make a stream wait on an event (for multi-device graph capture)
  streamWaitEvent(stream: bigint, event: bigint): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_stream_wait_event(stream, event);

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Stream wait event failed: error ${err}`);
    }
  }

  // ======================================================================
  // BF16 Operations
  // ======================================================================

  gemmBf16(
    A: bigint,
    B: bigint,
    C: bigint,
    M: number,
    N: number,
    K: number,
    alpha: number = 1.0,
    beta: number = 0.0
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_gemm_bf16(
      A,
      B,
      C,
      M,
      N,
      K,
      alpha,
      beta
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`GEMM BF16 failed: error ${err}`);
    }
  }

  gemmBf16TransB(
    A: bigint,
    B: bigint,
    C: bigint,
    M: number,
    N: number,
    K: number,
    alpha: number = 1.0,
    beta: number = 0.0
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_gemm_bf16_transb(
      A,
      B,
      C,
      M,
      N,
      K,
      alpha,
      beta
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`GEMM BF16 TransB failed: error ${err}`);
    }
  }

  rmsnormBf16(
    input: bigint,
    weight: bigint,
    output: bigint,
    batchSize: number,
    seqLen: number,
    hiddenSize: number,
    eps: number = 1e-5
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_rmsnorm_bf16(
      input,
      weight,
      output,
      batchSize,
      seqLen,
      hiddenSize,
      eps
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`RMSNorm BF16 failed: error ${err}`);
    }
  }

  rotaryEmbeddingBf16(
    q: bigint,
    k: bigint,
    cos: bigint,
    sin: bigint,
    batchSize: number,
    seqLen: number,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    positionOffset: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_rotary_embedding_bf16(
      q,
      k,
      cos,
      sin,
      batchSize,
      seqLen,
      numHeads,
      numKvHeads,
      headDim,
      positionOffset
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`RoPE BF16 failed: error ${err}`);
    }
  }

  // Batched RoPE for decode mode: each sequence has 1 token with its own position
  rotaryEmbeddingDecodeBf16(
    q: bigint,
    k: bigint,
    cos: bigint,
    sin: bigint,
    positions: bigint,  // GPU pointer to int32 array of positions
    batchSize: number,
    numHeads: number,
    numKvHeads: number,
    headDim: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_rotary_embedding_decode_bf16(
      q,
      k,
      cos,
      sin,
      positions,
      batchSize,
      numHeads,
      numKvHeads,
      headDim
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`RoPE decode BF16 failed: error ${err}`);
    }
  }

  siluBf16(input: bigint, output: bigint, numel: number): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_silu_bf16(
      input,
      output,
      numel
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`SiLU BF16 failed: error ${err}`);
    }
  }

  swigluBf16(
    gate: bigint,
    up: bigint,
    output: bigint,
    numel: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_swiglu_bf16(
      gate,
      up,
      output,
      numel
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`SwiGLU BF16 failed: error ${err}`);
    }
  }

  addBf16(a: bigint, b: bigint, output: bigint, numel: number): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_add_bf16(
      a,
      b,
      output,
      numel
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Add BF16 failed: error ${err}`);
    }
  }

  // Batched bias addition: output[i,j] = input[i,j] + bias[j]
  // Much more efficient than per-row kernel launches!
  addBiasBf16(input: bigint, bias: bigint, output: bigint, numRows: number, rowSize: number): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_add_bias_bf16(
      input,
      bias,
      output,
      numRows,
      rowSize
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Add bias BF16 failed: error ${err}`);
    }
  }

  // In-place batched bias addition
  addBiasInplaceBf16(data: bigint, bias: bigint, numRows: number, rowSize: number): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_add_bias_inplace_bf16(
      data,
      bias,
      numRows,
      rowSize
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Add bias inplace BF16 failed: error ${err}`);
    }
  }

  /**
   * GPU argmax for sampling - returns token indices instead of full logits
   */
  argmaxBf16(logits: bigint, outputTokens: bigint, batchSize: number, vocabSize: number): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_argmax_bf16(
      logits,
      outputTokens,
      batchSize,
      vocabSize
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Argmax BF16 failed: error ${err}`);
    }
  }

  softmaxBf16(
    input: bigint,
    output: bigint,
    batchSize: number,
    seqLen: number,
    vocabSize: number,
    temperature: number = 1.0
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_softmax_bf16(
      input,
      output,
      batchSize,
      seqLen,
      vocabSize,
      temperature
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Softmax BF16 failed: error ${err}`);
    }
  }

  embeddingBf16(
    weight: bigint,
    inputIds: bigint,
    output: bigint,
    batchSize: number,
    seqLen: number,
    vocabSize: number,
    hiddenSize: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_embedding_bf16(
      weight,
      inputIds,
      output,
      batchSize,
      seqLen,
      vocabSize,
      hiddenSize
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Embedding BF16 failed: error ${err}`);
    }
  }

  computeRopeCacheBf16(
    cosCache: bigint,
    sinCache: bigint,
    maxSeqLen: number,
    headDim: number,
    base: number,
    scalingFactor: number = 1.0
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_compute_rope_cache_bf16(
      cosCache,
      sinCache,
      maxSeqLen,
      headDim,
      base,
      scalingFactor
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Compute RoPE cache BF16 failed: error ${err}`);
    }
  }

  // BF16 to FP16 conversion on GPU
  convertBf16ToFp16(input: bigint, output: bigint, numElements: number): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_convert_bf16_to_fp16(
      input,
      output,
      BigInt(numElements)
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`BF16 to FP16 conversion failed: error ${err}`);
    }
  }

  // ======================================================================
  // MoE (Mixture of Experts) Operations
  // ======================================================================

  // Initialize MXFP4 dequantization lookup tables (call once at startup)
  initMxfp4Tables(): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_init_mxfp4_tables();

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Init MXFP4 tables failed: error ${err}`);
    }
  }

  // Dequantize MXFP4 weights to BF16
  mxfp4Dequant(
    blocks: bigint,
    scales: bigint,
    bias: bigint,
    output: bigint,
    numExperts: number,
    outFeatures: number,
    numBlocks: number,
    inFeatures: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_mxfp4_dequant(
      blocks,
      scales,
      bias,
      output,
      numExperts,
      outFeatures,
      numBlocks,
      inFeatures
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`MXFP4 dequant failed: error ${err}`);
    }
  }

  // Dequantize a single expert's MXFP4 weights to BF16
  mxfp4DequantSingleExpert(
    blocks: bigint,
    scales: bigint,
    output: bigint,
    expertIdx: number,
    numExperts: number,
    outFeatures: number,
    numBlocks: number,
    inFeatures: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_mxfp4_dequant_single_expert(
      blocks,
      scales,
      output,
      expertIdx,
      numExperts,
      outFeatures,
      numBlocks,
      inFeatures
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`MXFP4 single expert dequant failed: error ${err}`);
    }
  }

  // MoE router: compute logits and select top-k experts
  moeRouterTopK(
    hidden: bigint,
    routerWeight: bigint,
    routerBias: bigint,
    expertIndices: bigint,
    expertWeights: bigint,
    batchSize: number,
    seqLen: number,
    hiddenSize: number,
    numExperts: number,
    topK: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_moe_router_topk(
      hidden,
      routerWeight,
      routerBias,
      expertIndices,
      expertWeights,
      batchSize,
      seqLen,
      hiddenSize,
      numExperts,
      topK
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`MoE router topk failed: error ${err}`);
    }
  }

  // MoE SwiGLU: fused gate/up activation
  moeSwiglu(
    gateUp: bigint,
    output: bigint,
    batch: number,
    intermediateSize: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_moe_swiglu(
      gateUp,
      output,
      batch,
      intermediateSize
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`MoE swiglu failed: error ${err}`);
    }
  }

  // GPT-OSS activation: interleaved gate/up with custom gating
  gptOssActivation(
    gateUp: bigint,
    output: bigint,
    batch: number,
    intermediateSize: number,
    alpha: number = 1.702,
    limit: number = 7.0
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_gpt_oss_activation(
      gateUp,
      output,
      batch,
      intermediateSize,
      alpha,
      limit
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`GPT-OSS activation failed: error ${err}`);
    }
  }

  // Scale and add: output = output + scale * input
  scaleAddBf16(
    input: bigint,
    output: bigint,
    scale: number,
    numel: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_scale_add_bf16(
      input,
      output,
      scale,
      numel
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Scale add failed: error ${err}`);
    }
  }

  // Attention with sinks (for GPT-OSS)
  attentionWithSinksBf16(
    Q: bigint, K: bigint, V: bigint, sinks: bigint, O: bigint,
    batchSize: number, seqQ: number, seqKV: number, kvStride: number, qOffset: number,
    numHeads: number, numKvHeads: number, headDim: number,
    scale: number, isCausal: boolean
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_attention_with_sinks_bf16(
      Q, K, V, sinks, O,
      batchSize, seqQ, seqKV, kvStride, qOffset,
      numHeads, numKvHeads, headDim,
      scale, isCausal ? 1 : 0
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Attention with sinks failed: error ${err}`);
    }
  }

  memset(ptr: bigint, value: number, size: number): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_memset(ptr, value, BigInt(size));

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Memset failed: error ${err}`);
    }
  }

  // Fused MoE forward: single-token forward pass without CPU sync
  moeFusedForward(
    hidden: bigint,
    gateUpBlocks: bigint,
    gateUpScales: bigint,
    gateUpBias: bigint,  // 0n for nullptr
    downBlocks: bigint,
    downScales: bigint,
    downBias: bigint,    // 0n for nullptr
    expertIndices: bigint,
    expertWeights: bigint,
    output: bigint,
    hiddenSize: number,
    intermediateSize: number,
    numExperts: number,
    topK: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_moe_fused_forward(
      hidden,
      gateUpBlocks,
      gateUpScales,
      gateUpBias,
      downBlocks,
      downScales,
      downBias,
      expertIndices,
      expertWeights,
      output,
      hiddenSize,
      intermediateSize,
      numExperts,
      topK
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Fused MoE forward failed: error ${err}`);
    }
  }

  // Expert-parallel MoE forward: for TP>1 where experts are sharded across GPUs
  // Each rank only processes experts it owns, result must be allReduced
  moeFusedForwardEP(
    hidden: bigint,
    gateUpBlocks: bigint,
    gateUpScales: bigint,
    gateUpBias: bigint,
    downBlocks: bigint,
    downScales: bigint,
    downBias: bigint,
    expertIndices: bigint,
    expertWeights: bigint,
    output: bigint,
    hiddenSize: number,
    intermediateSize: number,
    expertsPerRank: number,
    topK: number,
    rank: number,
    worldSize: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_moe_fused_forward_ep(
      hidden,
      gateUpBlocks,
      gateUpScales,
      gateUpBias,
      downBlocks,
      downScales,
      downBias,
      expertIndices,
      expertWeights,
      output,
      hiddenSize,
      intermediateSize,
      expertsPerRank,
      topK,
      rank,
      worldSize
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Expert-parallel MoE forward failed: error ${err}`);
    }
  }

  // Optimized MoE forward: vectorized loads, larger tiles
  moeOptimizedForward(
    hidden: bigint,
    gateUpBlocks: bigint,
    gateUpScales: bigint,
    gateUpBias: bigint,
    downBlocks: bigint,
    downScales: bigint,
    downBias: bigint,
    expertIndices: bigint,
    expertWeights: bigint,
    output: bigint,
    gateUpBuffer: bigint,
    activatedBuffer: bigint,
    numTokens: number,
    hiddenSize: number,
    intermediateSize: number,
    numExperts: number,
    topK: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_moe_optimized_forward(
      hidden,
      gateUpBlocks,
      gateUpScales,
      gateUpBias,
      downBlocks,
      downScales,
      downBias,
      expertIndices,
      expertWeights,
      output,
      gateUpBuffer,
      activatedBuffer,
      numTokens,
      hiddenSize,
      intermediateSize,
      numExperts,
      topK
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Optimized MoE forward failed: error ${err}`);
    }
  }

  // Optimized EP MoE forward: for TP>1
  moeOptimizedForwardEP(
    hidden: bigint,
    gateUpBlocks: bigint,
    gateUpScales: bigint,
    gateUpBias: bigint,
    downBlocks: bigint,
    downScales: bigint,
    downBias: bigint,
    expertIndices: bigint,
    expertWeights: bigint,
    output: bigint,
    gateUpBuffer: bigint,
    activatedBuffer: bigint,
    numTokens: number,
    hiddenSize: number,
    intermediateSize: number,
    expertsPerRank: number,
    topK: number,
    rank: number,
    worldSize: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_moe_optimized_forward_ep(
      hidden,
      gateUpBlocks,
      gateUpScales,
      gateUpBias,
      downBlocks,
      downScales,
      downBias,
      expertIndices,
      expertWeights,
      output,
      gateUpBuffer,
      activatedBuffer,
      numTokens,
      hiddenSize,
      intermediateSize,
      expertsPerRank,
      topK,
      rank,
      worldSize
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`Optimized EP MoE forward failed: error ${err}`);
    }
  }

  // MoE sub-operation profiling
  moeEnableProfiling(enable: boolean): void {
    if (!this.lib) throw new Error("CUDA not available");
    this.lib.symbols.binfer_moe_enable_profiling(enable);
  }

  moeGetProfiling(): { gateUpMs: number; activationMs: number; downMs: number; callCount: number } {
    if (!this.lib) throw new Error("CUDA not available");

    const gateUpBuffer = new Float32Array(1);
    const activationBuffer = new Float32Array(1);
    const downBuffer = new Float32Array(1);
    const callCountBuffer = new Int32Array(1);

    this.lib.symbols.binfer_moe_get_profiling(
      ptr(gateUpBuffer),
      ptr(activationBuffer),
      ptr(downBuffer),
      ptr(callCountBuffer)
    );

    return {
      gateUpMs: gateUpBuffer[0],
      activationMs: activationBuffer[0],
      downMs: downBuffer[0],
      callCount: callCountBuffer[0],
    };
  }

  // Tensor core toggle
  moeEnableTensorCores(enable: boolean): void {
    if (!this.lib) throw new Error("CUDA not available");
    this.lib.symbols.binfer_moe_enable_tensor_cores(enable);
  }

  moeTensorCoresEnabled(): boolean {
    if (!this.lib) throw new Error("CUDA not available");
    return this.lib.symbols.binfer_moe_tensor_cores_enabled() as boolean;
  }

  // Pre-allocate MoE buffers for CUDA graph capture
  moePreallocateBuffers(numTokens: number, hiddenSize: number): void {
    if (!this.lib) throw new Error("CUDA not available");
    const err = this.lib.symbols.binfer_moe_preallocate_buffers(numTokens, hiddenSize);
    if (err !== 0) {
      throw new Error(`MoE preallocate buffers failed: error ${err}`);
    }
  }

  // Debug: just run gate_up kernel
  moeGateUpDebug(
    hidden: bigint,
    gateUpBlocks: bigint,
    gateUpScales: bigint,
    expertIndices: bigint,
    gateUpOut: bigint,
    hiddenSize: number,
    intermediateSize: number,
    numBlocks: number,
    numExperts: number,
    topK: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_moe_gate_up_debug(
      hidden,
      gateUpBlocks,
      gateUpScales,
      expertIndices,
      gateUpOut,
      hiddenSize,
      intermediateSize,
      numBlocks,
      numExperts,
      topK
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`moeGateUpDebug failed: error ${err}`);
    }
  }

  // ======================================================================
  // MoE cuBLAS - expert-grouped batching for maximum GPU utilization
  // ======================================================================

  // CUTLASS Grouped GEMM MoE
  initMoeCutlassTables(): void {
    if (!this.lib) throw new Error("CUDA not available");
    const err = this.lib.symbols.binfer_init_moe_cutlass_tables();
    if (err !== BinferError.SUCCESS) {
      throw new Error(`initMoeCutlassTables failed: error ${err}`);
    }
  }

  moeCutlassScratchSize(
    numTokens: number,
    hiddenSize: number,
    intermediateSize: number,
    numExperts: number,
    topK: number
  ): bigint {
    if (!this.lib) throw new Error("CUDA not available");
    return this.lib.symbols.binfer_moe_cutlass_scratch_size(
      numTokens, hiddenSize, intermediateSize, numExperts, topK
    ) as bigint;
  }

  moeCutlassForward(
    hidden: bigint,
    gateUpBlocks: bigint,
    gateUpScales: bigint,
    gateUpBias: bigint,
    downBlocks: bigint,
    downScales: bigint,
    downBias: bigint,
    expertIndices: bigint,
    expertWeights: bigint,
    output: bigint,
    scratch: bigint,
    scratchSize: bigint,
    numTokens: number,
    hiddenSize: number,
    intermediateSize: number,
    numExperts: number,
    topK: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_moe_cutlass_forward(
      hidden,
      gateUpBlocks,
      gateUpScales,
      gateUpBias,
      downBlocks,
      downScales,
      downBias,
      expertIndices,
      expertWeights,
      output,
      scratch,
      scratchSize,
      numTokens,
      hiddenSize,
      intermediateSize,
      numExperts,
      topK
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`moeCutlassForward failed: error ${err}`);
    }
  }

  moeCutlassForwardAsync(
    hidden: bigint,
    gateUpBlocks: bigint,
    gateUpScales: bigint,
    gateUpBias: bigint,
    downBlocks: bigint,
    downScales: bigint,
    downBias: bigint,
    expertIndices: bigint,
    expertWeights: bigint,
    output: bigint,
    scratch: bigint,
    scratchSize: bigint,
    numTokens: number,
    hiddenSize: number,
    intermediateSize: number,
    numExperts: number,
    topK: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_moe_cutlass_forward_async(
      hidden,
      gateUpBlocks,
      gateUpScales,
      gateUpBias,
      downBlocks,
      downScales,
      downBias,
      expertIndices,
      expertWeights,
      output,
      scratch,
      scratchSize,
      numTokens,
      hiddenSize,
      intermediateSize,
      numExperts,
      topK
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`moeCutlassForwardAsync failed: error ${err}`);
    }
  }

  moeCutlassScratchSizeEP(
    numTokens: number,
    hiddenSize: number,
    intermediateSize: number,
    expertsPerRank: number,
    topK: number
  ): bigint {
    if (!this.lib) throw new Error("CUDA not available");
    return this.lib.symbols.binfer_moe_cutlass_scratch_size_ep(
      numTokens, hiddenSize, intermediateSize, expertsPerRank, topK
    ) as bigint;
  }

  moeCutlassForwardEP(
    hidden: bigint,
    gateUpBlocks: bigint,
    gateUpScales: bigint,
    gateUpBias: bigint,
    downBlocks: bigint,
    downScales: bigint,
    downBias: bigint,
    expertIndices: bigint,
    expertWeights: bigint,
    output: bigint,
    scratch: bigint,
    scratchSize: bigint,
    numTokens: number,
    hiddenSize: number,
    intermediateSize: number,
    expertsPerRank: number,
    topK: number,
    rank: number,
    worldSize: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    const err = this.lib.symbols.binfer_moe_cutlass_forward_ep(
      hidden,
      gateUpBlocks,
      gateUpScales,
      gateUpBias,
      downBlocks,
      downScales,
      downBias,
      expertIndices,
      expertWeights,
      output,
      scratch,
      scratchSize,
      numTokens,
      hiddenSize,
      intermediateSize,
      expertsPerRank,
      topK,
      rank,
      worldSize
    );

    if (err !== BinferError.SUCCESS) {
      throw new Error(`moeCutlassForwardEP failed: error ${err}`);
    }
  }

  // =========================================================================
  // Marlin-style MoE kernel methods (single kernel launch with work-stealing)
  // =========================================================================

  /**
   * Get required scratch space size for Marlin MoE kernel (EP version).
   */
  moeMarlinScratchSizeEP(
    numTokens: number,
    numExperts: number,
    hiddenSize: number,
    intermediateSize: number,
    topK: number
  ): bigint {
    if (!this.lib) throw new Error("CUDA not available");
    return this.lib.symbols.binfer_moe_marlin_scratch_size_ep(
      numTokens, numExperts, hiddenSize, intermediateSize, topK
    ) as bigint;
  }

  /**
   * Run Marlin-style MoE forward pass with work-stealing (EP version).
   * Single kernel launch for all experts.
   */
  moeMarlinForwardEP(
    hidden: bigint,
    gateUpBlocks: bigint,
    gateUpScales: bigint,
    gateUpBias: bigint,
    downBlocks: bigint,
    downScales: bigint,
    downBias: bigint,
    expertIndices: bigint,
    expertWeights: bigint,
    output: bigint,
    scratch: bigint,
    scratchSize: bigint,
    numTokens: number,
    hiddenSize: number,
    intermediateSize: number,
    expertsPerRank: number,
    topK: number,
    rank: number,
    worldSize: number
  ): void {
    if (!this.lib) throw new Error("CUDA not available");

    this.lib.symbols.binfer_moe_marlin_forward_ep(
      hidden,
      gateUpBlocks,
      gateUpScales,
      gateUpBias,
      downBlocks,
      downScales,
      downBias,
      expertIndices,
      expertWeights,
      output,
      scratch,
      scratchSize,
      numTokens,
      hiddenSize,
      intermediateSize,
      expertsPerRank,
      topK,
      rank,
      worldSize
    );
  }

  // =========================================================================
  // GPU Direct Storage (GDS) methods
  // =========================================================================

  /**
   * Check if GDS is available on this system.
   */
  gdsAvailable(): boolean {
    if (!this.lib) return false;
    return this.lib.symbols.binfer_gds_available() === 1;
  }

  /**
   * Initialize the GDS driver.
   */
  gdsInit(): boolean {
    if (!this.lib) return false;
    const err = this.lib.symbols.binfer_gds_init();
    return err === BinferError.SUCCESS;
  }

  /**
   * Close the GDS driver.
   */
  gdsClose(): void {
    if (!this.lib) return;
    this.lib.symbols.binfer_gds_close();
  }

  /**
   * Register a GPU buffer for GDS DMA.
   */
  gdsRegisterBuffer(buffer: bigint, size: number): boolean {
    if (!this.lib) return false;
    const err = this.lib.symbols.binfer_gds_register_buffer(buffer, BigInt(size));
    return err === BinferError.SUCCESS;
  }

  /**
   * Deregister a GPU buffer from GDS.
   */
  gdsDeregisterBuffer(buffer: bigint): boolean {
    if (!this.lib) return false;
    const err = this.lib.symbols.binfer_gds_deregister_buffer(buffer);
    return err === BinferError.SUCCESS;
  }

  /**
   * Open a file for GDS.
   * Returns the file handle, or null if the file couldn't be opened with GDS.
   */
  gdsOpen(path: string): bigint | null {
    if (!this.lib) return null;

    const pathBuffer = Buffer.from(path + "\0", "utf-8");
    const handleBuffer = new ArrayBuffer(8);

    const err = this.lib.symbols.binfer_gds_open(ptr(pathBuffer), ptr(handleBuffer));
    if (err !== BinferError.SUCCESS) {
      return null;
    }

    const handleView = new BigUint64Array(handleBuffer);
    return handleView[0];
  }

  /**
   * Close a GDS file handle.
   */
  gdsCloseFile(handle: bigint): void {
    if (!this.lib) return;
    this.lib.symbols.binfer_gds_close_file(handle);
  }

  /**
   * Read from a file directly to GPU memory using GDS.
   * Returns the number of bytes read, or -1 on error.
   */
  gdsRead(
    handle: bigint,
    gpuBuffer: bigint,
    size: number,
    fileOffset: number,
    bufferOffset: number = 0
  ): number {
    if (!this.lib) return -1;

    const bytesRead = this.lib.symbols.binfer_gds_read(
      handle,
      gpuBuffer,
      BigInt(size),
      BigInt(fileOffset),
      BigInt(bufferOffset)
    );

    return Number(bytesRead);
  }

}

// Singleton for convenience
let _defaultBackend: CudaBackend | null = null;

export function getCudaBackend(): CudaBackend {
  if (!_defaultBackend) {
    _defaultBackend = new CudaBackend();
  }
  return _defaultBackend;
}

/**
 * CudaGraph - Easy-to-use wrapper for CUDA graph capture and replay.
 *
 * Usage:
 *   const graph = new CudaGraph(cuda);
 *
 *   // First run: capture the graph
 *   if (!graph.isReady()) {
 *     graph.beginCapture();
 *     // ... run your CUDA operations ...
 *     graph.endCapture();
 *   }
 *
 *   // Subsequent runs: replay the graph
 *   graph.launch();
 *   graph.sync();
 */
export class CudaGraph {
  private cuda: CudaBackend;
  private stream: bigint;
  private graph: bigint | null = null;
  private graphExec: bigint | null = null;
  private _isCapturing: boolean = false;

  constructor(cuda: CudaBackend) {
    this.cuda = cuda;
    this.stream = cuda.streamCreate();
  }

  /**
   * Check if the graph has been captured and is ready to replay.
   */
  isReady(): boolean {
    return this.graphExec !== null;
  }

  /**
   * Check if we're currently capturing.
   */
  isCapturing(): boolean {
    return this._isCapturing;
  }

  /**
   * Get the stream used for capture/replay.
   * Use this stream for all operations you want captured.
   */
  getStream(): bigint {
    return this.stream;
  }

  /**
   * Begin capturing operations into a graph.
   * All CUDA operations will be captured until endCapture().
   * This sets the current stream so all kernels run on the capture stream.
   */
  beginCapture(): void {
    if (this._isCapturing) {
      throw new Error("Already capturing");
    }
    // Set current stream so all kernels use our capture stream
    this.cuda.setCurrentStream(this.stream);
    // Use relaxed mode (2) to allow operations from any thread
    this.cuda.streamBeginCapture(this.stream, 2);
    this._isCapturing = true;
  }

  /**
   * End capture and create the executable graph.
   */
  endCapture(): void {
    if (!this._isCapturing) {
      throw new Error("Not capturing");
    }
    this._isCapturing = false;

    // End capture to get the graph
    this.graph = this.cuda.streamEndCapture(this.stream);

    // Reset current stream back to default
    this.cuda.setCurrentStream(0n);

    // Instantiate into an executable
    this.graphExec = this.cuda.graphInstantiate(this.graph);

    // We can destroy the graph now, we only need the executable
    this.cuda.graphDestroy(this.graph);
    this.graph = null;
  }

  /**
   * Launch the captured graph.
   * Must call sync() or wait otherwise before reading results.
   */
  launch(): void {
    if (!this.graphExec) {
      throw new Error("Graph not captured yet");
    }
    this.cuda.graphLaunch(this.graphExec, this.stream);
  }

  /**
   * Synchronize - wait for graph execution to complete.
   */
  sync(): void {
    this.cuda.streamSynchronize(this.stream);
  }

  /**
   * Destroy the graph and free resources.
   */
  destroy(): void {
    if (this.graphExec) {
      this.cuda.graphExecDestroy(this.graphExec);
      this.graphExec = null;
    }
    if (this.graph) {
      this.cuda.graphDestroy(this.graph);
      this.graph = null;
    }
    this.cuda.streamDestroy(this.stream);
  }
}

/**
 * Multi-device CUDA Graph for tensor parallelism.
 *
 * Captures operations across multiple GPUs into a single graph by using
 * stream events to fork/join the capture between devices.
 *
 * Usage:
 *   const graph = new MultiDeviceCudaGraph(cuda, numDevices);
 *   graph.beginCapture();
 *
 *   for (const device of devices) {
 *     graph.switchToDevice(device);
 *     // ... run CUDA operations on this device ...
 *   }
 *
 *   graph.endCapture();
 *   graph.launch();
 *   graph.sync();
 */
export class MultiDeviceCudaGraph {
  private cuda: CudaBackend;
  private numDevices: number;
  private streams: Map<number, bigint> = new Map();  // Per-device streams
  private events: Map<number, bigint> = new Map();   // Per-device fork/join events
  private graph: bigint | null = null;
  private graphExec: bigint | null = null;
  private _isCapturing: boolean = false;
  private currentDevice: number = 0;
  private originDevice: number = 0;

  constructor(cuda: CudaBackend, numDevices: number) {
    this.cuda = cuda;
    this.numDevices = numDevices;

    // Create a stream and event on each device
    for (let device = 0; device < numDevices; device++) {
      cuda.setDevice(device);
      const stream = cuda.streamCreate();
      const event = cuda.eventCreate();
      this.streams.set(device, stream);
      this.events.set(device, event);
    }

    // Reset to device 0
    cuda.setDevice(0);
  }

  isReady(): boolean {
    return this.graphExec !== null;
  }

  isCapturing(): boolean {
    return this._isCapturing;
  }

  getStream(device?: number): bigint {
    const dev = device ?? this.currentDevice;
    return this.streams.get(dev)!;
  }

  /**
   * Begin capturing operations into a graph.
   * Starts capture on device 0's stream.
   */
  beginCapture(): void {
    if (this._isCapturing) {
      throw new Error("Already capturing");
    }

    this.originDevice = 0;
    this.currentDevice = 0;
    this.cuda.setDevice(0);

    const stream = this.streams.get(0)!;
    this.cuda.setCurrentStream(stream);
    // Use relaxed mode (2) for multi-device capture
    this.cuda.streamBeginCapture(stream, 2);
    this._isCapturing = true;
  }

  /**
   * Switch capture to a different device.
   * Uses events to fork the capture graph to the new device's stream.
   */
  switchToDevice(device: number): void {
    if (!this._isCapturing) {
      throw new Error("Not capturing");
    }

    if (device === this.currentDevice) {
      // Already on this device, just make sure it's active
      this.cuda.setDevice(device);
      return;
    }

    const fromStream = this.streams.get(this.currentDevice)!;
    const toStream = this.streams.get(device)!;
    const event = this.events.get(this.currentDevice)!;

    // Record event on current stream to mark where we're leaving
    this.cuda.eventRecordStream(event, fromStream);

    // Switch to target device
    this.cuda.setDevice(device);

    // Make target stream wait on the event - this forks the capture
    this.cuda.streamWaitEvent(toStream, event);

    // Set current stream to target device's stream
    this.cuda.setCurrentStream(toStream);

    this.currentDevice = device;
  }

  /**
   * End capture and create the executable graph.
   * Must be called from the origin device (device 0).
   */
  endCapture(): void {
    if (!this._isCapturing) {
      throw new Error("Not capturing");
    }

    // If we're on a different device, join back to origin
    if (this.currentDevice !== this.originDevice) {
      const fromStream = this.streams.get(this.currentDevice)!;
      const toStream = this.streams.get(this.originDevice)!;
      const event = this.events.get(this.currentDevice)!;

      // Record event on current stream
      this.cuda.eventRecordStream(event, fromStream);

      // Switch back to origin device
      this.cuda.setDevice(this.originDevice);

      // Make origin stream wait on event - this joins the capture
      this.cuda.streamWaitEvent(toStream, event);

      this.cuda.setCurrentStream(toStream);
      this.currentDevice = this.originDevice;
    }

    this._isCapturing = false;

    // End capture on origin stream
    const originStream = this.streams.get(this.originDevice)!;
    this.graph = this.cuda.streamEndCapture(originStream);

    // Reset current stream
    this.cuda.setCurrentStream(0n);

    // Instantiate the graph
    this.graphExec = this.cuda.graphInstantiate(this.graph);

    // Destroy intermediate graph
    this.cuda.graphDestroy(this.graph);
    this.graph = null;
  }

  /**
   * Launch the captured graph.
   * Launches on origin device's stream.
   */
  launch(): void {
    if (!this.graphExec) {
      throw new Error("Graph not captured yet");
    }
    this.cuda.setDevice(this.originDevice);
    const stream = this.streams.get(this.originDevice)!;
    this.cuda.graphLaunch(this.graphExec, stream);
  }

  /**
   * Synchronize - wait for graph execution to complete.
   */
  sync(): void {
    // Sync all device streams to ensure all work is complete
    for (let device = 0; device < this.numDevices; device++) {
      this.cuda.setDevice(device);
      this.cuda.streamSynchronize(this.streams.get(device)!);
    }
    // Reset to device 0
    this.cuda.setDevice(0);
  }

  /**
   * Destroy the graph and free resources.
   */
  destroy(): void {
    if (this.graphExec) {
      this.cuda.graphExecDestroy(this.graphExec);
      this.graphExec = null;
    }
    if (this.graph) {
      this.cuda.graphDestroy(this.graph);
      this.graph = null;
    }

    // Destroy streams and events on each device
    for (let device = 0; device < this.numDevices; device++) {
      this.cuda.setDevice(device);
      const stream = this.streams.get(device);
      const event = this.events.get(device);
      if (stream) this.cuda.streamDestroy(stream);
      if (event) this.cuda.eventDestroy(event);
    }

    this.streams.clear();
    this.events.clear();
    this.cuda.setDevice(0);
  }
}
