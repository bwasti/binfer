// FlashAttention-3 FFI Bindings for Bun
// Provides high-performance attention on H100 GPUs

import { dlopen, FFIType, suffix } from "bun:ffi";
import { existsSync } from "fs";

// Error codes matching fa3_api.cu
export enum FA3Error {
  SUCCESS = 0,
  ERROR_INVALID_ARGUMENT = 1,
  ERROR_CUDA = 2,
  ERROR_UNSUPPORTED = 3,
}

// Find the FA3 library
function findFA3Library(): string {
  const possiblePaths = [
    `./cuda/build/libbinfer_fa3.${suffix}`,
    `./cuda/fa3/build/libbinfer_fa3.${suffix}`,
    `../cuda/build/libbinfer_fa3.${suffix}`,
    `/usr/local/lib/libbinfer_fa3.${suffix}`,
  ];

  for (const path of possiblePaths) {
    if (existsSync(path)) {
      return path;
    }
  }

  return ""; // Not found, will be handled gracefully
}

// FFI symbols definition
const fa3Symbols = {
  // BF16 forward pass
  fa3_fwd_bf16: {
    args: [
      FFIType.u64,  // Q - device ptr
      FFIType.u64,  // K - device ptr
      FFIType.u64,  // V - device ptr
      FFIType.u64,  // O - device ptr
      FFIType.u64,  // softmax_lse (can be 0/nullptr)
      FFIType.i32,  // batch_size
      FFIType.i32,  // seq_q
      FFIType.i32,  // seq_kv
      FFIType.i32,  // num_heads
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.f32,  // softmax_scale
      FFIType.bool, // is_causal
      FFIType.u64,  // stream (can be 0 for default)
    ] as const,
    returns: FFIType.i32,
  },

  // FP16 forward pass
  fa3_fwd_fp16: {
    args: [
      FFIType.u64,  // Q - device ptr
      FFIType.u64,  // K - device ptr
      FFIType.u64,  // V - device ptr
      FFIType.u64,  // O - device ptr
      FFIType.u64,  // softmax_lse (can be 0/nullptr)
      FFIType.i32,  // batch_size
      FFIType.i32,  // seq_q
      FFIType.i32,  // seq_kv
      FFIType.i32,  // num_heads
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.f32,  // softmax_scale
      FFIType.bool, // is_causal
      FFIType.u64,  // stream (can be 0 for default)
    ] as const,
    returns: FFIType.i32,
  },

  // BF16 forward with variable-length queries and paged KV cache
  fa3_fwd_varlen_paged_bf16: {
    args: [
      FFIType.u64,  // Q - device ptr [total_q, num_heads, head_dim]
      FFIType.u64,  // K_cache - device ptr [num_blocks, block_size, num_kv_heads, head_dim]
      FFIType.u64,  // V_cache - device ptr [num_blocks, block_size, num_kv_heads, head_dim]
      FFIType.u64,  // O - device ptr [total_q, num_heads, head_dim]
      FFIType.u64,  // softmax_lse - device ptr (can be 0 for nullptr)
      FFIType.u64,  // cu_seqlens_q - device ptr [batch_size + 1]
      FFIType.u64,  // seqused_k - device ptr [batch_size]
      FFIType.u64,  // block_table - device ptr [batch_size, max_blocks_per_seq]
      FFIType.i32,  // batch_size
      FFIType.i32,  // max_seqlen_q
      FFIType.i32,  // max_seqlen_k
      FFIType.i32,  // num_heads
      FFIType.i32,  // num_kv_heads
      FFIType.i32,  // head_dim
      FFIType.i32,  // block_size
      FFIType.i32,  // max_blocks_per_seq
      FFIType.f32,  // softmax_scale
      FFIType.bool, // is_causal
      FFIType.u64,  // stream (can be 0 for default stream)
    ] as const,
    returns: FFIType.i32,
  },
};

export class FA3Backend {
  private lib: ReturnType<typeof dlopen<typeof fa3Symbols>> | null = null;
  private _available: boolean = false;

  constructor() {
    try {
      const libPath = findFA3Library();
      if (libPath) {
        this.lib = dlopen(libPath, fa3Symbols);
        this._available = true;
      }
    } catch (e) {
      // FA3 not available, will fall back to custom kernels
      this._available = false;
    }
  }

  get available(): boolean {
    return this._available;
  }

  /**
   * FlashAttention-3 forward pass for BF16
   *
   * @param Q Query tensor [batch, seq_q, num_heads, head_dim]
   * @param K Key tensor [batch, seq_kv, num_kv_heads, head_dim]
   * @param V Value tensor [batch, seq_kv, num_kv_heads, head_dim]
   * @param O Output tensor [batch, seq_q, num_heads, head_dim]
   * @param batchSize Batch size
   * @param seqQ Query sequence length
   * @param seqKV Key/value sequence length
   * @param numHeads Number of query heads
   * @param numKvHeads Number of KV heads (for GQA)
   * @param headDim Head dimension (must be 64, 96, 128, 192, or 256)
   * @param softmaxScale Softmax scale factor (typically 1/sqrt(head_dim))
   * @param isCausal Whether to use causal masking
   */
  flashAttentionBF16(
    Q: bigint,
    K: bigint,
    V: bigint,
    O: bigint,
    batchSize: number,
    seqQ: number,
    seqKV: number,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    softmaxScale: number,
    isCausal: boolean
  ): void {
    if (!this.lib) {
      throw new Error("FA3 library not loaded");
    }

    const result = this.lib.symbols.fa3_fwd_bf16(
      Q,
      K,
      V,
      O,
      0n, // softmax_lse = nullptr
      batchSize,
      seqQ,
      seqKV,
      numHeads,
      numKvHeads,
      headDim,
      softmaxScale,
      isCausal,
      0n  // default stream
    );

    if (result !== FA3Error.SUCCESS) {
      throw new Error(`FA3 BF16 attention failed with error ${result}`);
    }
  }

  /**
   * FlashAttention-3 forward pass for FP16
   */
  flashAttentionFP16(
    Q: bigint,
    K: bigint,
    V: bigint,
    O: bigint,
    batchSize: number,
    seqQ: number,
    seqKV: number,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    softmaxScale: number,
    isCausal: boolean
  ): void {
    if (!this.lib) {
      throw new Error("FA3 library not loaded");
    }

    const result = this.lib.symbols.fa3_fwd_fp16(
      Q,
      K,
      V,
      O,
      0n, // softmax_lse = nullptr
      batchSize,
      seqQ,
      seqKV,
      numHeads,
      numKvHeads,
      headDim,
      softmaxScale,
      isCausal,
      0n  // default stream
    );

    if (result !== FA3Error.SUCCESS) {
      throw new Error(`FA3 FP16 attention failed with error ${result}`);
    }
  }

  /**
   * FlashAttention-3 with variable-length queries and paged KV cache.
   * Used for continuous batching / dynamic batching inference.
   *
   * @param Q Query tensor [total_q_tokens, num_heads, head_dim]
   * @param KCache Paged K cache [num_blocks, block_size, num_kv_heads, head_dim]
   * @param VCache Paged V cache [num_blocks, block_size, num_kv_heads, head_dim]
   * @param O Output tensor [total_q_tokens, num_heads, head_dim]
   * @param cuSeqlensQ Cumulative query lengths [batch_size + 1] (device ptr)
   * @param seqUsedK Per-sequence K lengths [batch_size] (device ptr)
   * @param blockTable Block table [batch_size, maxBlocksPerSeq] (device ptr)
   * @param batchSize Number of sequences
   * @param maxSeqlenQ Maximum query sequence length
   * @param maxSeqlenK Maximum KV sequence length
   * @param numHeads Number of query heads
   * @param numKvHeads Number of KV heads
   * @param headDim Head dimension
   * @param blockSize Tokens per block (typically 16)
   * @param maxBlocksPerSeq Maximum blocks per sequence
   * @param softmaxScale Scale factor (typically 1/sqrt(head_dim))
   * @param isCausal Whether to use causal masking
   */
  flashAttentionPagedBF16(
    Q: bigint,
    KCache: bigint,
    VCache: bigint,
    O: bigint,
    cuSeqlensQ: bigint,
    seqUsedK: bigint,
    blockTable: bigint,
    batchSize: number,
    maxSeqlenQ: number,
    maxSeqlenK: number,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    blockSize: number,
    maxBlocksPerSeq: number,
    softmaxScale: number,
    isCausal: boolean
  ): void {
    if (!this.lib) {
      throw new Error("FA3 library not loaded");
    }

    const result = this.lib.symbols.fa3_fwd_varlen_paged_bf16(
      Q,
      KCache,
      VCache,
      O,
      0n, // softmax_lse = nullptr
      cuSeqlensQ,
      seqUsedK,
      blockTable,
      batchSize,
      maxSeqlenQ,
      maxSeqlenK,
      numHeads,
      numKvHeads,
      headDim,
      blockSize,
      maxBlocksPerSeq,
      softmaxScale,
      isCausal,
      0n  // default stream
    );

    if (result !== FA3Error.SUCCESS) {
      throw new Error(`FA3 paged BF16 attention failed with error ${result}`);
    }
  }
}

// Singleton instance
let fa3Instance: FA3Backend | null = null;

export function getFA3Backend(): FA3Backend {
  if (!fa3Instance) {
    fa3Instance = new FA3Backend();
  }
  return fa3Instance;
}
