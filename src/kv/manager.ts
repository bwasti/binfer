// KV Cache Manager - contiguous memory implementation
// For basic inference without paging (simpler, good for single requests)

import { Tensor, DType } from "../tensor/tensor";
import { getCudaBackend, CudaBackend } from "../backend/cuda/bindings";

export interface KVCacheConfig {
  numLayers: number;
  numKvHeads: number;
  headDim: number;
  maxSeqLen: number;
  dtype: DType;
}

export interface LayerKVCache {
  k: Tensor;  // [batch, max_seq, num_kv_heads, head_dim]
  v: Tensor;  // [batch, max_seq, num_kv_heads, head_dim]
}

/**
 * Interface for KV cache implementations.
 * Allows for both contiguous and device-local implementations.
 */
export interface IKVCache {
  readonly seqLen: number;
  readonly batchSize: number;
  readonly maxSeqLen: number;
  getLayer(layerIdx: number): LayerKVCache;
  update(layerIdx: number, kNew: Tensor, vNew: Tensor): LayerKVCache;
  advanceSeqLen(numTokens: number): void;
  reset(): void;
  getValidCache(layerIdx: number): LayerKVCache;
  dispose(): void;
  memoryUsage(): number;
}

/**
 * Contiguous KV cache for a single batch.
 * Simpler than paged attention but less memory efficient for variable-length batches.
 */
export class KVCache implements IKVCache {
  private config: KVCacheConfig;
  private cuda: CudaBackend;
  private caches: LayerKVCache[];
  private _seqLen: number = 0;
  private _batchSize: number;

  constructor(config: KVCacheConfig, batchSize: number = 1) {
    this.config = config;
    this._batchSize = batchSize;
    this.cuda = getCudaBackend();
    this.caches = [];

    // Allocate cache for each layer
    for (let i = 0; i < config.numLayers; i++) {
      const shape = [batchSize, config.maxSeqLen, config.numKvHeads, config.headDim];
      this.caches.push({
        k: Tensor.empty(shape, { dtype: config.dtype }),
        v: Tensor.empty(shape, { dtype: config.dtype }),
      });
    }
  }

  /**
   * Current sequence length in the cache.
   */
  get seqLen(): number {
    return this._seqLen;
  }

  /**
   * Batch size.
   */
  get batchSize(): number {
    return this._batchSize;
  }

  /**
   * Maximum sequence length the cache can hold.
   */
  get maxSeqLen(): number {
    return this.config.maxSeqLen;
  }

  /**
   * Get the KV cache for a specific layer.
   */
  getLayer(layerIdx: number): LayerKVCache {
    if (layerIdx < 0 || layerIdx >= this.config.numLayers) {
      throw new Error(`Invalid layer index: ${layerIdx}`);
    }
    return this.caches[layerIdx];
  }

  /**
   * Update the cache with new K, V values for a layer.
   * Returns the updated cache tensors.
   */
  update(
    layerIdx: number,
    kNew: Tensor,  // [batch, seq_new, num_kv_heads, head_dim]
    vNew: Tensor   // [batch, seq_new, num_kv_heads, head_dim]
  ): LayerKVCache {
    const cache = this.getLayer(layerIdx);
    const seqNew = kNew.shape[1];

    if (this._seqLen + seqNew > this.config.maxSeqLen) {
      throw new Error(
        `Cache overflow: ${this._seqLen} + ${seqNew} > ${this.config.maxSeqLen}`
      );
    }

    // Update cache using CUDA kernel (dtype-aware dispatch)
    if (this.config.dtype === DType.BFloat16) {
      this.cuda.kvCacheUpdateBf16(
        cache.k.ptr,
        cache.v.ptr,
        kNew.ptr,
        vNew.ptr,
        this._batchSize,
        this._seqLen,
        seqNew,
        this.config.numKvHeads,
        this.config.headDim,
        this.config.maxSeqLen
      );
    } else {
      this.cuda.kvCacheUpdateF16(
        cache.k.ptr,
        cache.v.ptr,
        kNew.ptr,
        vNew.ptr,
        this._batchSize,
        this._seqLen,
        seqNew,
        this.config.numKvHeads,
        this.config.headDim,
        this.config.maxSeqLen
      );
    }

    // Update seq_len after all layers have been updated
    // (caller should update once after all layers)
    return cache;
  }

  /**
   * Advance the sequence position after updating all layers.
   */
  advanceSeqLen(numTokens: number): void {
    this._seqLen += numTokens;
  }

  /**
   * Reset the cache (start new sequence).
   */
  reset(): void {
    this._seqLen = 0;
    // Note: We don't need to zero the memory, just reset the position
  }

  /**
   * Get a view of the cache up to current seq_len.
   * Useful for attention computation.
   */
  getValidCache(layerIdx: number): LayerKVCache {
    const cache = this.getLayer(layerIdx);
    // Return tensors with virtual shape limited to current seq_len
    // The actual memory is still the full cache, but we track the valid portion
    return {
      k: Tensor.fromPtr(
        cache.k.ptr,
        [this._batchSize, this._seqLen, this.config.numKvHeads, this.config.headDim],
        this.config.dtype
      ),
      v: Tensor.fromPtr(
        cache.v.ptr,
        [this._batchSize, this._seqLen, this.config.numKvHeads, this.config.headDim],
        this.config.dtype
      ),
    };
  }

  /**
   * Free all cache memory.
   */
  dispose(): void {
    for (const cache of this.caches) {
      cache.k.dispose();
      cache.v.dispose();
    }
    this.caches = [];
  }

  /**
   * Get memory usage in bytes.
   */
  memoryUsage(): number {
    if (this.caches.length === 0) return 0;
    return this.caches.length * 2 * this.caches[0].k.nbytes;
  }
}

/**
 * Create a KV cache from model config.
 */
export function createKVCache(
  numLayers: number,
  numKvHeads: number,
  headDim: number,
  maxSeqLen: number,
  batchSize: number = 1,
  dtype: DType = DType.Float16
): KVCache {
  return new KVCache(
    { numLayers, numKvHeads, headDim, maxSeqLen, dtype },
    batchSize
  );
}
