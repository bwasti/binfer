// Device Context - Abstraction for per-device state management
// Supports both single-GPU and multi-GPU tensor parallelism

import { Tensor, DType } from "../tensor/tensor";
import { IKVCache, KVCacheConfig, LayerKVCache, createKVCache } from "../kv/manager";
import { getCudaBackend, CudaBackend } from "../backend/cuda/bindings";
import { LlamaConfig } from "../model/config";
import { MemoryPool, getDeviceMemoryPool } from "../tensor/memory";

/**
 * Per-device state for inference.
 * Manages weights, KV cache, RoPE cache, and memory allocation for a single GPU.
 */
export interface DeviceContext {
  readonly device: number;
  readonly weights: Map<string, { ptr: bigint; info: any }>;
  readonly dtype: DType;
  kvCache: IKVCache | null;
  ropeCache: { cos: Tensor; sin: Tensor } | null;

  alloc(shape: number[], dtype?: DType): Tensor;
  free(tensor: Tensor): void;
  getWeight(name: string): Tensor;
  hasWeight(name: string): boolean;
  setActive(): void;
}

/**
 * Device context for single-GPU inference.
 * Uses the global memory pool via Tensor.empty().
 */
export class SingleDeviceContext implements DeviceContext {
  readonly device: number = 0;
  readonly weights: Map<string, { ptr: bigint; info: any }>;
  kvCache: IKVCache | null = null;
  ropeCache: { cos: Tensor; sin: Tensor } | null = null;

  private cuda: CudaBackend;
  private _dtype: DType;

  constructor(weights: Map<string, { ptr: bigint; info: any }>, dtype: DType = DType.Float16) {
    this.weights = weights;
    this.cuda = getCudaBackend();
    this._dtype = dtype;
  }

  get dtype(): DType {
    return this._dtype;
  }

  setActive(): void {
    // No-op for single GPU
  }

  alloc(shape: number[], dtype?: DType): Tensor {
    return Tensor.empty(shape, { dtype: dtype ?? this._dtype });
  }

  free(tensor: Tensor): void {
    tensor.dispose();
  }

  getWeight(name: string): Tensor {
    const w = this.weights.get(name);
    if (!w) {
      throw new Error(`Weight not found: ${name}`);
    }
    // Use the dtype stored with the weight (important for uint8 MXFP4 blocks/scales)
    return Tensor.fromPtr(w.ptr, w.info.shape, w.info.dtype ?? this._dtype);
  }

  hasWeight(name: string): boolean {
    return this.weights.has(name);
  }
}

/**
 * Device context for multi-GPU tensor parallelism.
 * Uses per-device memory pool for fast allocation/deallocation.
 */
export class MultiDeviceContext implements DeviceContext {
  readonly device: number;
  readonly weights: Map<string, { ptr: bigint; info: any }>;
  kvCache: IKVCache | null = null;
  ropeCache: { cos: Tensor; sin: Tensor } | null = null;

  private cuda: CudaBackend;
  private _dtype: DType;
  private pool: MemoryPool;

  constructor(
    device: number,
    weights: Map<string, { ptr: bigint; info: any }>,
    cuda: CudaBackend,
    dtype: DType = DType.Float16
  ) {
    this.device = device;
    this.weights = weights;
    this.cuda = cuda;
    this._dtype = dtype;
    // Get or create per-device memory pool
    this.pool = getDeviceMemoryPool(device, cuda);
  }

  get dtype(): DType {
    return this._dtype;
  }

  setActive(): void {
    this.cuda.setDevice(this.device);
  }

  alloc(shape: number[], dtype?: DType): Tensor {
    const useType = dtype ?? this._dtype;
    const numel = shape.reduce((a, b) => a * b, 1);
    const elementSize = useType === DType.Float16 || useType === DType.BFloat16 ? 2 : 4;
    const size = numel * elementSize;
    // Use memory pool instead of direct malloc
    const ptr = this.pool.alloc(size);
    return Tensor.fromPtr(ptr, shape, useType, this.device);
  }

  free(tensor: Tensor): void {
    // Return to pool instead of direct free
    this.pool.free(tensor.ptr);
  }

  getWeight(name: string): Tensor {
    const w = this.weights.get(name);
    if (!w) {
      throw new Error(`Weight not found on device ${this.device}: ${name}`);
    }
    // Use the dtype stored with the weight (important for uint8 MXFP4 blocks/scales)
    return Tensor.fromPtr(w.ptr, w.info.shape, w.info.dtype ?? this._dtype, this.device);
  }

  hasWeight(name: string): boolean {
    return this.weights.has(name);
  }
}

/**
 * Device-local KV cache that allocates tensors on a specific GPU.
 */
export class DeviceLocalKVCache implements IKVCache {
  private config: KVCacheConfig;
  private cuda: CudaBackend;
  private caches: LayerKVCache[];
  private _seqLen: number = 0;
  private _batchSize: number;

  constructor(
    config: KVCacheConfig,
    batchSize: number,
    caches: LayerKVCache[],
    cuda: CudaBackend
  ) {
    this.config = config;
    this._batchSize = batchSize;
    this.caches = caches;
    this.cuda = cuda;
  }

  get seqLen(): number {
    return this._seqLen;
  }

  get batchSize(): number {
    return this._batchSize;
  }

  get maxSeqLen(): number {
    return this.config.maxSeqLen;
  }

  getLayer(layerIdx: number): LayerKVCache {
    if (layerIdx < 0 || layerIdx >= this.config.numLayers) {
      throw new Error(`Invalid layer index: ${layerIdx}`);
    }
    return this.caches[layerIdx];
  }

  update(layerIdx: number, kNew: Tensor, vNew: Tensor): LayerKVCache {
    const cache = this.getLayer(layerIdx);
    const seqNew = kNew.shape[1];

    if (this._seqLen + seqNew > this.config.maxSeqLen) {
      throw new Error(
        `Cache overflow: ${this._seqLen} + ${seqNew} > ${this.config.maxSeqLen}`
      );
    }

    // dtype-aware dispatch
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

    return cache;
  }

  advanceSeqLen(numTokens: number): void {
    this._seqLen += numTokens;
  }

  reset(): void {
    this._seqLen = 0;
  }

  getValidCache(layerIdx: number): LayerKVCache {
    const cache = this.getLayer(layerIdx);
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

  dispose(): void {
    for (const cache of this.caches) {
      this.cuda.free(cache.k.ptr);
      this.cuda.free(cache.v.ptr);
    }
    this.caches = [];
  }

  memoryUsage(): number {
    if (this.caches.length === 0) return 0;
    return this.caches.length * 2 * this.caches[0].k.nbytes;
  }
}

/**
 * Create a device-local KV cache with tensors allocated on the specified device.
 */
export function createDeviceLocalKVCache(
  numLayers: number,
  numKvHeads: number,
  headDim: number,
  maxSeqLen: number,
  batchSize: number,
  device: number,
  cuda: CudaBackend,
  dtype: DType = DType.Float16
): IKVCache {
  const shape = [batchSize, maxSeqLen, numKvHeads, headDim];
  const numel = shape.reduce((a, b) => a * b, 1);
  const size = numel * 2; // fp16/bf16 = 2 bytes

  const caches: LayerKVCache[] = [];

  for (let i = 0; i < numLayers; i++) {
    const kPtr = cuda.malloc(size);
    const vPtr = cuda.malloc(size);
    caches.push({
      k: Tensor.fromPtr(kPtr, shape, dtype, device),
      v: Tensor.fromPtr(vPtr, shape, dtype, device),
    });
  }

  return new DeviceLocalKVCache(
    { numLayers, numKvHeads, headDim, maxSeqLen, dtype },
    batchSize,
    caches,
    cuda
  );
}
