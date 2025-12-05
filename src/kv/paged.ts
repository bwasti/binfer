// Paged KV Cache - Block-based memory management (vLLM-style)
// Enables efficient memory usage for variable-length sequences

import { Tensor, DType } from "../tensor/tensor";
import { getCudaBackend } from "../backend/cuda/bindings";

// Block size in tokens - each block holds this many KV pairs
export const BLOCK_SIZE = 16;

/**
 * PagedKVPool - Contiguous GPU buffer for FA3 paged attention
 *
 * Layout: K and V caches are stored as:
 *   [num_blocks, block_size, num_kv_heads, head_dim]
 *
 * This layout is required by Flash Attention 3's paged attention mode.
 * Each layer has its own K and V pool.
 */
export class PagedKVPool {
  private numBlocks: number;
  private blockSize: number;
  private numKvHeads: number;
  private headDim: number;
  private dtype: DType;
  private device: number;

  // Per-layer K/V caches
  private kCaches: bigint[];  // [layer] -> GPU ptr
  private vCaches: bigint[];  // [layer] -> GPU ptr

  // Block allocation state
  private freeBlocks: Set<number>;
  private allocatedBlocks: Map<number, Set<number>>;  // seqId -> Set<blockId>

  // Bytes per element
  private bytesPerElement: number;
  private blockBytes: number;

  constructor(
    numLayers: number,
    numKvHeads: number,
    headDim: number,
    maxMemoryGB: number,
    dtype: DType = DType.BFloat16,
    device: number = 0,
    quiet: boolean = false
  ) {
    this.blockSize = BLOCK_SIZE;
    this.numKvHeads = numKvHeads;
    this.headDim = headDim;
    this.dtype = dtype;
    this.device = device;
    this.bytesPerElement = dtype === DType.Float16 || dtype === DType.BFloat16 ? 2 : 4;

    // Calculate block size in bytes (K + V for one layer)
    const kvPerBlock = this.blockSize * numKvHeads * headDim * this.bytesPerElement;
    const bytesPerBlockPerLayer = 2 * kvPerBlock;  // K + V
    const bytesPerBlockTotal = bytesPerBlockPerLayer * numLayers;

    // Calculate total blocks that fit in memory
    const maxBytes = maxMemoryGB * 1024 * 1024 * 1024;
    this.numBlocks = Math.floor(maxBytes / bytesPerBlockTotal);
    this.blockBytes = kvPerBlock;  // Bytes for K or V per block per layer

    if (!quiet) {
      console.log(`PagedKVPool[device ${device}]: ${this.numBlocks} blocks of ${BLOCK_SIZE} tokens each`);
      console.log(`  Block size: ${(bytesPerBlockTotal / 1024).toFixed(2)} KB (all layers)`);
      console.log(`  Total KV cache: ${(this.numBlocks * bytesPerBlockTotal / 1024 / 1024 / 1024).toFixed(2)} GB`);
    }

    // Allocate contiguous GPU memory for each layer on the specified device
    const cuda = getCudaBackend();
    cuda.setDevice(device);
    const cacheSize = this.numBlocks * this.blockBytes;

    this.kCaches = [];
    this.vCaches = [];
    for (let layer = 0; layer < numLayers; layer++) {
      this.kCaches.push(cuda.malloc(cacheSize));
      this.vCaches.push(cuda.malloc(cacheSize));
    }

    // Initialize free block pool
    this.freeBlocks = new Set();
    for (let i = 0; i < this.numBlocks; i++) {
      this.freeBlocks.add(i);
    }

    this.allocatedBlocks = new Map();
  }

  /**
   * Get the device this pool is allocated on
   */
  getDevice(): number {
    return this.device;
  }

  /**
   * Get K cache pointer for a layer
   */
  getKCachePtr(layer: number): bigint {
    return this.kCaches[layer];
  }

  /**
   * Get V cache pointer for a layer
   */
  getVCachePtr(layer: number): bigint {
    return this.vCaches[layer];
  }

  /**
   * Get the block size (tokens per block)
   */
  getBlockSize(): number {
    return this.blockSize;
  }

  /**
   * Get total number of blocks
   */
  getTotalBlocks(): number {
    return this.numBlocks;
  }

  /**
   * Get number of free blocks
   */
  getNumFreeBlocks(): number {
    return this.freeBlocks.size;
  }

  /**
   * Allocate blocks for a sequence
   * Returns the physical block IDs, or null if allocation fails
   */
  allocateBlocks(seqId: number, numBlocks: number): number[] | null {
    if (this.freeBlocks.size < numBlocks) {
      return null;  // OOM
    }

    const blockIds: number[] = [];
    const seqBlocks = this.allocatedBlocks.get(seqId) || new Set();

    for (let i = 0; i < numBlocks; i++) {
      const blockId = this.freeBlocks.values().next().value!;
      this.freeBlocks.delete(blockId);
      blockIds.push(blockId);
      seqBlocks.add(blockId);
    }

    this.allocatedBlocks.set(seqId, seqBlocks);
    return blockIds;
  }

  /**
   * Free all blocks for a sequence
   */
  freeSequence(seqId: number): void {
    const seqBlocks = this.allocatedBlocks.get(seqId);
    if (seqBlocks) {
      for (const blockId of seqBlocks) {
        this.freeBlocks.add(blockId);
      }
      this.allocatedBlocks.delete(seqId);
    }
  }

  /**
   * Check if we can allocate more blocks
   */
  canAllocate(numBlocks: number): boolean {
    return this.freeBlocks.size >= numBlocks;
  }

  /**
   * Calculate blocks needed for a given number of tokens
   */
  static blocksForTokens(numTokens: number): number {
    return Math.ceil(numTokens / BLOCK_SIZE);
  }

  /**
   * Dispose all allocated GPU memory
   */
  dispose(): void {
    const cuda = getCudaBackend();
    for (const ptr of this.kCaches) {
      cuda.free(ptr);
    }
    for (const ptr of this.vCaches) {
      cuda.free(ptr);
    }
    this.kCaches = [];
    this.vCaches = [];
    this.freeBlocks.clear();
    this.allocatedBlocks.clear();
  }
}

/**
 * Sequence KV state - tracks a single sequence's blocks and metadata
 */
export class SequenceKVState {
  public readonly seqId: number;
  public blockIds: number[];  // Physical block IDs
  public numTokens: number;   // Current token count

  constructor(seqId: number) {
    this.seqId = seqId;
    this.blockIds = [];
    this.numTokens = 0;
  }

  /**
   * Get block table as Int32Array for CUDA kernel
   */
  getBlockTable(maxBlocksPerSeq: number): Int32Array {
    const table = new Int32Array(maxBlocksPerSeq);
    for (let i = 0; i < this.blockIds.length; i++) {
      table[i] = this.blockIds[i];
    }
    return table;
  }

  /**
   * Get the number of blocks currently allocated
   */
  get numBlocks(): number {
    return this.blockIds.length;
  }

  /**
   * Add new block IDs (after allocation)
   */
  addBlocks(newBlockIds: number[]): void {
    this.blockIds.push(...newBlockIds);
  }
}

/**
 * Physical block of KV cache memory.
 * Each block holds BLOCK_SIZE tokens worth of K and V tensors.
 */
export interface PhysicalBlock {
  blockId: number;
  k: Tensor;  // [block_size, num_kv_heads, head_dim]
  v: Tensor;  // [block_size, num_kv_heads, head_dim]
  refCount: number;  // For copy-on-write
}

/**
 * Logical block mapping for a sequence.
 * Maps logical block indices to physical block IDs.
 */
export interface BlockTable {
  sequenceId: number;
  physicalBlockIds: number[];  // Logical index -> Physical block ID
  numTokens: number;           // Actual tokens in this sequence
}

/**
 * Block allocator manages physical GPU memory blocks.
 */
export class BlockAllocator {
  private numLayers: number;
  private numKvHeads: number;
  private headDim: number;
  private dtype: DType;

  private totalBlocks: number;
  private freeBlocks: Set<number>;
  private blocks: Map<number, PhysicalBlock[]>;  // blockId -> blocks per layer

  constructor(
    numLayers: number,
    numKvHeads: number,
    headDim: number,
    maxMemoryGB: number,
    dtype: DType = DType.Float16
  ) {
    this.numLayers = numLayers;
    this.numKvHeads = numKvHeads;
    this.headDim = headDim;
    this.dtype = dtype;

    // Calculate block size in bytes
    const bytesPerElement = dtype === DType.Float16 ? 2 : 4;
    const kvPerBlock = BLOCK_SIZE * numKvHeads * headDim * bytesPerElement;
    const bytesPerBlock = 2 * kvPerBlock * numLayers;  // K + V for all layers

    // Calculate total blocks that fit in memory
    const maxBytes = maxMemoryGB * 1024 * 1024 * 1024;
    this.totalBlocks = Math.floor(maxBytes / bytesPerBlock);

    console.log(`BlockAllocator: ${this.totalBlocks} blocks of ${BLOCK_SIZE} tokens each`);
    console.log(`  Memory per block: ${(bytesPerBlock / 1024 / 1024).toFixed(2)} MB`);
    console.log(`  Total KV cache: ${(this.totalBlocks * bytesPerBlock / 1024 / 1024 / 1024).toFixed(2)} GB`);

    // Initialize free block pool
    this.freeBlocks = new Set();
    for (let i = 0; i < this.totalBlocks; i++) {
      this.freeBlocks.add(i);
    }

    this.blocks = new Map();
  }

  /**
   * Allocate a new physical block.
   * Returns null if no blocks available.
   */
  allocate(): number | null {
    if (this.freeBlocks.size === 0) {
      return null;
    }

    const blockId = this.freeBlocks.values().next().value;
    this.freeBlocks.delete(blockId);

    // Lazily allocate GPU memory for this block
    if (!this.blocks.has(blockId)) {
      const layerBlocks: PhysicalBlock[] = [];
      for (let layer = 0; layer < this.numLayers; layer++) {
        layerBlocks.push({
          blockId,
          k: Tensor.empty([BLOCK_SIZE, this.numKvHeads, this.headDim], { dtype: this.dtype }),
          v: Tensor.empty([BLOCK_SIZE, this.numKvHeads, this.headDim], { dtype: this.dtype }),
          refCount: 1,
        });
      }
      this.blocks.set(blockId, layerBlocks);
    } else {
      // Reset ref count
      for (const block of this.blocks.get(blockId)!) {
        block.refCount = 1;
      }
    }

    return blockId;
  }

  /**
   * Free a physical block.
   */
  free(blockId: number): void {
    const layerBlocks = this.blocks.get(blockId);
    if (layerBlocks) {
      for (const block of layerBlocks) {
        block.refCount--;
        if (block.refCount <= 0) {
          this.freeBlocks.add(blockId);
        }
      }
    }
  }

  /**
   * Get physical block for a layer.
   */
  getBlock(blockId: number, layer: number): PhysicalBlock | null {
    const layerBlocks = this.blocks.get(blockId);
    if (!layerBlocks) return null;
    return layerBlocks[layer];
  }

  /**
   * Increment ref count (for copy-on-write).
   */
  incrementRef(blockId: number): void {
    const layerBlocks = this.blocks.get(blockId);
    if (layerBlocks) {
      for (const block of layerBlocks) {
        block.refCount++;
      }
    }
  }

  /**
   * Get number of free blocks.
   */
  getNumFreeBlocks(): number {
    return this.freeBlocks.size;
  }

  /**
   * Get total number of blocks.
   */
  getTotalBlocks(): number {
    return this.totalBlocks;
  }

  /**
   * Dispose all allocated memory.
   */
  dispose(): void {
    for (const layerBlocks of this.blocks.values()) {
      for (const block of layerBlocks) {
        block.k.dispose();
        block.v.dispose();
      }
    }
    this.blocks.clear();
    this.freeBlocks.clear();
  }
}

/**
 * Paged KV cache for a single sequence.
 */
export class PagedKVCache {
  private allocator: BlockAllocator;
  private blockTable: BlockTable;
  private numLayers: number;

  constructor(
    allocator: BlockAllocator,
    sequenceId: number,
    numLayers: number
  ) {
    this.allocator = allocator;
    this.numLayers = numLayers;
    this.blockTable = {
      sequenceId,
      physicalBlockIds: [],
      numTokens: 0,
    };
  }

  /**
   * Get sequence ID.
   */
  get sequenceId(): number {
    return this.blockTable.sequenceId;
  }

  /**
   * Get current sequence length.
   */
  get seqLen(): number {
    return this.blockTable.numTokens;
  }

  /**
   * Get number of allocated blocks.
   */
  get numBlocks(): number {
    return this.blockTable.physicalBlockIds.length;
  }

  /**
   * Allocate blocks for new tokens.
   * Returns false if allocation fails (OOM).
   */
  allocateTokens(numNewTokens: number): boolean {
    const newTotal = this.blockTable.numTokens + numNewTokens;
    const blocksNeeded = Math.ceil(newTotal / BLOCK_SIZE);

    while (this.blockTable.physicalBlockIds.length < blocksNeeded) {
      const blockId = this.allocator.allocate();
      if (blockId === null) {
        return false;  // OOM
      }
      this.blockTable.physicalBlockIds.push(blockId);
    }

    this.blockTable.numTokens = newTotal;
    return true;
  }

  /**
   * Get the block table as an Int32Array for passing to CUDA kernel.
   * Padded to maxBlocks length.
   */
  getBlockTableArray(maxBlocks: number): Int32Array {
    const arr = new Int32Array(maxBlocks);
    for (let i = 0; i < this.blockTable.physicalBlockIds.length; i++) {
      arr[i] = this.blockTable.physicalBlockIds[i];
    }
    return arr;
  }

  /**
   * Get physical block for a layer and logical block index.
   */
  getBlock(layer: number, logicalBlockIdx: number): PhysicalBlock | null {
    if (logicalBlockIdx >= this.blockTable.physicalBlockIds.length) {
      return null;
    }
    const physicalId = this.blockTable.physicalBlockIds[logicalBlockIdx];
    return this.allocator.getBlock(physicalId, layer);
  }

  /**
   * Get the slot index within a block for a token position.
   */
  static getSlotIndex(tokenPos: number): number {
    return tokenPos % BLOCK_SIZE;
  }

  /**
   * Get the logical block index for a token position.
   */
  static getBlockIndex(tokenPos: number): number {
    return Math.floor(tokenPos / BLOCK_SIZE);
  }

  /**
   * Free all blocks.
   */
  dispose(): void {
    for (const blockId of this.blockTable.physicalBlockIds) {
      this.allocator.free(blockId);
    }
    this.blockTable.physicalBlockIds = [];
    this.blockTable.numTokens = 0;
  }
}

/**
 * Manager for multiple paged KV caches (one per sequence in a batch).
 */
export class PagedKVCacheManager {
  private allocator: BlockAllocator;
  private caches: Map<number, PagedKVCache>;
  private numLayers: number;
  private nextSequenceId: number;

  constructor(
    numLayers: number,
    numKvHeads: number,
    headDim: number,
    maxMemoryGB: number,
    dtype: DType = DType.Float16
  ) {
    this.allocator = new BlockAllocator(
      numLayers,
      numKvHeads,
      headDim,
      maxMemoryGB,
      dtype
    );
    this.caches = new Map();
    this.numLayers = numLayers;
    this.nextSequenceId = 0;
  }

  /**
   * Create a new paged KV cache for a sequence.
   */
  createCache(): PagedKVCache {
    const seqId = this.nextSequenceId++;
    const cache = new PagedKVCache(this.allocator, seqId, this.numLayers);
    this.caches.set(seqId, cache);
    return cache;
  }

  /**
   * Get cache by sequence ID.
   */
  getCache(sequenceId: number): PagedKVCache | undefined {
    return this.caches.get(sequenceId);
  }

  /**
   * Remove a cache.
   */
  removeCache(sequenceId: number): void {
    const cache = this.caches.get(sequenceId);
    if (cache) {
      cache.dispose();
      this.caches.delete(sequenceId);
    }
  }

  /**
   * Get number of free blocks.
   */
  getNumFreeBlocks(): number {
    return this.allocator.getNumFreeBlocks();
  }

  /**
   * Check if we can allocate more tokens.
   */
  canAllocate(numTokens: number): boolean {
    const blocksNeeded = Math.ceil(numTokens / BLOCK_SIZE);
    return this.allocator.getNumFreeBlocks() >= blocksNeeded;
  }

  /**
   * Get memory usage stats.
   */
  getStats(): { totalBlocks: number; freeBlocks: number; usedBlocks: number } {
    const total = this.allocator.getTotalBlocks();
    const free = this.allocator.getNumFreeBlocks();
    return {
      totalBlocks: total,
      freeBlocks: free,
      usedBlocks: total - free,
    };
  }

  /**
   * Dispose all caches and free memory.
   */
  dispose(): void {
    for (const cache of this.caches.values()) {
      cache.dispose();
    }
    this.caches.clear();
    this.allocator.dispose();
  }
}

/**
 * Create a paged KV cache manager.
 */
export function createPagedKVCacheManager(
  numLayers: number,
  numKvHeads: number,
  headDim: number,
  maxMemoryGB: number = 10,
  dtype: DType = DType.Float16
): PagedKVCacheManager {
  return new PagedKVCacheManager(numLayers, numKvHeads, headDim, maxMemoryGB, dtype);
}
