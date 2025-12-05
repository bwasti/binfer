// GPU Memory management for tensor allocation

import { CudaBackend, getCudaBackend } from "../backend/cuda/bindings";

export interface MemoryBlock {
  ptr: bigint;
  size: number;
  inUse: boolean;
}

/**
 * Simple memory pool for GPU tensors.
 * Reduces allocation overhead by reusing memory blocks.
 */
export class MemoryPool {
  private cuda: CudaBackend;
  private freeBlocks: Map<number, MemoryBlock[]>; // size -> list of free blocks
  private allocatedBlocks: Map<bigint, MemoryBlock>; // ptr -> block info
  private totalAllocated: number = 0;
  private device: number;  // Device this pool is for

  constructor(cuda?: CudaBackend, device: number = 0) {
    this.cuda = cuda ?? getCudaBackend();
    this.freeBlocks = new Map();
    this.allocatedBlocks = new Map();
    this.device = device;
  }

  /**
   * Allocate memory of the given size.
   * Tries to reuse an existing free block first.
   */
  alloc(size: number): bigint {
    // Round up to 256-byte alignment
    const alignedSize = Math.ceil(size / 256) * 256;

    // Try to find a free block of the right size
    const blocks = this.freeBlocks.get(alignedSize);
    if (blocks && blocks.length > 0) {
      const block = blocks.pop()!;
      block.inUse = true;
      this.allocatedBlocks.set(block.ptr, block);
      return block.ptr;
    }

    // Ensure we allocate on the correct device
    this.cuda.setDevice(this.device);

    // Allocate new block
    const ptr = this.cuda.malloc(alignedSize);
    const block: MemoryBlock = {
      ptr,
      size: alignedSize,
      inUse: true,
    };
    this.allocatedBlocks.set(ptr, block);
    this.totalAllocated += alignedSize;

    return ptr;
  }

  /**
   * Free memory back to the pool.
   * The memory is not immediately released to CUDA, but marked for reuse.
   */
  free(ptr: bigint): void {
    const block = this.allocatedBlocks.get(ptr);
    if (!block) {
      console.warn("Attempted to free unknown pointer");
      return;
    }

    block.inUse = false;
    this.allocatedBlocks.delete(ptr);

    // Add to free list
    let blocks = this.freeBlocks.get(block.size);
    if (!blocks) {
      blocks = [];
      this.freeBlocks.set(block.size, blocks);
    }
    blocks.push(block);
  }

  /**
   * Release all memory back to CUDA.
   */
  releaseAll(): void {
    // Free all blocks (both in-use and free)
    for (const [, block] of this.allocatedBlocks) {
      this.cuda.free(block.ptr);
    }
    for (const [, blocks] of this.freeBlocks) {
      for (const block of blocks) {
        this.cuda.free(block.ptr);
      }
    }

    this.allocatedBlocks.clear();
    this.freeBlocks.clear();
    this.totalAllocated = 0;
  }

  /**
   * Get memory usage statistics.
   */
  stats(): { totalAllocated: number; inUse: number; pooled: number } {
    let inUse = 0;
    for (const [, block] of this.allocatedBlocks) {
      inUse += block.size;
    }

    let pooled = 0;
    for (const [, blocks] of this.freeBlocks) {
      for (const block of blocks) {
        pooled += block.size;
      }
    }

    return {
      totalAllocated: this.totalAllocated,
      inUse,
      pooled,
    };
  }
}

// Global memory pool
let _globalPool: MemoryPool | null = null;

export function getGlobalMemoryPool(): MemoryPool {
  if (!_globalPool) {
    _globalPool = new MemoryPool();
  }
  return _globalPool;
}

// Per-device memory pools for multi-GPU
const _devicePools: Map<number, MemoryPool> = new Map();

export function getDeviceMemoryPool(device: number, cuda?: CudaBackend): MemoryPool {
  let pool = _devicePools.get(device);
  if (!pool) {
    pool = new MemoryPool(cuda, device);
    _devicePools.set(device, pool);
  }
  return pool;
}

export function releaseAllDevicePools(): void {
  for (const pool of _devicePools.values()) {
    pool.releaseAll();
  }
  _devicePools.clear();
}
