// Pre-sharded Weight Cache
// Caches sharded weights to disk for faster multi-GPU loading

import { existsSync, mkdirSync, readdirSync, unlinkSync, statSync, openSync, writeSync, closeSync } from "fs";
import { join } from "path";
import { getCudaBackend, CudaBackend } from "../backend/cuda/bindings";
import { DType, dtypeFromString, DTYPE_SIZES } from "../tensor/tensor";
import { LlamaConfig } from "../model/config";

export interface ShardCacheMetadata {
  modelPath: string;
  modelHash: string;  // Hash of config + weight index for cache invalidation
  worldSize: number;
  dtype: string;
  createdAt: number;
  tensors: Record<string, {
    shape: number[];
    dtype: string;
    offset: number;
    size: number;
  }>;
}

export interface ShardCacheInfo {
  cacheDir: string;
  metadata: ShardCacheMetadata;
  valid: boolean;
}

export interface ShardCacheLogger {
  log: (msg: string) => void;
  progress: (msg: string) => void;  // For status updates that overwrite
}

export interface CacheLoadResult {
  weights: Map<number, Map<string, { ptr: bigint; info: any }>>;
  // Pool pointers for efficient cleanup (one per device) - only set for pooled allocation
  poolPtrs?: bigint[];
}

/**
 * Manages pre-sharded weight cache for tensor parallelism.
 */
export class ShardCache {
  private cuda: CudaBackend;
  private cacheBaseDir: string;
  private logger: ShardCacheLogger;

  constructor(cacheDir?: string, logger?: ShardCacheLogger) {
    this.cuda = getCudaBackend();
    // Default cache location
    this.cacheBaseDir = cacheDir || join(process.env.HOME || "/tmp", ".cache", "binfer", "shards");
    // Default logger uses console
    this.logger = logger ?? {
      log: (msg: string) => console.log(msg),
      progress: (msg: string) => process.stdout.write(`\r${msg}`),
    };
  }

  setLogger(logger: ShardCacheLogger): void {
    this.logger = logger;
  }

  /**
   * Get the cache directory for a specific model and world size.
   */
  getCacheDir(modelPath: string, worldSize: number): string {
    // Use model path hash + world size for cache key
    const pathHash = this.hashString(modelPath);
    return join(this.cacheBaseDir, `${pathHash}_tp${worldSize}`);
  }

  /**
   * Check if a valid cache exists for the given model configuration.
   * Fast path - just check files exist, trust the cache.
   */
  async checkCache(
    modelPath: string,
    worldSize: number,
    _config: LlamaConfig
  ): Promise<ShardCacheInfo | null> {
    const cacheDir = this.getCacheDir(modelPath, worldSize);
    const metadataPath = join(cacheDir, "metadata.json");

    if (!existsSync(metadataPath)) {
      return null;
    }

    try {
      const metadata: ShardCacheMetadata = JSON.parse(
        await Bun.file(metadataPath).text()
      );

      // Quick check that shard files exist
      for (let device = 0; device < worldSize; device++) {
        const shardPath = join(cacheDir, `shard_${device}.bin`);
        if (!existsSync(shardPath)) {
          return null;
        }
      }

      return { cacheDir, metadata, valid: true };
    } catch (e) {
      return null;
    }
  }

  /**
   * Load pre-sharded weights from cache.
   * Uses GPU Direct Storage (GDS) if available for faster DMA,
   * otherwise falls back to streaming reads with pinned memory.
   */
  async loadFromCache(
    cacheInfo: ShardCacheInfo,
    worldSize: number
  ): Promise<CacheLoadResult> {
    // Try GDS first
    const gdsAvailable = this.cuda.gdsAvailable();
    if (gdsAvailable) {
      this.logger.progress(`Loading with GDS...`);
      const gdsResult = await this.loadFromCacheGDS(cacheInfo, worldSize);
      if (gdsResult) {
        return { weights: gdsResult };  // GDS uses individual mallocs, no pooling
      }
      // GDS failed, silently fall back to streaming
    }

    // Fallback: streaming reads with pinned memory (uses pooled allocation)
    return this.loadFromCacheStreaming(cacheInfo, worldSize);
  }

  /**
   * Load using GPU Direct Storage (GDS) for direct file-to-GPU DMA.
   * Returns null if GDS cannot be used for this path.
   */
  private async loadFromCacheGDS(
    cacheInfo: ShardCacheInfo,
    worldSize: number
  ): Promise<Map<number, Map<string, { ptr: bigint; info: any }>> | null> {
    const { cacheDir, metadata } = cacheInfo;

    // Initialize GDS
    if (!this.cuda.gdsInit()) {
      return null;
    }

    const startTime = Date.now();
    let totalBytes = 0;
    let tensorCount = 0;
    const totalTensors = Object.keys(metadata.tensors).length;

    const perDeviceWeights: Map<number, Map<string, { ptr: bigint; info: any }>> = new Map();

    try {
      for (let device = 0; device < worldSize; device++) {
        perDeviceWeights.set(device, new Map());
        this.cuda.setDevice(device);

        const shardPath = join(cacheDir, `shard_${device}.bin`);

        // Open file for GDS
        const gdsHandle = this.cuda.gdsOpen(shardPath);
        if (!gdsHandle) {
          // GDS can't open this file (e.g., filesystem not supported)
          this.cuda.gdsClose();
          return null;
        }

        try {
          // Load each tensor for this device
          for (const [name, tensorInfo] of Object.entries(metadata.tensors)) {
            const devicePrefix = `device_${device}_`;
            if (!name.startsWith(devicePrefix)) continue;

            const realName = name.slice(devicePrefix.length);

            // Allocate GPU memory
            const gpuPtr = this.cuda.malloc(tensorInfo.size);

            // Register buffer for GDS
            if (!this.cuda.gdsRegisterBuffer(gpuPtr, tensorInfo.size)) {
              // Registration failed, fall back
              this.cuda.free(gpuPtr);
              this.cuda.gdsCloseFile(gdsHandle);
              this.cuda.gdsClose();
              return null;
            }

            // Read directly from file to GPU
            const bytesRead = this.cuda.gdsRead(
              gdsHandle,
              gpuPtr,
              tensorInfo.size,
              tensorInfo.offset,
              0
            );

            // Deregister buffer (still usable, just not for GDS)
            this.cuda.gdsDeregisterBuffer(gpuPtr);

            if (bytesRead !== tensorInfo.size) {
              // Read failed
              this.cuda.free(gpuPtr);
              this.cuda.gdsCloseFile(gdsHandle);
              this.cuda.gdsClose();
              return null;
            }

            const dtype = dtypeFromString(tensorInfo.dtype);
            perDeviceWeights.get(device)!.set(realName, {
              ptr: gpuPtr,
              info: { shape: tensorInfo.shape, dtype },
            });

            totalBytes += tensorInfo.size;
            tensorCount++;

            // Update progress and yield to event loop for spinner animation
            if (tensorCount % 20 === 0 || tensorCount === totalTensors) {
              const elapsedSec = (Date.now() - startTime) / 1000;
              const throughputGBps = elapsedSec > 0 ? (totalBytes / 1e9) / elapsedSec : 0;
              const progress = ((tensorCount / totalTensors) * 100).toFixed(1);
              this.logger.progress(
                `Loading cache (GDS): ${progress}% (${tensorCount}/${totalTensors} tensors) | ${throughputGBps.toFixed(2)} GB/s`
              );
              // Yield to event loop so spinner can animate
              await new Promise(r => setImmediate(r));
            }
          }

          this.cuda.gdsCloseFile(gdsHandle);
        } catch (e) {
          this.cuda.gdsCloseFile(gdsHandle);
          throw e;
        }

        this.logger.progress(`Syncing GPU ${device + 1}/${worldSize}...`);
        this.cuda.synchronize();
      }

      this.cuda.gdsClose();
      this.cuda.setDevice(0);

      const elapsedSec = (Date.now() - startTime) / 1000;
      const throughputGBps = (totalBytes / 1e9) / elapsedSec;
      this.logger.log(
        `Loaded cache (GDS): ${(totalBytes / 1e9).toFixed(2)} GB in ${elapsedSec.toFixed(1)}s (${throughputGBps.toFixed(2)} GB/s)`
      );

      return perDeviceWeights;
    } catch (e) {
      this.cuda.gdsClose();
      this.cuda.setDevice(0);
      return null;
    }
  }

  /**
   * Helper to start a batch of reads across all devices
   */
  private startBatchRead(
    round: number,
    batchSize: number,
    worldSize: number,
    deviceTensorLists: Array<Array<[string, { offset: number; size: number; shape: number[]; dtype: string }]>>,
    shardFiles: any[]
  ): Array<{
    device: number;
    name: string;
    info: { offset: number; size: number; shape: number[]; dtype: string };
    promise: Promise<ArrayBuffer>;
  }> {
    const readPromises: Array<{
      device: number;
      name: string;
      info: { offset: number; size: number; shape: number[]; dtype: string };
      promise: Promise<ArrayBuffer>;
    }> = [];

    for (let device = 0; device < worldSize; device++) {
      const tensors = deviceTensorLists[device];
      for (let i = round; i < Math.min(round + batchSize, tensors.length); i++) {
        const [name, info] = tensors[i];
        readPromises.push({
          device,
          name,
          info,
          promise: shardFiles[device].slice(info.offset, info.offset + info.size).arrayBuffer(),
        });
      }
    }

    return readPromises;
  }

  /**
   * Load using parallel slice reads with pinned memory.
   * Uses pooled allocation and stream-based double buffering for maximum throughput.
   */
  private async loadFromCacheStreaming(
    cacheInfo: ShardCacheInfo,
    worldSize: number
  ): Promise<CacheLoadResult> {
    const { cacheDir, metadata } = cacheInfo;
    const perDeviceWeights: Map<number, Map<string, { ptr: bigint; info: any }>> = new Map();

    this.logger.progress(`Allocating GPU memory...`);
    await new Promise(r => setImmediate(r));  // Allow spinner to update

    const startTime = Date.now();
    let totalBytes = 0;

    // Calculate total size per device for pooled allocation
    const deviceSizes: number[] = new Array(worldSize).fill(0);
    const deviceTensorLists: Array<Array<[string, { offset: number; size: number; shape: number[]; dtype: string }]>> = [];

    for (let device = 0; device < worldSize; device++) {
      const deviceTensors: Array<[string, { offset: number; size: number; shape: number[]; dtype: string }]> = [];
      for (const [name, tensorInfo] of Object.entries(metadata.tensors)) {
        const devicePrefix = `device_${device}_`;
        if (name.startsWith(devicePrefix)) {
          deviceTensors.push([name, tensorInfo as any]);
          deviceSizes[device] += tensorInfo.size;
        }
      }
      deviceTensors.sort((a, b) => a[1].offset - b[1].offset);
      deviceTensorLists.push(deviceTensors);
      perDeviceWeights.set(device, new Map());
    }

    // Allocate one big pool per device + track offset for carving out tensors
    const poolPtrs: bigint[] = [];
    const poolOffsets: number[] = new Array(worldSize).fill(0);
    for (let device = 0; device < worldSize; device++) {
      this.cuda.setDevice(device);
      poolPtrs.push(this.cuda.malloc(deviceSizes[device]));
    }

    // Allocate double-buffered pinned memory and streams per device
    const maxTensorSize = Math.max(...Object.values(metadata.tensors).map(t => t.size));
    const pinnedPtrs: Array<[bigint, bigint]> = [];
    const streams: Array<[bigint, bigint]> = [];

    for (let device = 0; device < worldSize; device++) {
      this.cuda.setDevice(device);
      pinnedPtrs.push([
        this.cuda.mallocHost(maxTensorSize),
        this.cuda.mallocHost(maxTensorSize),
      ]);
      streams.push([
        this.cuda.streamCreate(),
        this.cuda.streamCreate(),
      ]);
    }

    // Track which buffer/stream is in-flight per device
    const bufferIndex: number[] = new Array(worldSize).fill(0);

    // Open all shard files
    const shardFiles = [];
    for (let device = 0; device < worldSize; device++) {
      shardFiles.push(Bun.file(join(cacheDir, `shard_${device}.bin`)));
    }

    // Process all devices in parallel using round-robin tensor loading
    const maxTensorsPerDevice = Math.max(...deviceTensorLists.map(l => l.length));
    const BATCH_SIZE = 8;

    // Start first batch read
    let currentReadPromises = this.startBatchRead(0, BATCH_SIZE, worldSize, deviceTensorLists, shardFiles);

    for (let round = 0; round < maxTensorsPerDevice; round += BATCH_SIZE) {
      // Wait for current batch reads
      const readResults = await Promise.all(currentReadPromises.map(r => r.promise));

      // Start NEXT batch read immediately (pipeline I/O with GPU transfer)
      const nextRound = round + BATCH_SIZE;
      let nextReadPromises: typeof currentReadPromises = [];
      if (nextRound < maxTensorsPerDevice) {
        nextReadPromises = this.startBatchRead(nextRound, BATCH_SIZE, worldSize, deviceTensorLists, shardFiles);
      }

      // Process current batch results - group by device for GPU transfers
      const resultsByDevice: Map<number, Array<{ name: string; info: any; data: ArrayBuffer }>> = new Map();
      for (let i = 0; i < currentReadPromises.length; i++) {
        const { device, name, info } = currentReadPromises[i];
        if (!resultsByDevice.has(device)) {
          resultsByDevice.set(device, []);
        }
        resultsByDevice.get(device)!.push({ name, info, data: readResults[i] });
      }

      // Transfer to each GPU using stream-based double buffering with pooled allocation
      for (const [device, results] of resultsByDevice) {
        this.cuda.setDevice(device);
        const [bufferA, bufferB] = pinnedPtrs[device];
        const [streamA, streamB] = streams[device];

        for (const { name, info, data } of results) {
          const realName = name.slice(`device_${device}_`.length);
          const tensorData = new Uint8Array(data);

          // Get current buffer/stream and flip for next iteration
          const currentIdx = bufferIndex[device];
          const currentBuffer = currentIdx === 0 ? bufferA : bufferB;
          const currentStream = currentIdx === 0 ? streamA : streamB;
          bufferIndex[device] = 1 - currentIdx;

          // Wait for THIS stream's previous transfer to complete (not the whole device)
          this.cuda.streamSynchronize(currentStream);

          // Copy to pinned memory
          this.cuda.memcpyHostToPinned(currentBuffer, tensorData, info.size);

          // Carve out GPU pointer from pool
          const gpuPtr = poolPtrs[device] + BigInt(poolOffsets[device]);
          poolOffsets[device] += info.size;

          // Async H2D on this stream
          this.cuda.memcpyH2DAsyncStream(gpuPtr, currentBuffer, info.size, currentStream);

          const dtype = dtypeFromString(info.dtype);
          perDeviceWeights.get(device)!.set(realName, {
            ptr: gpuPtr,
            info: { shape: info.shape, dtype },
          });

          totalBytes += info.size;
        }
      }

      // Update progress
      const elapsedSec = (Date.now() - startTime) / 1000;
      const throughputGBps = elapsedSec > 0 ? (totalBytes / 1e9) / elapsedSec : 0;
      const totalSize = Object.values(metadata.tensors).reduce((sum, t) => sum + t.size, 0);
      const progress = ((totalBytes / totalSize) * 100).toFixed(1);
      this.logger.progress(
        `Loading cache: ${progress}% | ${throughputGBps.toFixed(2)} GB/s`
      );
      await new Promise(r => setImmediate(r));

      // Move to next batch
      currentReadPromises = nextReadPromises;
    }

    // Final sync on all streams
    this.logger.progress(`Syncing GPU transfers...`);
    await new Promise(r => setImmediate(r));
    for (let device = 0; device < worldSize; device++) {
      this.cuda.setDevice(device);
      const [streamA, streamB] = streams[device];
      this.cuda.streamSynchronize(streamA);
      this.cuda.streamSynchronize(streamB);
    }

    // Free pinned memory and streams
    for (let device = 0; device < worldSize; device++) {
      this.cuda.setDevice(device);
      const [bufferA, bufferB] = pinnedPtrs[device];
      const [streamA, streamB] = streams[device];
      this.cuda.freeHost(bufferA);
      this.cuda.freeHost(bufferB);
      this.cuda.streamDestroy(streamA);
      this.cuda.streamDestroy(streamB);
    }
    this.cuda.setDevice(0);

    const elapsedSec = (Date.now() - startTime) / 1000;
    const throughputGBps = (totalBytes / 1e9) / elapsedSec;
    this.logger.log(
      `Loaded cache: ${(totalBytes / 1e9).toFixed(2)} GB in ${elapsedSec.toFixed(1)}s (${throughputGBps.toFixed(2)} GB/s)`
    );

    return { weights: perDeviceWeights, poolPtrs };
  }

  /**
   * Save sharded weights to cache using streaming writes.
   * Writes tensors one at a time to avoid memory limits.
   */
  async saveToCache(
    modelPath: string,
    config: LlamaConfig,
    worldSize: number,
    perDeviceWeights: Map<number, Map<string, { ptr: bigint; info: any }>>,
    dtype: string = "float16"
  ): Promise<void> {
    const cacheDir = this.getCacheDir(modelPath, worldSize);

    // Create cache directory
    if (!existsSync(cacheDir)) {
      mkdirSync(cacheDir, { recursive: true });
    }

    const startTime = Date.now();

    const metadata: ShardCacheMetadata = {
      modelPath,
      modelHash: this.computeModelHash(modelPath, config),
      worldSize,
      dtype,
      createdAt: Date.now(),
      tensors: {},
    };

    let totalBytes = 0;
    let tensorCount = 0;

    // Count total tensors across all devices
    let totalTensors = 0;
    for (let device = 0; device < worldSize; device++) {
      totalTensors += perDeviceWeights.get(device)!.size;
    }

    // Find largest tensor for pinned buffer allocation
    let maxTensorSize = 0;
    for (let device = 0; device < worldSize; device++) {
      const weights = perDeviceWeights.get(device)!;
      for (const [_, weightInfo] of weights) {
        const numel = weightInfo.info.shape.reduce((a: number, b: number) => a * b, 1);
        const elementSize = DTYPE_SIZES[weightInfo.info.dtype as DType] ?? 2;
        maxTensorSize = Math.max(maxTensorSize, numel * elementSize);
      }
    }

    // Allocate pinned memory for D2H copies (reuse for all tensors)
    const pinnedPtr = this.cuda.mallocHost(maxTensorSize);

    // Save each device's weights to a separate file using streaming writes
    for (let device = 0; device < worldSize; device++) {
      this.cuda.setDevice(device);
      const weights = perDeviceWeights.get(device)!;

      const shardPath = join(cacheDir, `shard_${device}.bin`);
      // Open file for streaming writes
      const fd = openSync(shardPath, 'w');
      let offset = 0;

      for (const [name, weightInfo] of weights) {
        const numel = weightInfo.info.shape.reduce((a: number, b: number) => a * b, 1);
        const elementSize = DTYPE_SIZES[weightInfo.info.dtype as DType] ?? 2;
        const size = numel * elementSize;

        // Copy from GPU to pinned memory
        const tensorBuffer = new ArrayBuffer(size);
        this.cuda.memcpyD2H(tensorBuffer, weightInfo.ptr, size);

        // Write directly to file
        writeSync(fd, new Uint8Array(tensorBuffer));

        // Record in metadata with device prefix
        const metaName = `device_${device}_${name}`;
        metadata.tensors[metaName] = {
          shape: weightInfo.info.shape,
          dtype: weightInfo.info.dtype as string,
          offset,
          size,
        };

        offset += size;
        tensorCount++;
        totalBytes += size;

        // Update progress
        if (tensorCount % 20 === 0 || tensorCount === totalTensors) {
          const elapsedSec = (Date.now() - startTime) / 1000;
          const throughputGBps = elapsedSec > 0 ? (totalBytes / 1e9) / elapsedSec : 0;
          const progress = ((tensorCount / totalTensors) * 100).toFixed(1);
          this.logger.progress(
            `Saving cache: ${progress}% (${tensorCount}/${totalTensors} tensors) | ${throughputGBps.toFixed(2)} GB/s`
          );
        }
      }

      closeSync(fd);
    }

    // Free pinned memory
    this.cuda.freeHost(pinnedPtr);

    // Write metadata
    const metadataPath = join(cacheDir, "metadata.json");
    await Bun.write(metadataPath, JSON.stringify(metadata, null, 2));

    this.cuda.setDevice(0);

    const elapsedSec = (Date.now() - startTime) / 1000;
    const throughputGBps = (totalBytes / 1e9) / elapsedSec;
    this.logger.log(
      `Saved cache: ${(totalBytes / 1e9).toFixed(2)} GB in ${elapsedSec.toFixed(1)}s (${throughputGBps.toFixed(2)} GB/s)`
    );
  }

  /**
   * Clear cache for a specific model.
   */
  clearCache(modelPath: string, worldSize: number): void {
    const cacheDir = this.getCacheDir(modelPath, worldSize);
    if (existsSync(cacheDir)) {
      const files = readdirSync(cacheDir);
      for (const file of files) {
        unlinkSync(join(cacheDir, file));
      }
      this.logger.log(`Cleared cache: ${cacheDir}`);
    }
  }

  /**
   * Get total cache size in bytes.
   */
  getCacheSize(): number {
    if (!existsSync(this.cacheBaseDir)) return 0;

    let totalSize = 0;
    const dirs = readdirSync(this.cacheBaseDir);
    for (const dir of dirs) {
      const dirPath = join(this.cacheBaseDir, dir);
      if (statSync(dirPath).isDirectory()) {
        const files = readdirSync(dirPath);
        for (const file of files) {
          totalSize += statSync(join(dirPath, file)).size;
        }
      }
    }
    return totalSize;
  }

  /**
   * Compute a hash for cache invalidation.
   */
  private computeModelHash(modelPath: string, config: LlamaConfig): string {
    // Hash key fields that affect sharding
    const key = JSON.stringify({
      path: modelPath,
      layers: config.numHiddenLayers,
      hidden: config.hiddenSize,
      heads: config.numAttentionHeads,
      kvHeads: config.numKeyValueHeads,
      intermediate: config.intermediateSize,
      vocab: config.vocabSize,
    });
    return this.hashString(key);
  }

  /**
   * Simple string hash function.
   */
  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(16).padStart(8, "0");
  }
}
