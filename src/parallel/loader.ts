// Multi-GPU Model Loader with Tensor Parallelism
// Loads model weights and distributes them across GPUs

import { spawn } from "bun";
import { existsSync, readdirSync } from "fs";
import { join } from "path";
import { getCudaBackend, CudaBackend } from "../backend/cuda/bindings";
import { ModelLoader, LoadedModel, SharedMemoryInfo } from "../model/loader";
import { LlamaConfig, parseLlamaConfig } from "../model/config";
import { Tensor, DType, dtypeFromString, DTYPE_SIZES } from "../tensor/tensor";
import { ShardCache, CacheLoadResult } from "./shard_cache";
import { parseSafetensorsHeader, dtypeToString } from "../loader/safetensors";

export interface TensorParallelModel {
  config: LlamaConfig;
  localPath: string;
  // Map of device -> Map of weight name -> tensor
  perDeviceWeights: Map<number, Map<string, { ptr: bigint; info: any }>>;
  worldSize: number;
  // Pool pointers for efficient cleanup (one per device) - set when loaded from cache
  poolPtrs?: bigint[];
}

export interface TPLoaderOptions {
  useCache?: boolean;  // Enable pre-sharded weight cache (default: true)
  cacheDir?: string;   // Custom cache directory
  logger?: {
    log: (msg: string) => void;
    warn: (msg: string) => void;
    progress?: (msg: string) => void;  // For status updates that overwrite the line
  };
}

/**
 * Load and distribute model weights across multiple GPUs for tensor parallelism.
 */
export class TensorParallelModelLoader {
  private loader: ModelLoader;
  private cuda: CudaBackend;
  private shardCache: ShardCache;
  private useCache: boolean;
  private logger: { log: (msg: string) => void; warn: (msg: string) => void; progress: (msg: string) => void };
  private replicatedWarnings: Map<string, number> = new Map(); // pattern -> count

  constructor(options: TPLoaderOptions = {}) {
    this.loader = new ModelLoader();
    this.cuda = getCudaBackend();
    this.useCache = options.useCache !== false;  // Default to true

    // Set up logger with progress support
    const defaultProgress = (msg: string) => process.stdout.write(`\r${msg}`);
    this.logger = {
      log: options.logger?.log ?? console.log,
      warn: options.logger?.warn ?? console.warn,
      progress: options.logger?.progress ?? defaultProgress,
    };

    // Create shard cache with logger
    this.shardCache = new ShardCache(options.cacheDir, {
      log: this.logger.log,
      progress: this.logger.progress,
    });
  }

  /** Track a replicated weight warning (deduplicated by pattern) */
  private trackReplicatedWarning(pattern: string): void {
    const count = this.replicatedWarnings.get(pattern) || 0;
    this.replicatedWarnings.set(pattern, count + 1);
  }

  /** Emit summary of replicated warnings */
  private emitReplicatedSummary(): void {
    if (this.replicatedWarnings.size === 0) return;
    const total = Array.from(this.replicatedWarnings.values()).reduce((a, b) => a + b, 0);
    this.logger.log(`  Replicated ${total} tensors that couldn't be partitioned (biases, etc.)`);
    this.replicatedWarnings.clear();
  }

  /**
   * Replicate tensor data to all GPUs using parallel async H2D copies.
   * Uses pinned memory staging for maximum throughput.
   */
  private replicateToAllGPUs(
    tensorData: ArrayBuffer,
    worldSize: number,
    perDeviceWeights: Map<number, Map<string, { ptr: bigint; info: any }>>,
    name: string,
    shape: number[],
    dtype: string
  ): void {
    const byteLength = tensorData.byteLength;

    // For small tensors, just do sequential copies (pinned memory overhead not worth it)
    if (byteLength < 1024 * 1024) {  // < 1MB
      for (let device = 0; device < worldSize; device++) {
        this.cuda.setDevice(device);
        const gpuPtr = this.cuda.malloc(byteLength);
        this.cuda.memcpyH2D(gpuPtr, tensorData, byteLength);
        perDeviceWeights.get(device)!.set(name, {
          ptr: gpuPtr,
          info: { shape, dtype },
        });
      }
      return;
    }

    // Allocate pinned staging buffer
    const pinnedPtr = this.cuda.mallocHost(byteLength);

    // Copy to pinned memory
    this.cuda.memcpyHostToPinned(pinnedPtr, tensorData, byteLength);

    // Allocate GPU memory on all devices
    const gpuPtrs: bigint[] = [];
    for (let device = 0; device < worldSize; device++) {
      this.cuda.setDevice(device);
      gpuPtrs.push(this.cuda.malloc(byteLength));
    }

    // Start async copies to all GPUs
    for (let device = 0; device < worldSize; device++) {
      this.cuda.setDevice(device);
      this.cuda.memcpyH2DAsync(gpuPtrs[device], pinnedPtr, byteLength);
    }

    // Synchronize all devices
    for (let device = 0; device < worldSize; device++) {
      this.cuda.setDevice(device);
      this.cuda.synchronize();
    }

    // Free pinned memory
    this.cuda.freeHost(pinnedPtr);

    // Store pointers in weight maps
    for (let device = 0; device < worldSize; device++) {
      perDeviceWeights.get(device)!.set(name, {
        ptr: gpuPtrs[device],
        info: { shape, dtype },
      });
    }
  }

  /**
   * Load model and distribute weights across GPUs.
   *
   * Weight partitioning:
   * - Column parallel (QKV, gate, up): partition along output dim (dim 0)
   * - Row parallel (o_proj, down): partition along input dim (dim 1)
   * - Replicated (embeddings, norms, lm_head): copy to all GPUs
   */
  async load(modelId: string, worldSize: number): Promise<TensorParallelModel> {
    this.logger.progress(`Loading ${worldSize}-GPU tensor parallel model...`);

    // Check we have enough GPUs
    const deviceCount = this.cuda.getDeviceCount();
    if (deviceCount < worldSize) {
      throw new Error(`Need ${worldSize} GPUs but only ${deviceCount} available`);
    }

    // Resolve model ID to local path
    let localPath: string;
    if (existsSync(modelId)) {
      localPath = modelId;
    } else {
      // Download from HuggingFace Hub if needed
      this.logger.progress(`Resolving ${modelId}...`);
      localPath = await this.downloadModel(modelId);
    }

    // Load config first (needed for cache check)
    this.logger.progress(`Loading config...`);
    await new Promise(r => setImmediate(r));  // Allow spinner to update
    const config = await this.loadConfig(localPath);

    // Check for cached shards
    if (this.useCache) {
      this.logger.progress(`Checking shard cache...`);
      await new Promise(r => setImmediate(r));  // Allow spinner to update
      const cacheInfo = await this.shardCache.checkCache(localPath, worldSize, config);
      if (cacheInfo) {
        this.logger.progress(`Loading from cache...`);
        const cacheResult = await this.shardCache.loadFromCache(cacheInfo, worldSize);

        // Update config dtype from cache metadata
        if (cacheInfo.metadata.dtype === "float16") {
          config.dtype = "float16";
        } else if (cacheInfo.metadata.dtype === "bfloat16") {
          config.dtype = "bfloat16";
        }

        return {
          config,
          localPath,
          perDeviceWeights: cacheResult.weights,
          worldSize,
          poolPtrs: cacheResult.poolPtrs,
        };
      }
    }

    // Native loading - read safetensors directly without Python
    this.logger.progress(`Scanning safetensors files...`);
    const tensorIndex = await this.buildTensorIndex(localPath);

    // Detect dtype from first weight tensor
    const firstTensor = tensorIndex.tensors.values().next().value;
    if (firstTensor) {
      const dtype = dtypeToString(firstTensor.dtype);
      if (dtype === "float16") {
        config.dtype = "float16";
      } else if (dtype === "bfloat16") {
        config.dtype = "bfloat16";
      }
    }

    this.logger.progress(`Distributing ${(tensorIndex.totalBytes / 1e9).toFixed(1)}GB across ${worldSize} GPUs...`);

    // TODO: Use pooled allocation like cache loader for better performance:
    // 1. Call calculatePerDeviceSizes() to get total per-device memory needs
    // 2. Allocate one big pool per device upfront
    // 3. Modify helper methods to carve out from pools instead of individual mallocs
    // 4. Use pinned memory + async H2D transfers

    const perDeviceWeights: Map<number, Map<string, { ptr: bigint; info: any }>> = new Map();

    // Initialize weight maps for each device
    for (let device = 0; device < worldSize; device++) {
      perDeviceWeights.set(device, new Map());
    }

    // Process weights in large batches for better I/O throughput
    const tensorList = Array.from(tensorIndex.tensors.entries());
    const totalWeights = tensorList.length;
    const startTime = Date.now();
    const BATCH_SIZE = 128;  // Larger batches for better throughput

    let weightCount = 0;
    let totalBytesProcessed = 0;

    for (let batchStart = 0; batchStart < tensorList.length; batchStart += BATCH_SIZE) {
      const batchEnd = Math.min(batchStart + BATCH_SIZE, tensorList.length);
      const batch = tensorList.slice(batchStart, batchEnd);

      // Read all weights in batch concurrently from their respective files
      const batchData = await Promise.all(
        batch.map(async ([name, info]) => {
          const file = tensorIndex.files[info.fileIndex];
          const tensorBlob = file.slice(info.byteOffset, info.byteOffset + info.byteLength);
          const tensorData = await tensorBlob.arrayBuffer();
          return { name, info, tensorData };
        })
      );

      // Process each weight in the batch
      for (const { name, info, tensorData } of batchData) {
        const shape = info.shape;
        const strategy = this.getPartitionStrategy(name, shape, worldSize);

        // Convert safetensors dtype to our internal dtype
        const dtype = dtypeFromString(dtypeToString(info.dtype));

        totalBytesProcessed += info.byteLength;

        switch (strategy.type) {
          case "replicate":
            // Copy to all devices using parallel async H2D
            this.replicateToAllGPUs(tensorData, worldSize, perDeviceWeights, name, shape, dtype);
            break;

          case "partition_rows":
          case "partition_cols":
            // Partition and copy to each device
            const dim = strategy.type === "partition_rows" ? 0 : 1;
            await this.distributePartitionedWeightDirect(
              name, tensorData, shape, dim, worldSize, perDeviceWeights, dtype
            );
            break;

          case "partition_experts":
            // Partition experts across devices (dim 0 = expert dimension)
            await this.distributeExpertWeightDirect(
              name, tensorData, shape, worldSize, perDeviceWeights, dtype
            );
            break;
        }

        weightCount++;
      }

      // Update progress after each batch
      const progress = (weightCount / totalWeights) * 100;
      const progressPct = Math.floor(progress / 5) * 5;
      const prevProgress = ((weightCount - batch.length) / totalWeights) * 100;
      const prevProgressPct = Math.floor(prevProgress / 5) * 5;
      if (progressPct > prevProgressPct && progressPct > 0) {
        const elapsedSec = (Date.now() - startTime) / 1000;
        const throughputGBps = (totalBytesProcessed / 1e9) / elapsedSec;
        this.logger.progress(`Sharding weights: ${progressPct}% (${weightCount}/${totalWeights} tensors) | ${throughputGBps.toFixed(2)} GB/s`);
      }
    }

    const finalElapsedSec = (Date.now() - startTime) / 1000;
    const finalThroughputGBps = (totalBytesProcessed / 1e9) / finalElapsedSec;
    this.logger.log(
      `  Sharded weights | ${(totalBytesProcessed / 1e9).toFixed(2)} GB in ${finalElapsedSec.toFixed(1)}s (${finalThroughputGBps.toFixed(2)} GB/s H2D)`
    );

    // Reset to device 0
    this.cuda.setDevice(0);

    // Save to cache for future runs
    if (this.useCache) {
      await this.shardCache.saveToCache(localPath, config, worldSize, perDeviceWeights, config.dtype);
    }

    this.emitReplicatedSummary();
    this.logger.log(`Model distributed across ${worldSize} GPUs`);

    return {
      config,
      localPath,
      perDeviceWeights,
      worldSize,
    };
  }

  /**
   * Load weights to shared memory without GPU copy.
   */
  private async loadWeightsToSharedMemory(modelId: string): Promise<{
    info: SharedMemoryInfo;
    proc: ReturnType<typeof spawn>;
  }> {
    // Similar to ModelLoader but doesn't copy to GPU
    const pythonScript = `${import.meta.dir}/../../python/weight_loader.py`;
    const proc = spawn([
      "python3",
      pythonScript,
      modelId,
      "--wait",
      "--quiet"
    ], {
      stdout: "pipe",
      stderr: "inherit",  // Show progress output
      stdin: "pipe",
    });

    // Read the first line (JSON output) from stdout
    // Note: We can't read until done because the process is in --wait mode
    const reader = proc.stdout.getReader();
    let output = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      output += new TextDecoder().decode(value);
      // Check if we have a complete JSON line
      if (output.includes("\n")) {
        break;
      }
    }
    reader.releaseLock();

    // Parse the JSON output (first line)
    const firstLine = output.trim().split('\n')[0];
    if (!firstLine) {
      throw new Error(`Python weight loader produced no output. Check stderr above.`);
    }
    const info = JSON.parse(firstLine) as SharedMemoryInfo;

    return { info, proc };
  }

  /**
   * Download model from HuggingFace Hub.
   */
  private async downloadModel(modelId: string): Promise<string> {
    const scriptPath = join(import.meta.dir, "../../python/hf_download.py");

    const proc = spawn(["python3", scriptPath, "download", modelId], {
      stdout: "pipe",
      stderr: "inherit",
    });

    const output = await new Response(proc.stdout).text();
    const exitCode = await proc.exited;

    if (exitCode !== 0) {
      throw new Error(`Failed to download model: exit code ${exitCode}`);
    }

    const result = JSON.parse(output.trim());
    return result.path;
  }

  /**
   * Load config from local path.
   */
  private async loadConfig(localPath: string): Promise<LlamaConfig> {
    const configPath = join(localPath, "config.json");
    const configJson = JSON.parse(await Bun.file(configPath).text());
    return parseLlamaConfig(configJson);
  }

  /**
   * Build an index of all tensors in the model's safetensor files.
   * This allows efficient batch reading without Python.
   */
  private async buildTensorIndex(localPath: string): Promise<{
    tensors: Map<string, {
      name: string;
      dtype: string;
      shape: number[];
      byteOffset: number;
      byteLength: number;
      fileIndex: number;
    }>;
    files: ReturnType<typeof Bun.file>[];
    filePaths: string[];
    totalBytes: number;
  }> {
    const tensors = new Map<string, {
      name: string;
      dtype: string;
      shape: number[];
      byteOffset: number;
      byteLength: number;
      fileIndex: number;
    }>();
    const files: ReturnType<typeof Bun.file>[] = [];
    const filePaths: string[] = [];
    let totalBytes = 0;

    // Find safetensor files
    const safetensorFiles: string[] = [];
    const singleFile = join(localPath, "model.safetensors");
    if (existsSync(singleFile)) {
      safetensorFiles.push(singleFile);
    } else {
      // Look for sharded files
      const allFiles = readdirSync(localPath);
      for (const file of allFiles) {
        if (file.endsWith(".safetensors")) {
          safetensorFiles.push(join(localPath, file));
        }
      }
      safetensorFiles.sort();
    }

    if (safetensorFiles.length === 0) {
      throw new Error(`No safetensors files found in ${localPath}`);
    }

    // Parse each file's header
    for (let fileIndex = 0; fileIndex < safetensorFiles.length; fileIndex++) {
      const filePath = safetensorFiles[fileIndex];
      const file = Bun.file(filePath);
      files.push(file);
      filePaths.push(filePath);

      // Read header (first 1MB should be enough for any reasonable model)
      const headerSlice = await file.slice(0, Math.min(file.size, 1024 * 1024)).arrayBuffer();
      const parsed = parseSafetensorsHeader(new Uint8Array(headerSlice));

      for (const [name, info] of parsed.tensors) {
        tensors.set(name, {
          name,
          dtype: info.dtype,
          shape: info.shape,
          byteOffset: info.byteOffset,
          byteLength: info.byteLength,
          fileIndex,
        });
        totalBytes += info.byteLength;
      }
    }

    return { tensors, files, filePaths, totalBytes };
  }

  /**
   * Determine how to partition a weight.
   */
  private getPartitionStrategy(
    name: string,
    shape: number[],
    worldSize: number
  ): { type: "replicate" | "partition_rows" | "partition_cols" | "partition_experts" } {
    // MoE expert weights: shard experts across GPUs (Expert Parallelism)
    // Shape is [num_experts, ...] - partition along dim 0 (expert dimension)
    if (name.includes(".experts.")) {
      const numExperts = shape[0];
      if (numExperts % worldSize === 0) {
        return { type: "partition_experts" };
      }
      this.trackReplicatedWarning("experts");
      return { type: "replicate" };
    }

    // Replicated weights
    if (
      name.includes("embed_tokens") ||
      name.includes("layernorm") ||
      name.includes("norm.weight") ||
      name.includes("lm_head") ||
      name.includes("q_norm") ||
      name.includes("k_norm")
    ) {
      return { type: "replicate" };
    }

    // Column parallel (partition output dimension = rows)
    if (
      name.includes("q_proj") ||
      name.includes("k_proj") ||
      name.includes("v_proj") ||
      name.includes("gate_proj") ||
      name.includes("up_proj")
    ) {
      // Check if partitionable
      if (shape[0] % worldSize === 0) {
        return { type: "partition_rows" };
      }
      // Extract layer type for summary (e.g., "q_proj.bias")
      const match = name.match(/\.(q_proj|k_proj|v_proj|gate_proj|up_proj)\.(weight|bias)/);
      this.trackReplicatedWarning(match ? `${match[1]}.${match[2]}` : "column_parallel");
      return { type: "replicate" };
    }

    // Row parallel (partition input dimension = cols)
    if (name.includes("o_proj") || name.includes("down_proj")) {
      if (shape[1] % worldSize === 0) {
        return { type: "partition_cols" };
      }
      const match = name.match(/\.(o_proj|down_proj)\.(weight|bias)/);
      this.trackReplicatedWarning(match ? `${match[1]}.${match[2]}` : "row_parallel");
      return { type: "replicate" };
    }

    // Default: replicate
    return { type: "replicate" };
  }

  /**
   * Calculate per-device memory requirements for all tensors.
   * Returns array of sizes indexed by device.
   */
  private calculatePerDeviceSizes(
    tensorIndex: {
      tensors: Map<string, { name: string; dtype: string; shape: number[]; byteLength: number }>;
    },
    worldSize: number
  ): number[] {
    const deviceSizes: number[] = new Array(worldSize).fill(0);

    for (const [name, info] of tensorIndex.tensors) {
      const shape = info.shape;
      const strategy = this.getPartitionStrategy(name, shape, worldSize);
      const elementSize = DTYPE_SIZES[dtypeFromString(dtypeToString(info.dtype))] || 2;

      switch (strategy.type) {
        case "replicate":
          // Each device gets full tensor
          for (let d = 0; d < worldSize; d++) {
            deviceSizes[d] += info.byteLength;
          }
          break;

        case "partition_rows": {
          // Partition along dim 0
          const chunkSize = Math.floor(shape[0] / worldSize);
          if (chunkSize === 0) {
            // Falls back to replicate
            for (let d = 0; d < worldSize; d++) {
              deviceSizes[d] += info.byteLength;
            }
          } else {
            const rowElements = shape.length > 1 ? shape[1] : 1;
            const sliceBytes = chunkSize * rowElements * elementSize;
            for (let d = 0; d < worldSize; d++) {
              deviceSizes[d] += sliceBytes;
            }
          }
          break;
        }

        case "partition_cols": {
          // Partition along dim 1
          const chunkSize = Math.floor(shape[1] / worldSize);
          if (chunkSize === 0) {
            for (let d = 0; d < worldSize; d++) {
              deviceSizes[d] += info.byteLength;
            }
          } else {
            const sliceBytes = shape[0] * chunkSize * elementSize;
            for (let d = 0; d < worldSize; d++) {
              deviceSizes[d] += sliceBytes;
            }
          }
          break;
        }

        case "partition_experts": {
          // Partition experts along dim 0
          const numExperts = shape[0];
          const expertsPerRank = numExperts / worldSize;
          let expertElements = 1;
          for (let i = 1; i < shape.length; i++) {
            expertElements *= shape[i];
          }
          const sliceBytes = expertsPerRank * expertElements * elementSize;
          for (let d = 0; d < worldSize; d++) {
            deviceSizes[d] += sliceBytes;
          }
          break;
        }
      }
    }

    return deviceSizes;
  }

  /**
   * Distribute a partitioned weight directly from ArrayBuffer.
   */
  private async distributePartitionedWeightDirect(
    name: string,
    tensorData: ArrayBuffer,
    shape: number[],
    partitionDim: number,
    worldSize: number,
    perDeviceWeights: Map<number, Map<string, { ptr: bigint; info: any }>>,
    dtype: DType = DType.Float16
  ): Promise<void> {
    // Handle 1D tensors or tensors with missing dimensions
    if (shape.length === 1) {
      // For 1D tensors, partition along dim 0
      partitionDim = 0;
    }

    const dimSize = shape[partitionDim];
    const chunkSize = Math.floor(dimSize / worldSize);

    // If chunk size is 0, replicate instead
    if (chunkSize === 0) {
      this.trackReplicatedWarning("other");
      this.replicateToAllGPUs(tensorData, worldSize, perDeviceWeights, name, shape, dtype);
      return;
    }

    const elementSize = DTYPE_SIZES[dtype];

    for (let device = 0; device < worldSize; device++) {
      const start = device * chunkSize;

      let partitionedShape: number[];
      let sliceSize: number;
      let partitionedBuffer: ArrayBuffer;

      if (shape.length === 1) {
        // 1D tensor
        partitionedShape = [chunkSize];
        sliceSize = chunkSize * elementSize;
        const startByte = start * elementSize;
        partitionedBuffer = tensorData.slice(startByte, startByte + sliceSize);

      } else if (partitionDim === 0) {
        // Partition rows: [chunkSize, shape[1]]
        partitionedShape = [chunkSize, shape[1]];
        sliceSize = chunkSize * shape[1] * elementSize;

        // Copy the rows for this partition
        const startByte = start * shape[1] * elementSize;
        partitionedBuffer = tensorData.slice(startByte, startByte + sliceSize);

      } else {
        // Partition columns: [shape[0], chunkSize]
        partitionedShape = [shape[0], chunkSize];
        sliceSize = shape[0] * chunkSize * elementSize;
        partitionedBuffer = new ArrayBuffer(sliceSize);

        // Copy columns - need to gather non-contiguous data
        const partitionedView = new Uint8Array(partitionedBuffer);
        const srcView = new Uint8Array(tensorData);

        for (let row = 0; row < shape[0]; row++) {
          const srcOffset = (row * shape[1] + start) * elementSize;
          const dstOffset = row * chunkSize * elementSize;
          const rowBytes = chunkSize * elementSize;

          partitionedView.set(
            srcView.subarray(srcOffset, srcOffset + rowBytes),
            dstOffset
          );
        }
      }

      // Validate before GPU copy
      if (sliceSize === 0 || partitionedBuffer.byteLength === 0) {
        throw new Error(`Invalid partition for ${name}: sliceSize=${sliceSize}, bufferSize=${partitionedBuffer.byteLength}`);
      }

      // Copy to device
      this.cuda.setDevice(device);
      const gpuPtr = this.cuda.malloc(sliceSize);
      this.cuda.memcpyH2D(gpuPtr, partitionedBuffer, sliceSize);

      perDeviceWeights.get(device)!.set(name, {
        ptr: gpuPtr,
        info: { shape: partitionedShape, dtype },
      });
    }
  }

  /**
   * Distribute expert weights across devices (Expert Parallelism).
   * Expert weights have shape [num_experts, ...] and we shard along dim 0.
   * Each GPU gets experts [rank * experts_per_rank, (rank+1) * experts_per_rank).
   */
  private async distributeExpertWeightDirect(
    name: string,
    tensorData: ArrayBuffer,
    shape: number[],
    worldSize: number,
    perDeviceWeights: Map<number, Map<string, { ptr: bigint; info: any }>>,
    dtype: DType = DType.Float16
  ): Promise<void> {
    const numExperts = shape[0];
    const expertsPerRank = numExperts / worldSize;

    // Calculate size of each expert (all dimensions after dim 0)
    const elementSize = DTYPE_SIZES[dtype];
    let expertElementCount = 1;
    for (let i = 1; i < shape.length; i++) {
      expertElementCount *= shape[i];
    }
    const expertBytes = expertElementCount * elementSize;

    for (let device = 0; device < worldSize; device++) {
      const startExpert = device * expertsPerRank;

      // Build partitioned shape: [expertsPerRank, ...rest of dims]
      const partitionedShape = [expertsPerRank, ...shape.slice(1)];
      const sliceSize = expertsPerRank * expertBytes;

      // Extract contiguous chunk of experts for this device
      const startByte = startExpert * expertBytes;
      const partitionedBuffer = tensorData.slice(startByte, startByte + sliceSize);

      if (sliceSize === 0 || partitionedBuffer.byteLength === 0) {
        throw new Error(`Invalid expert partition for ${name}: sliceSize=${sliceSize}, bufferSize=${partitionedBuffer.byteLength}`);
      }

      // Copy to device
      this.cuda.setDevice(device);
      const gpuPtr = this.cuda.malloc(sliceSize);
      this.cuda.memcpyH2D(gpuPtr, partitionedBuffer, sliceSize);

      perDeviceWeights.get(device)!.set(name, {
        ptr: gpuPtr,
        info: { shape: partitionedShape, dtype },
      });
    }
  }

  /**
   * Distribute a partitioned weight to devices.
   */
  private async distributePartitionedWeight(
    name: string,
    weightInfo: { ptr: bigint; info: any },
    partitionDim: number,
    worldSize: number,
    perDeviceWeights: Map<number, Map<string, { ptr: bigint; info: any }>>
  ): Promise<void> {
    const fullShape = weightInfo.info.shape;
    const dimSize = fullShape[partitionDim];
    const chunkSize = dimSize / worldSize;

    // Read full weight from GPU
    const tensor = Tensor.fromPtr(weightInfo.ptr, fullShape, DType.Float16);
    const fullData = tensor.toArray();

    for (let device = 0; device < worldSize; device++) {
      const start = device * chunkSize;

      let partitionedShape: number[];
      let partitionedData: Float32Array;

      if (partitionDim === 0) {
        // Partition rows
        partitionedShape = [chunkSize, fullShape[1]];
        partitionedData = new Float32Array(chunkSize * fullShape[1]);

        for (let i = 0; i < chunkSize; i++) {
          const srcOffset = (start + i) * fullShape[1];
          const dstOffset = i * fullShape[1];
          partitionedData.set(fullData.subarray(srcOffset, srcOffset + fullShape[1]), dstOffset);
        }
      } else {
        // Partition columns
        partitionedShape = [fullShape[0], chunkSize];
        partitionedData = new Float32Array(fullShape[0] * chunkSize);

        for (let i = 0; i < fullShape[0]; i++) {
          const srcOffset = i * fullShape[1] + start;
          const dstOffset = i * chunkSize;
          for (let j = 0; j < chunkSize; j++) {
            partitionedData[dstOffset + j] = fullData[srcOffset + j];
          }
        }
      }

      // Copy to device
      this.cuda.setDevice(device);
      const partitionedTensor = Tensor.fromArray(partitionedData, partitionedShape, DType.Float16);

      perDeviceWeights.get(device)!.set(name, {
        ptr: partitionedTensor.ptr,
        info: { shape: partitionedShape, dtype: DType.Float16 },
      });
    }
  }

  /**
   * Free all distributed weights.
   */
  freeModel(model: TensorParallelModel): void {
    if (model.poolPtrs) {
      // Pooled allocation - free the pools instead of individual tensors
      for (let device = 0; device < model.poolPtrs.length; device++) {
        this.cuda.setDevice(device);
        this.cuda.free(model.poolPtrs[device]);
      }
    } else {
      // Individual allocations - free each tensor
      for (const [device, weights] of model.perDeviceWeights) {
        this.cuda.setDevice(device);
        for (const [name, weightInfo] of weights) {
          this.cuda.free(weightInfo.ptr);
        }
      }
    }
    this.cuda.setDevice(0);
  }
}
