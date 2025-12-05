// Unit tests for ShardCache
import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import { ShardCache } from "../src/parallel/shard_cache";
import { getCudaBackend, CudaBackend } from "../src/backend/cuda/bindings";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "fs";
import { join } from "path";

describe("ShardCache", () => {
  let cuda: CudaBackend;
  let testCacheDir: string;
  let shardCache: ShardCache;

  beforeAll(() => {
    cuda = getCudaBackend();
    cuda.initCublas(1);
    testCacheDir = "/tmp/binfer_test_cache_" + Date.now();
    shardCache = new ShardCache(testCacheDir, {
      log: () => {},
      progress: () => {},
    });
  });

  afterAll(() => {
    // Cleanup test directory
    if (existsSync(testCacheDir)) {
      rmSync(testCacheDir, { recursive: true });
    }
  });

  describe("getCacheDir", () => {
    it("should generate consistent cache directory for same model", () => {
      const dir1 = (shardCache as any).getCacheDir("/path/to/model", 4);
      const dir2 = (shardCache as any).getCacheDir("/path/to/model", 4);
      expect(dir1).toBe(dir2);
    });

    it("should generate different directories for different TP sizes", () => {
      const dir4 = (shardCache as any).getCacheDir("/path/to/model", 4);
      const dir8 = (shardCache as any).getCacheDir("/path/to/model", 8);
      expect(dir4).not.toBe(dir8);
      expect(dir4).toContain("_tp4");
      expect(dir8).toContain("_tp8");
    });
  });

  describe("checkCache", () => {
    it("should return null for non-existent cache", async () => {
      const result = await shardCache.checkCache("/nonexistent/model", 4, {} as any);
      expect(result).toBeNull();
    });

    it("should return valid cache when path and worldSize match", async () => {
      // Create a mock cache
      const modelPath = "/tmp/test_model_" + Date.now();
      const worldSize = 2;
      const cacheDir = (shardCache as any).getCacheDir(modelPath, worldSize);

      mkdirSync(cacheDir, { recursive: true });

      // Write metadata - must include correct modelPath
      const metadata = {
        modelPath,  // Must match the path we're checking with
        modelHash: "any_hash_doesnt_matter",
        worldSize,
        dtype: "bfloat16",
        tensorCount: 2,
        tensors: {
          "device_0_weight": { offset: 0, size: 100, shape: [10, 10], dtype: "bfloat16" },
          "device_1_weight": { offset: 0, size: 100, shape: [10, 10], dtype: "bfloat16" },
        },
      };
      writeFileSync(join(cacheDir, "metadata.json"), JSON.stringify(metadata));

      // Write shard files
      writeFileSync(join(cacheDir, "shard_0.bin"), Buffer.alloc(100));
      writeFileSync(join(cacheDir, "shard_1.bin"), Buffer.alloc(100));

      // Should return valid cache (path and worldSize match)
      const mockConfig = { numHiddenLayers: 1, hiddenSize: 1, numAttentionHeads: 1, numKeyValueHeads: 1, intermediateSize: 1, vocabSize: 1 };
      const result = await shardCache.checkCache(modelPath, worldSize, mockConfig as any);
      expect(result).not.toBeNull();
      expect(result?.valid).toBe(true);

      // Cleanup
      rmSync(cacheDir, { recursive: true });
    });
  });

  describe("parallel reads", () => {
    it("should read multiple slices in parallel", async () => {
      // Create a test file
      const testFile = "/tmp/parallel_read_test_" + Date.now() + ".bin";
      const size = 10 * 1024 * 1024; // 10MB
      writeFileSync(testFile, Buffer.alloc(size, 0x42));

      const file = Bun.file(testFile);
      const PARALLEL = 4;
      const CHUNK_SIZE = size / PARALLEL;

      const start = performance.now();
      const promises = [];
      for (let i = 0; i < PARALLEL; i++) {
        promises.push(file.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE).arrayBuffer());
      }
      const results = await Promise.all(promises);
      const elapsed = performance.now() - start;

      // Verify all chunks were read
      let totalRead = 0;
      for (const result of results) {
        totalRead += result.byteLength;
      }
      expect(totalRead).toBe(size);

      // Cleanup
      rmSync(testFile);
    });
  });

  describe("double buffering", () => {
    it("should allocate two pinned buffers", () => {
      const size = 1024;
      const ptr1 = cuda.mallocHost(size);
      const ptr2 = cuda.mallocHost(size);

      expect(ptr1).not.toBe(0n);
      expect(ptr2).not.toBe(0n);
      expect(ptr1).not.toBe(ptr2);

      cuda.freeHost(ptr1);
      cuda.freeHost(ptr2);
    });

    it("should copy data to pinned memory correctly", () => {
      const size = 1024;
      const pinnedPtr = cuda.mallocHost(size);
      const testData = new Uint8Array(size).fill(0xAB);

      cuda.memcpyHostToPinned(pinnedPtr, testData, size);

      // Allocate GPU memory and copy to verify
      const gpuPtr = cuda.malloc(size);
      cuda.memcpyH2DAsync(gpuPtr, pinnedPtr, size);
      cuda.synchronize();

      // Copy back to verify
      const resultBuffer = new ArrayBuffer(size);
      cuda.memcpyD2H(resultBuffer, gpuPtr, size);
      const result = new Uint8Array(resultBuffer);

      expect(result[0]).toBe(0xAB);
      expect(result[size - 1]).toBe(0xAB);

      cuda.free(gpuPtr);
      cuda.freeHost(pinnedPtr);
    });
  });

  describe("GDS availability", () => {
    it("should check GDS availability without crashing", () => {
      const available = cuda.gdsAvailable();
      // Just check it returns a boolean and doesn't crash
      expect(typeof available).toBe("boolean");
    });

    it("should handle GDS init/close gracefully", () => {
      // Skip this test - GDS init can hang if nvidia-fs kernel module isn't loaded
      // even when cuFile library is present
      console.log("Skipping GDS init test (may hang without nvidia-fs module)");
    });
  });
});

describe("Cache Loading Performance", () => {
  const CACHE_DIR = process.env.HOME + "/.cache/binfer/shards/621c1e1b_tp8";

  it("should load cached weights within expected time", async () => {
    // Skip if cache doesn't exist
    if (!existsSync(CACHE_DIR)) {
      console.log("Skipping: cache not found at", CACHE_DIR);
      return;
    }

    const cuda = getCudaBackend();
    cuda.initCublas(8);

    const metadataPath = join(CACHE_DIR, "metadata.json");
    const metadata = await Bun.file(metadataPath).json();

    const shardCache = new ShardCache(undefined, {
      log: () => {},
      progress: () => {},
    });

    const cacheInfo = { cacheDir: CACHE_DIR, metadata };

    const startTime = Date.now();
    const result = await shardCache.loadFromCache(cacheInfo, 8);
    const elapsed = (Date.now() - startTime) / 1000;

    // Calculate throughput
    let totalBytes = 0;
    for (const tensor of Object.values(metadata.tensors) as any[]) {
      totalBytes += tensor.size;
    }
    const throughputGBps = (totalBytes / 1e9) / elapsed;

    console.log(`Loaded ${(totalBytes / 1e9).toFixed(2)} GB in ${elapsed.toFixed(1)}s (${throughputGBps.toFixed(2)} GB/s)`);

    // Expect at least 2 GB/s throughput (our optimized version should hit 2.5+)
    expect(throughputGBps).toBeGreaterThan(2.0);

    // Cleanup GPU memory
    if (result.poolPtrs) {
      // Pooled allocation - free the pools
      for (let device = 0; device < result.poolPtrs.length; device++) {
        cuda.setDevice(device);
        cuda.free(result.poolPtrs[device]);
      }
    } else {
      // Individual allocations
      for (const [device, deviceWeights] of result.weights) {
        cuda.setDevice(device);
        for (const [name, { ptr }] of deviceWeights) {
          cuda.free(ptr);
        }
      }
    }
    cuda.setDevice(0);
  }, 120000); // 2 minute timeout
});
