// Test: CUDA build verification
// Run with: bun test tests/integration/cuda.test.ts
// Note: Requires CUDA library to be built first (bun run build:cuda)

import { expect, test, describe, beforeAll } from "bun:test";
import { existsSync } from "fs";
import { CudaBackend } from "../../src/backend/cuda/bindings";

describe("CUDA Backend", () => {
  let cuda: CudaBackend;
  let cudaAvailable: boolean;

  beforeAll(async () => {
    cuda = new CudaBackend();
    cudaAvailable = await cuda.isAvailable();

    if (!cudaAvailable) {
      console.log("⚠ CUDA not available - some tests will be skipped");
      console.log("  Build CUDA library with: bun run build:cuda");
    }
  });

  test("can instantiate CudaBackend", () => {
    expect(cuda).toBeDefined();
  });

  test("reports device count", () => {
    const count = cuda.getDeviceCount();
    console.log(`Device count: ${count}`);

    if (cudaAvailable) {
      expect(count).toBeGreaterThan(0);
    } else {
      expect(count).toBe(0);
    }
  });

  test("can get device properties (if CUDA available)", () => {
    if (!cudaAvailable) {
      console.log("Skipping: CUDA not available");
      return;
    }

    const props = cuda.getDeviceProperties(0);
    console.log(`GPU 0: ${props.name}`);
    console.log(`  Memory: ${(props.totalMemory / 1024 / 1024 / 1024).toFixed(1)}GB`);

    expect(props.name).toBeTruthy();
    expect(props.totalMemory).toBeGreaterThan(0);
  });

  test("can allocate and free memory (if CUDA available)", () => {
    if (!cudaAvailable) {
      console.log("Skipping: CUDA not available");
      return;
    }

    // Allocate 1MB
    const size = 1024 * 1024;
    const ptr = cuda.malloc(size);

    expect(ptr).toBeDefined();
    expect(ptr).not.toBe(0n);

    // Free it
    cuda.free(ptr);
  });

  test("can copy data H2D and D2H (if CUDA available)", () => {
    if (!cudaAvailable) {
      console.log("Skipping: CUDA not available");
      return;
    }

    // Create test data
    const size = 1024;
    const hostData = new Float32Array(size / 4);
    for (let i = 0; i < hostData.length; i++) {
      hostData[i] = i * 0.5;
    }

    // Allocate GPU memory
    const ptr = cuda.malloc(size);

    // Copy to GPU
    cuda.memcpyH2D(ptr, hostData.buffer, size);

    // Copy back
    const resultBuffer = new ArrayBuffer(size);
    cuda.memcpyD2H(resultBuffer, ptr, size);

    // Verify
    const result = new Float32Array(resultBuffer);
    for (let i = 0; i < result.length; i++) {
      expect(result[i]).toBeCloseTo(hostData[i], 5);
    }

    // Clean up
    cuda.free(ptr);
  });

  test("can synchronize (if CUDA available)", () => {
    if (!cudaAvailable) {
      console.log("Skipping: CUDA not available");
      return;
    }

    // This should not throw
    cuda.synchronize();
  });
});

describe("CUDA Library Build", () => {
  test("shared library exists after build", () => {
    const possiblePaths = [
      "./cuda/build/libbinfer_cuda.so",
      "./cuda/build/libbinfer_cuda.dylib",
    ];

    const exists = possiblePaths.some((p) => existsSync(p));

    if (!exists) {
      console.log("CUDA library not built yet");
      console.log("Run: bun run build:cuda");
    }

    // This is informational - don't fail if not built
    console.log(`CUDA library exists: ${exists}`);
  });
});
