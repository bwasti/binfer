// CUDA Profiler - Event-based timing for inference operations
// Uses CUDA events for accurate GPU timing with minimal overhead

import { getCudaBackend, CudaBackend } from "../backend/cuda/bindings";
import { existsSync, mkdirSync, writeFileSync, readFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";

export interface ProfileSpan {
  name: string;
  device: number;
  startEvent: bigint;
  endEvent: bigint;
  durationMs?: number;
  flops?: number;    // FLOPs for this operation
  bytes?: number;    // Memory bytes for this operation
  wallStartMs?: number;  // Wall-clock start time (for overhead tracking)
  wallEndMs?: number;    // Wall-clock end time
}

/**
 * GPU performance characteristics from microbenchmark
 */
export interface GpuPeakPerformance {
  peakTflops: number;      // Peak TFLOPs (measured via GEMM)
  peakBandwidthGBps: number;  // Peak memory bandwidth GB/s
  deviceName?: string;
}

/**
 * Detailed metrics for a single operation type
 */
export interface OperationMetrics {
  timeMs: number;
  flops: number;
  bytes: number;
  count: number;  // Number of times this operation was called
}

export interface LayerMetrics {
  layerIdx: number;
  inputNorm: number;
  qkv: number;
  qkNorm: number;
  rope: number;
  attention: number;
  outputProj: number;
  attnAllReduce: number;
  postNorm: number;
  mlpGateUp: number;
  mlpSwiglu: number;
  mlpDown: number;
  mlpAllReduce: number;
  residual: number;
  total: number;
}

export interface GenerationMetrics {
  prefill: {
    tokens: number;
    totalMs: number;
    tokensPerSec: number;
    layers: LayerMetrics[];
  };
  decode: {
    tokens: number;
    totalMs: number;
    tokensPerSec: number;
    avgLayerMs: number;
  };
  breakdown: {
    embedding: number;
    layers: number;
    finalNorm: number;
    lmHead: number;
    sampling: number;
  };
}

/**
 * CUDA Profiler for measuring GPU kernel execution times.
 *
 * Usage:
 *   const profiler = new CudaProfiler();
 *   profiler.enable();
 *
 *   const span = profiler.startSpan("attention", 0);
 *   // ... do work ...
 *   profiler.endSpan(span);
 *
 *   const metrics = profiler.collectMetrics();
 */
export class CudaProfiler {
  private cuda: CudaBackend;
  private enabled: boolean = false;
  private spans: ProfileSpan[] = [];
  private eventPools: Map<number, bigint[]> = new Map();  // device -> event pool
  private currentDevice: number = 0;

  constructor() {
    this.cuda = getCudaBackend();
  }

  enable(): void {
    this.enabled = true;
    this.spans = [];
  }

  disable(): void {
    this.enabled = false;
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Get an event from the pool or create a new one.
   */
  private getEvent(device: number): bigint {
    let pool = this.eventPools.get(device);
    if (!pool) {
      pool = [];
      this.eventPools.set(device, pool);
    }
    if (pool.length > 0) {
      return pool.pop()!;
    }
    return this.cuda.eventCreate();
  }

  /**
   * Return an event to the pool for reuse.
   */
  private returnEvent(event: bigint, device: number): void {
    let pool = this.eventPools.get(device);
    if (!pool) {
      pool = [];
      this.eventPools.set(device, pool);
    }
    pool.push(event);
  }

  /**
   * Start timing a span. Returns null if profiling is disabled.
   */
  startSpan(name: string, device: number = 0): ProfileSpan | null {
    if (!this.enabled) return null;

    // Record wall-clock time for overhead tracking
    const wallStartMs = performance.now();

    // Ensure we're on the right device
    if (device !== this.currentDevice) {
      this.cuda.setDevice(device);
      this.currentDevice = device;
    }

    const startEvent = this.getEvent(device);
    this.cuda.eventRecord(startEvent);

    return {
      name,
      device,
      startEvent,
      endEvent: 0n,
      wallStartMs,
    };
  }

  /**
   * End timing a span with optional FLOPs and bytes tracking.
   */
  endSpan(span: ProfileSpan | null, flops?: number, bytes?: number): void {
    if (!span || !this.enabled) return;

    // Record wall-clock end time first (before any CUDA calls)
    span.wallEndMs = performance.now();

    // Ensure we're on the right device
    if (span.device !== this.currentDevice) {
      this.cuda.setDevice(span.device);
      this.currentDevice = span.device;
    }

    span.endEvent = this.getEvent(span.device);
    span.flops = flops;
    span.bytes = bytes;
    this.cuda.eventRecord(span.endEvent);
    this.spans.push(span);
  }

  /**
   * Synchronize and compute elapsed times for all spans.
   * Returns aggregated timing, FLOPs, and bytes per operation name.
   */
  collectMetrics(): Map<string, number> {
    if (!this.enabled || this.spans.length === 0) {
      return new Map();
    }

    // Synchronize all devices
    const deviceCount = this.cuda.getDeviceCount();
    for (let i = 0; i < deviceCount; i++) {
      this.cuda.setDevice(i);
      this.cuda.synchronize();
    }

    const timeMetrics = new Map<string, number>();
    const flopsMetrics = new Map<string, number>();
    const bytesMetrics = new Map<string, number>();
    let totalWallTimeMs = 0;
    let totalGpuTimeMs = 0;

    for (const span of this.spans) {
      try {
        // Switch to the device where this span was recorded
        this.cuda.setDevice(span.device);

        // Compute elapsed time
        span.durationMs = this.cuda.eventElapsedTime(span.startEvent, span.endEvent);

        // Track wall-clock time for overhead calculation
        if (span.wallStartMs !== undefined && span.wallEndMs !== undefined) {
          const wallTime = span.wallEndMs - span.wallStartMs;
          totalWallTimeMs += wallTime;
        }
        totalGpuTimeMs += span.durationMs;

        // Aggregate timing by name
        const existingTime = timeMetrics.get(span.name) || 0;
        timeMetrics.set(span.name, existingTime + span.durationMs);

        // Aggregate FLOPs by name
        if (span.flops) {
          const existingFlops = flopsMetrics.get(span.name) || 0;
          flopsMetrics.set(span.name, existingFlops + span.flops);
        }

        // Aggregate bytes by name
        if (span.bytes) {
          const existingBytes = bytesMetrics.get(span.name) || 0;
          bytesMetrics.set(span.name, existingBytes + span.bytes);
        }
      } catch {
        // Skip events that can't be timed (e.g., cross-device issues)
      }

      // Return events to pool
      this.returnEvent(span.startEvent, span.device);
      this.returnEvent(span.endEvent, span.device);
    }

    // Reset to device 0
    this.cuda.setDevice(0);

    // Clear spans for next collection
    this.spans = [];

    // Merge into single map with prefixes for FLOPs and bytes
    const metrics = new Map<string, number>(timeMetrics);
    for (const [name, flops] of flopsMetrics) {
      metrics.set(`${name}__flops`, flops);
    }
    for (const [name, bytes] of bytesMetrics) {
      metrics.set(`${name}__bytes`, bytes);
    }

    // Add framework overhead metric (wall time - GPU time)
    // Note: If GPU time > wall time, it means kernels are overlapped (async execution working well)
    // We track both for analysis
    const overheadMs = Math.max(0, totalWallTimeMs - totalGpuTimeMs);
    if (overheadMs > 0.1) {  // Only show if > 0.1ms
      metrics.set("framework", overheadMs);
    }

    // Add wall-clock and GPU utilization metrics
    metrics.set("__wall_time_ms__", totalWallTimeMs);
    metrics.set("__gpu_time_ms__", totalGpuTimeMs);

    return metrics;
  }

  /**
   * Get raw spans without aggregation.
   */
  getRawSpans(): ProfileSpan[] {
    if (!this.enabled || this.spans.length === 0) {
      return [];
    }

    // Synchronize all events
    this.cuda.synchronize();

    for (const span of this.spans) {
      span.durationMs = this.cuda.eventElapsedTime(span.startEvent, span.endEvent);
    }

    return [...this.spans];
  }

  /**
   * Clear all spans and return events to pool.
   */
  reset(): void {
    for (const span of this.spans) {
      this.returnEvent(span.startEvent, span.device);
      if (span.endEvent !== 0n) {
        this.returnEvent(span.endEvent, span.device);
      }
    }
    this.spans = [];
  }

  /**
   * Cleanup all events.
   */
  dispose(): void {
    this.reset();
    for (const [device, pool] of this.eventPools) {
      this.cuda.setDevice(device);
      for (const event of pool) {
        this.cuda.eventDestroy(event);
      }
    }
    this.eventPools.clear();
    this.cuda.setDevice(0);
  }

  /**
   * Format metrics as a human-readable report.
   * Includes per-operation MFU and memory bandwidth stats.
   */
  static formatReport(
    metrics: Map<string, number>,
    totalTokens: number,
    totalMs: number,
    gpuPeakTflops?: number,
    gpuPeakBandwidthGBps?: number
  ): string {
    // Get peak performance from microbenchmark if not provided
    const perf = getGpuPeakPerformance();
    const peakTflops = gpuPeakTflops ?? perf.peakTflops;
    const peakBwGBps = gpuPeakBandwidthGBps ?? perf.peakBandwidthGBps;

    const lines: string[] = [];
    lines.push("=== Inference Profile ===");
    lines.push(`Total: ${totalTokens} tokens in ${totalMs.toFixed(1)}ms (${(totalTokens / totalMs * 1000).toFixed(1)} tok/s)`);
    lines.push(`GPU Peak: ${peakTflops.toFixed(0)} TFLOPs, ${peakBwGBps.toFixed(0)} GB/s`);

    // Extract special metrics
    const prefillTokens = metrics.get("__prefill_tokens__") || 0;
    const decodeTokens = metrics.get("__decode_tokens__") || 0;
    const wallTimeMs = metrics.get("__wall_time_ms__") || 0;
    const gpuTimeMs = metrics.get("__gpu_time_ms__") || 0;

    if (prefillTokens > 0 || decodeTokens > 0) {
      lines.push(`Tokens: ${prefillTokens} prefill, ${decodeTokens} decode`);
    }

    // Show wall vs GPU time if available (helps identify framework overhead)
    if (wallTimeMs > 0 && gpuTimeMs > 0) {
      const asyncEfficiency = gpuTimeMs > wallTimeMs
        ? "kernels overlapped (async efficient)"
        : `${(wallTimeMs - gpuTimeMs).toFixed(1)}ms framework overhead`;
      lines.push(`Timing: ${wallTimeMs.toFixed(1)}ms wall, ${gpuTimeMs.toFixed(1)}ms GPU (${asyncEfficiency})`);
    }

    // Build operation data with timing, flops, bytes
    interface OpData {
      timeMs: number;
      flops: number;
      bytes: number;
    }
    const opData = new Map<string, OpData>();

    for (const [key, value] of metrics) {
      if (key.startsWith("__")) continue;

      if (key.endsWith("__flops")) {
        const opName = key.replace("__flops", "");
        const existing = opData.get(opName) || { timeMs: 0, flops: 0, bytes: 0 };
        existing.flops = value;
        opData.set(opName, existing);
      } else if (key.endsWith("__bytes")) {
        const opName = key.replace("__bytes", "");
        const existing = opData.get(opName) || { timeMs: 0, flops: 0, bytes: 0 };
        existing.bytes = value;
        opData.set(opName, existing);
      } else {
        // It's a timing metric
        const existing = opData.get(key) || { timeMs: 0, flops: 0, bytes: 0 };
        existing.timeMs = value;
        opData.set(key, existing);
      }
    }

    // Calculate totals
    let totalFlops = 0, totalBytes = 0, totalMeasuredMs = 0;
    for (const data of opData.values()) {
      totalFlops += data.flops;
      totalBytes += data.bytes;
      totalMeasuredMs += data.timeMs;
    }

    // Sort operations by time descending
    const sorted = [...opData.entries()].sort((a, b) => b[1].timeMs - a[1].timeMs);

    if (sorted.length > 0) {
      lines.push("");
      lines.push("Per-Operation Breakdown:");
      lines.push("  " + "Operation".padEnd(18) + "Time".padStart(10) + "  %".padStart(6) +
                 "  TFLOPs/s".padStart(10) + "  MFU".padStart(7) + "  GB/s".padStart(8) + "  BW%".padStart(6));
      lines.push("  " + "-".repeat(65));

      // Add TOTAL row at the top
      {
        const totalTflopsPerSec = totalFlops > 0 ? (totalFlops / 1e12) / (totalMeasuredMs / 1000) : 0;
        const totalMfu = totalTflopsPerSec > 0 ? (totalTflopsPerSec / peakTflops) * 100 : 0;
        const totalGbps = totalBytes > 0 ? (totalBytes / 1e9) / (totalMeasuredMs / 1000) : 0;
        const totalBwUtil = totalGbps > 0 ? (totalGbps / peakBwGBps) * 100 : 0;
        const timeStr = totalMeasuredMs.toFixed(1) + "ms";
        const tflopsStr = totalFlops > 0 ? totalTflopsPerSec.toFixed(1) : "-";
        const mfuStr = totalFlops > 0 ? totalMfu.toFixed(1) + "%" : "-";
        const gbpsStr = totalBytes > 0 ? totalGbps.toFixed(0) : "-";
        const bwStr = totalBytes > 0 ? totalBwUtil.toFixed(1) + "%" : "-";
        lines.push(`  ${"TOTAL".padEnd(18)}${timeStr.padStart(10)}${"100.0%".padStart(7)}` +
                   `${tflopsStr.padStart(10)}${mfuStr.padStart(7)}${gbpsStr.padStart(8)}${bwStr.padStart(6)}`);
        lines.push("  " + "-".repeat(65));
      }

      for (const [name, data] of sorted) {
        const pct = totalMeasuredMs > 0 ? (data.timeMs / totalMeasuredMs * 100) : 0;
        const timeStr = data.timeMs.toFixed(1) + "ms";

        // Calculate per-op MFU and bandwidth
        let tflopsStr = "-", mfuStr = "-", gbpsStr = "-", bwStr = "-";
        if (data.flops > 0 && data.timeMs > 0) {
          const opTflopsPerSec = (data.flops / 1e12) / (data.timeMs / 1000);
          const opMfu = (opTflopsPerSec / peakTflops) * 100;
          tflopsStr = opTflopsPerSec.toFixed(1);
          mfuStr = opMfu.toFixed(1) + "%";
        }
        if (data.bytes > 0 && data.timeMs > 0) {
          const opGbps = (data.bytes / 1e9) / (data.timeMs / 1000);
          const opBwUtil = (opGbps / peakBwGBps) * 100;
          gbpsStr = opGbps.toFixed(0);
          bwStr = opBwUtil.toFixed(1) + "%";
        }

        lines.push(`  ${name.padEnd(18)}${timeStr.padStart(10)}${pct.toFixed(1).padStart(6)}%` +
                   `${tflopsStr.padStart(10)}${mfuStr.padStart(7)}${gbpsStr.padStart(8)}${bwStr.padStart(6)}`);
      }
    }

    // Add MoE sub-operation breakdown if available
    const moeGateUpMs = metrics.get("__moe_gate_up_ms__");
    const moeActivationMs = metrics.get("__moe_activation_ms__");
    const moeDownMs = metrics.get("__moe_down_ms__");
    const moeCallCount = metrics.get("__moe_call_count__");

    if (moeCallCount && moeCallCount > 0) {
      const gateUpMs = moeGateUpMs || 0;
      const activationMs = moeActivationMs || 0;
      const downMs = moeDownMs || 0;
      const moeTotalMs = gateUpMs + activationMs + downMs;

      lines.push("");
      lines.push("MoE Sub-Operation Breakdown:");
      lines.push(`  Calls: ${moeCallCount}`);
      lines.push(`  Gate+Up:    ${gateUpMs.toFixed(2)}ms (${(gateUpMs/moeTotalMs*100).toFixed(1)}%)`);
      lines.push(`  Activation: ${activationMs.toFixed(2)}ms (${(activationMs/moeTotalMs*100).toFixed(1)}%)`);
      lines.push(`  Down:       ${downMs.toFixed(2)}ms (${(downMs/moeTotalMs*100).toFixed(1)}%)`);
      lines.push(`  Total:      ${moeTotalMs.toFixed(2)}ms`);
    }

    return lines.join("\n");
  }
}

/**
 * Format large numbers with SI suffixes
 */
function formatNumber(n: number): string {
  if (n >= 1e15) return (n / 1e15).toFixed(2) + "P";
  if (n >= 1e12) return (n / 1e12).toFixed(2) + "T";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "G";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(2) + "K";
  return n.toFixed(0);
}

// Cached GPU peak performance from microbenchmark
let _gpuPeakPerformance: GpuPeakPerformance | null = null;

// Cache directory for benchmark results
const CACHE_DIR = join(homedir(), ".cache", "binfer", "benchmarks");

/**
 * Get a unique identifier for the current GPU configuration.
 */
function getGpuCacheKey(): string {
  const cuda = getCudaBackend();
  const deviceCount = cuda.getDeviceCount();
  // Use device name as identifier (could also use UUID if available)
  cuda.setDevice(0);
  const deviceName = cuda.getDeviceName?.(0) ?? "unknown";
  // Create a simple hash from device info
  return `gpu_${deviceCount}_${deviceName.replace(/[^a-zA-Z0-9]/g, "_")}`;
}

/**
 * Load cached benchmark results if available.
 */
function loadCachedBenchmark(): GpuPeakPerformance | null {
  try {
    const cacheKey = getGpuCacheKey();
    const cachePath = join(CACHE_DIR, `${cacheKey}.json`);
    if (existsSync(cachePath)) {
      const data = JSON.parse(readFileSync(cachePath, "utf-8"));
      // Validate cached data
      if (data.peakTflops && data.peakBandwidthGBps) {
        return data as GpuPeakPerformance;
      }
    }
  } catch {
    // Cache miss or invalid data
  }
  return null;
}

/**
 * Save benchmark results to cache.
 */
function saveBenchmarkCache(perf: GpuPeakPerformance): void {
  try {
    if (!existsSync(CACHE_DIR)) {
      mkdirSync(CACHE_DIR, { recursive: true });
    }
    const cacheKey = getGpuCacheKey();
    const cachePath = join(CACHE_DIR, `${cacheKey}.json`);
    writeFileSync(cachePath, JSON.stringify(perf, null, 2));
  } catch {
    // Ignore cache write errors
  }
}

/**
 * Run a microbenchmark to measure actual GPU peak performance.
 * Uses a large GEMM and memory copy to estimate peak TFLOPs and bandwidth.
 * Results are cached to disk based on GPU identifier.
 */
export function measureGpuPeakPerformance(forceRemeasure: boolean = false): GpuPeakPerformance {
  // Check memory cache
  if (_gpuPeakPerformance && !forceRemeasure) {
    return _gpuPeakPerformance;
  }

  // Check disk cache
  if (!forceRemeasure) {
    const cached = loadCachedBenchmark();
    if (cached) {
      _gpuPeakPerformance = cached;
      return cached;
    }
  }

  const cuda = getCudaBackend();
  const originalDevice = 0;
  cuda.setDevice(originalDevice);

  // Warmup and benchmark parameters
  const warmupIters = 3;
  const benchIters = 10;

  // --- GEMM Benchmark (for peak TFLOPs) ---
  // Use a large square matrix for maximum compute utilization
  const M = 4096, N = 4096, K = 4096;
  const elemSize = 2; // fp16/bf16
  const aSize = M * K * elemSize;
  const bSize = K * N * elemSize;
  const cSize = M * N * elemSize;

  const aPtr = cuda.malloc(aSize);
  const bPtr = cuda.malloc(bSize);
  const cPtr = cuda.malloc(cSize);

  // Zero initialize (prevents NaN issues)
  cuda.memset(aPtr, 0, aSize);
  cuda.memset(bPtr, 0, bSize);
  cuda.memset(cPtr, 0, cSize);

  // Warmup
  for (let i = 0; i < warmupIters; i++) {
    cuda.gemmBf16TransB(aPtr, bPtr, cPtr, M, N, K);
  }
  cuda.synchronize();

  // Benchmark GEMM
  const gemmStart = cuda.eventCreate();
  const gemmEnd = cuda.eventCreate();
  cuda.eventRecord(gemmStart);
  for (let i = 0; i < benchIters; i++) {
    cuda.gemmBf16TransB(aPtr, bPtr, cPtr, M, N, K);
  }
  cuda.eventRecord(gemmEnd);
  cuda.synchronize();

  const gemmMs = cuda.eventElapsedTime(gemmStart, gemmEnd);
  const gemmFlops = 2 * M * N * K * benchIters;
  const peakTflops = (gemmFlops / 1e12) / (gemmMs / 1000);

  cuda.eventDestroy(gemmStart);
  cuda.eventDestroy(gemmEnd);
  cuda.free(aPtr);
  cuda.free(bPtr);
  cuda.free(cPtr);

  // --- Memory Bandwidth Benchmark ---
  const copySize = 256 * 1024 * 1024; // 256 MB
  const srcPtr = cuda.malloc(copySize);
  const dstPtr = cuda.malloc(copySize);
  cuda.memset(srcPtr, 0, copySize);

  // Warmup
  for (let i = 0; i < warmupIters; i++) {
    cuda.memcpyD2D(dstPtr, srcPtr, copySize);
  }
  cuda.synchronize();

  // Benchmark memory copy
  const memStart = cuda.eventCreate();
  const memEnd = cuda.eventCreate();
  cuda.eventRecord(memStart);
  for (let i = 0; i < benchIters; i++) {
    cuda.memcpyD2D(dstPtr, srcPtr, copySize);
  }
  cuda.eventRecord(memEnd);
  cuda.synchronize();

  const memMs = cuda.eventElapsedTime(memStart, memEnd);
  const totalBytes = copySize * benchIters * 2; // read + write
  const peakBandwidthGBps = (totalBytes / 1e9) / (memMs / 1000);

  cuda.eventDestroy(memStart);
  cuda.eventDestroy(memEnd);
  cuda.free(srcPtr);
  cuda.free(dstPtr);

  _gpuPeakPerformance = {
    peakTflops,
    peakBandwidthGBps,
  };

  // Save to disk cache
  saveBenchmarkCache(_gpuPeakPerformance);

  return _gpuPeakPerformance;
}

/**
 * Get cached GPU peak performance, or run benchmark if not measured.
 */
export function getGpuPeakPerformance(): GpuPeakPerformance {
  if (_gpuPeakPerformance) {
    return _gpuPeakPerformance;
  }
  // Try loading from cache
  const cached = loadCachedBenchmark();
  if (cached) {
    _gpuPeakPerformance = cached;
    return cached;
  }
  // Run benchmark
  return measureGpuPeakPerformance();
}

// Singleton profiler instance
let _profiler: CudaProfiler | null = null;

export function getProfiler(): CudaProfiler {
  if (!_profiler) {
    _profiler = new CudaProfiler();
  }
  return _profiler;
}
