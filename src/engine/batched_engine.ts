// Batched Inference Engine with Paged Attention and Tensor Parallelism
// Supports continuous/dynamic batching for high throughput serving with multi-GPU support

import { Tensor, DType } from "../tensor/tensor";
import { LlamaConfig } from "../model/config";
import { PagedKVPool, SequenceKVState, BLOCK_SIZE } from "../kv/paged";
export { SequenceKVState } from "../kv/paged";
import { getCudaBackend, CudaBackend, CudaGraph, MultiDeviceCudaGraph } from "../backend/cuda/bindings";
import { TensorParallelContext } from "./tp_context";
import { DeviceContext } from "./device_context";
import { Sampler, SamplingParams } from "./sampler";
import { CudaProfiler, getProfiler } from "../profiler/profiler";

/**
 * Batch of sequences for prefill or decode
 */
export interface SequenceBatch {
  seqIds: number[];
  inputTokens: number[][];      // Per-sequence tokens
  kvStates: SequenceKVState[];  // KV state per sequence
}

/**
 * Result of a batched forward pass
 */
export interface BatchedForwardResult {
  logits: Float32Array[];  // Per-sequence logits (last position for prefill, all for decode)
}

/**
 * Batched Inference Engine with TP support using paged attention.
 *
 * Key features:
 * - Uses contiguous PagedKVPool per device (each device has its own KV heads)
 * - Batches multiple sequences together in single kernel calls
 * - Uses paged attention for KV cache lookups
 * - Supports tensor parallelism (TP>1) with NCCL all-reduce
 */
export class BatchedInferenceEngine {
  private ctx: TensorParallelContext;
  private cuda: CudaBackend;
  private kvPools: Map<number, PagedKVPool>;  // device -> pool
  private maxBlocksPerSeq: number;
  private numLayers: number;

  // RoPE cache (per-device)
  private ropeInitialized = false;
  private ropeCaches: Map<number, { cos: bigint; sin: bigint }> = new Map();
  private ropeMaxSeqLen: number = 0;

  // MoE
  private mxfp4InitializedDevices: Set<number> = new Set();
  // MoE scratch buffers (per-device) - pre-allocated for batched MoE
  private moeGateUpBuffers: Map<number, bigint> = new Map();
  private moeActivatedBuffers: Map<number, bigint> = new Map();
  private moeMaxTokens: number = 0;  // Max tokens these buffers can handle
  // cuBLAS MoE scratch buffers (per-device)
  private moeCublasScratch: Map<number, bigint> = new Map();
  private moeCublasScratchSize: Map<number, bigint> = new Map();

  // Scratch buffers for batched operations (host memory)
  private maxBatchSize: number;
  private maxKvMemoryGB: number;
  private quiet: boolean;

  // Profiling
  private profiler: CudaProfiler;
  private profilingEnabled = false;
  private flopsAccum = 0;         // FLOPs accumulated this generation
  private bytesAccum = 0;         // Memory bytes read/written
  private prefillTokens = 0;      // Tokens processed in prefill
  private decodeTokens = 0;       // Tokens generated in decode

  // CUDA Graph capture for decode loop
  private cudaGraphsEnabled = false;
  private isCapturing = false;  // True during graph capture (skip synchronization)
  private capturingGraph: MultiDeviceCudaGraph | null = null;  // Current graph being captured (for device switching)
  // Warmup counter: paddedBatchSize -> count (capture after 2 warmup runs)
  private graphWarmupCount: Map<number, number> = new Map();
  // Pre-allocated buffers for CUDA graphs (allocated on first warmup, used for all subsequent runs)
  private graphBuffersCache: Map<number, {
    inputIds: Map<number, bigint>;
    positions: Map<number, bigint>;
    blockTables: Map<number, bigint>;
    kvContextLens: Map<number, bigint>;
    attnContextLens: Map<number, bigint>;
    hiddenStates: Map<number, bigint>;
    outputLogits: Map<number, bigint>;
    outputTokens: Map<number, bigint>;
  }> = new Map();
  // Graph cache: paddedBatchSize -> captured graph info
  private graphCache: Map<number, {
    graph: CudaGraph | MultiDeviceCudaGraph;
    // Persistent buffers (per device) - pointers stay fixed, values updated before launch
    inputIds: Map<number, bigint>;      // [paddedBatchSize] int32
    positions: Map<number, bigint>;      // [paddedBatchSize] int32
    blockTables: Map<number, bigint>;    // [paddedBatchSize * maxBlocksPerSeq] int32
    kvContextLens: Map<number, bigint>;  // [paddedBatchSize] int32
    attnContextLens: Map<number, bigint>;// [paddedBatchSize] int32
    hiddenStates: Map<number, bigint>;   // [paddedBatchSize, hiddenSize] fp16/bf16
    outputLogits: Map<number, bigint>;   // [paddedBatchSize, vocabSize] fp32 (for argmax)
    outputTokens: Map<number, bigint>;   // [paddedBatchSize] int32 (argmax result)
  }> = new Map();

  constructor(
    ctx: TensorParallelContext,
    maxKvMemoryGB: number = 10,
    maxSeqLen: number = 4096,
    maxBatchSize: number = 64,
    quiet: boolean = false
  ) {
    this.ctx = ctx;
    this.cuda = getCudaBackend();
    this.profiler = getProfiler();
    this.maxBatchSize = maxBatchSize;
    this.maxKvMemoryGB = maxKvMemoryGB;
    this.quiet = quiet;
    this.maxBlocksPerSeq = Math.ceil(maxSeqLen / BLOCK_SIZE);
    this.numLayers = ctx.config.numHiddenLayers;

    // Create paged KV pool per device
    // Each device has its share of KV heads (numKvHeadsPerRank)
    const dtype = ctx.config.dtype === "bfloat16" ? DType.BFloat16 : DType.Float16;
    this.kvPools = new Map();

    for (const deviceCtx of ctx.devices) {
      deviceCtx.setActive();
      // Each device gets its share of KV heads and proportional memory
      const pool = new PagedKVPool(
        ctx.config.numHiddenLayers,
        ctx.numKvHeadsPerRank,  // Sharded KV heads
        ctx.config.headDim,
        maxKvMemoryGB / ctx.worldSize,  // Divide memory across devices
        dtype,
        deviceCtx.device,
        quiet
      );
      this.kvPools.set(deviceCtx.device, pool);
    }

    // Initialize RoPE cache (using tp_context which supports YaRN)
    this.ctx.initRopeCache(maxSeqLen);

    // Reset to device 0
    this.cuda.setDevice(0);
  }

  get config(): LlamaConfig {
    return this.ctx.config;
  }

  get worldSize(): number {
    return this.ctx.worldSize;
  }

  private get isBf16(): boolean {
    return this.ctx.config.dtype === "bfloat16";
  }

  private get isMoE(): boolean {
    return (this.ctx.config.numLocalExperts ?? 0) > 0;
  }

  /**
   * Switch to a device during graph capture (for multi-device graphs).
   * This handles the stream event handoff between devices.
   * Call this before deviceCtx.setActive() when iterating over devices during capture.
   */
  private switchDeviceForCapture(device: number): void {
    if (this.capturingGraph) {
      this.capturingGraph.switchToDevice(device);
    }
  }

  // ============================================================================
  // Dtype-aware operation dispatchers
  // ============================================================================

  private gemmTransB(A: bigint, B: bigint, C: bigint, M: number, N: number, K: number): void {
    if (this.isBf16) {
      this.cuda.gemmBf16TransB(A, B, C, M, N, K);
    } else {
      this.cuda.gemmF16TransB(A, B, C, M, N, K);
    }
  }

  private rmsnorm(
    input: bigint, weight: bigint, output: bigint,
    batchSize: number, seqLen: number, hiddenSize: number, eps: number
  ): void {
    if (this.isBf16) {
      this.cuda.rmsnormBf16(input, weight, output, batchSize, seqLen, hiddenSize, eps);
    } else {
      this.cuda.rmsnormF16(input, weight, output, batchSize, seqLen, hiddenSize, eps);
    }
  }

  private embedding(
    weight: bigint, inputIds: bigint, output: bigint,
    batchSize: number, seqLen: number, vocabSize: number, hiddenSize: number
  ): void {
    if (this.isBf16) {
      this.cuda.embeddingBf16(weight, inputIds, output, batchSize, seqLen, vocabSize, hiddenSize);
    } else {
      this.cuda.embeddingF16(weight, inputIds, output, batchSize, seqLen, vocabSize, hiddenSize);
    }
  }

  private swiglu(gate: bigint, up: bigint, output: bigint, numel: number): void {
    if (this.isBf16) {
      this.cuda.swigluBf16(gate, up, output, numel);
    } else {
      this.cuda.swigluF16(gate, up, output, numel);
    }
  }

  private add(a: bigint, b: bigint, output: bigint, numel: number): void {
    if (this.isBf16) {
      this.cuda.addBf16(a, b, output, numel);
    } else {
      this.cuda.addF16(a, b, output, numel);
    }
  }

  // Batched bias addition: output[i,j] = data[i,j] + bias[j]
  // Much more efficient than per-row kernel launches!
  private addBias(data: bigint, bias: bigint, numRows: number, rowSize: number): void {
    if (this.isBf16) {
      this.cuda.addBiasInplaceBf16(data, bias, numRows, rowSize);
    } else {
      // For F16, fall back to the slower per-row approach for now
      // TODO: Add F16 batched bias kernel
      for (let row = 0; row < numRows; row++) {
        this.cuda.addF16(data + BigInt(row * rowSize * 2), bias, data + BigInt(row * rowSize * 2), rowSize);
      }
    }
  }

  private rotaryEmbedding(
    q: bigint, k: bigint, cos: bigint, sin: bigint,
    batchSize: number, seqLen: number, numHeads: number, numKvHeads: number,
    headDim: number, positionOffset: number
  ): void {
    if (this.isBf16) {
      this.cuda.rotaryEmbeddingBf16(q, k, cos, sin, batchSize, seqLen, numHeads, numKvHeads, headDim, positionOffset);
    } else {
      this.cuda.rotaryEmbeddingF16(q, k, cos, sin, batchSize, seqLen, numHeads, numKvHeads, headDim, positionOffset);
    }
  }

  // Batched RoPE for decode mode: each sequence has 1 token with its own position
  private rotaryEmbeddingDecode(
    q: bigint, k: bigint, cos: bigint, sin: bigint,
    positions: bigint,  // GPU pointer to int32 array
    batchSize: number, numHeads: number, numKvHeads: number, headDim: number
  ): void {
    if (this.isBf16) {
      this.cuda.rotaryEmbeddingDecodeBf16(q, k, cos, sin, positions, batchSize, numHeads, numKvHeads, headDim);
    } else {
      // Fall back to per-sequence for F16 (TODO: add F16 decode kernel)
      // This shouldn't happen often since most models use BF16
      throw new Error("Decode RoPE not yet implemented for F16");
    }
  }

  private flashAttention(
    Q: bigint, K: bigint, V: bigint, O: bigint,
    batchSize: number, seqQ: number, seqKV: number, kvStride: number, qOffset: number,
    numHeads: number, numKvHeads: number, headDim: number, softmaxScale: number, isCausal: boolean
  ): void {
    if (this.isBf16) {
      this.cuda.flashAttentionBf16(Q, K, V, O, batchSize, seqQ, seqKV, kvStride, qOffset,
        numHeads, numKvHeads, headDim, softmaxScale, isCausal);
    } else {
      this.cuda.flashAttentionF16(Q, K, V, O, batchSize, seqQ, seqKV, kvStride, qOffset,
        numHeads, numKvHeads, headDim, softmaxScale, isCausal);
    }
  }

  private attentionWithSinks(
    Q: bigint, K: bigint, V: bigint, sinks: bigint, O: bigint,
    batchSize: number, seqQ: number, seqKV: number, kvStride: number, qOffset: number,
    numHeads: number, numKvHeads: number, headDim: number, softmaxScale: number, isCausal: boolean
  ): void {
    // Attention with sinks only available in bf16
    this.cuda.attentionWithSinksBf16(Q, K, V, sinks, O, batchSize, seqQ, seqKV, kvStride, qOffset,
      numHeads, numKvHeads, headDim, softmaxScale, isCausal);
  }

  private pagedAttention(
    Q: bigint, kCache: bigint, vCache: bigint, O: bigint,
    blockTables: Int32Array, contextLens: Int32Array,
    batchSize: number, numHeads: number, numKvHeads: number, headDim: number,
    blockSize: number, maxBlocksPerSeq: number, softmaxScale: number
  ): void {
    if (this.isBf16) {
      this.cuda.pagedAttentionBf16(Q, kCache, vCache, O, blockTables, contextLens,
        batchSize, numHeads, numKvHeads, headDim, blockSize, maxBlocksPerSeq, softmaxScale);
    } else {
      this.cuda.pagedAttentionF16(Q, kCache, vCache, O, blockTables, contextLens,
        batchSize, numHeads, numKvHeads, headDim, blockSize, maxBlocksPerSeq, softmaxScale);
    }
  }

  private pagedAttentionWithSinks(
    Q: bigint, kCache: bigint, vCache: bigint, sinks: bigint, O: bigint,
    blockTables: Int32Array, contextLens: Int32Array,
    batchSize: number, numHeads: number, numKvHeads: number, headDim: number,
    blockSize: number, maxBlocksPerSeq: number, softmaxScale: number
  ): void {
    // Only available in bf16
    this.cuda.pagedAttentionWithSinksBf16(Q, kCache, vCache, sinks, O, blockTables, contextLens,
      batchSize, numHeads, numKvHeads, headDim, blockSize, maxBlocksPerSeq, softmaxScale);
  }

  private pagedKvCacheUpdate(
    kCache: bigint, vCache: bigint, kNew: bigint, vNew: bigint,
    blockTable: Int32Array, contextLen: Int32Array,
    numSeqs: number, seqLen: number, numKvHeads: number, headDim: number, maxBlocksPerSeq: number
  ): void {
    if (this.isBf16) {
      this.cuda.pagedKvCacheUpdateBf16(kCache, vCache, kNew, vNew, blockTable, contextLen,
        numSeqs, seqLen, numKvHeads, headDim, maxBlocksPerSeq);
    } else {
      this.cuda.pagedKvCacheUpdateF16(kCache, vCache, kNew, vNew, blockTable, contextLen,
        numSeqs, seqLen, numKvHeads, headDim, maxBlocksPerSeq);
    }
  }

  // Device-pointer variants for pre-uploaded block tables (avoids 40x cudaMalloc per layer)
  private pagedKvCacheUpdateDevice(
    kCache: bigint, vCache: bigint, kNew: bigint, vNew: bigint,
    dBlockTables: bigint, dContextLens: bigint,
    numSeqs: number, seqLen: number, numKvHeads: number, headDim: number, maxBlocksPerSeq: number
  ): void {
    this.cuda.pagedKvCacheUpdateBf16Device(kCache, vCache, kNew, vNew, dBlockTables, dContextLens,
      numSeqs, seqLen, numKvHeads, headDim, maxBlocksPerSeq);
  }

  private pagedAttentionDevice(
    Q: bigint, kCache: bigint, vCache: bigint, O: bigint,
    dBlockTables: bigint, dContextLens: bigint,
    batchSize: number, numHeads: number, numKvHeads: number, headDim: number,
    blockSize: number, maxBlocksPerSeq: number, softmaxScale: number
  ): void {
    this.cuda.pagedAttentionBf16Device(Q, kCache, vCache, O, dBlockTables, dContextLens,
      batchSize, numHeads, numKvHeads, headDim, blockSize, maxBlocksPerSeq, softmaxScale);
  }

  private pagedAttentionWithSinksDevice(
    Q: bigint, kCache: bigint, vCache: bigint, sinks: bigint, O: bigint,
    dBlockTables: bigint, dContextLens: bigint,
    batchSize: number, numHeads: number, numKvHeads: number, headDim: number,
    blockSize: number, maxBlocksPerSeq: number, softmaxScale: number
  ): void {
    this.cuda.pagedAttentionWithSinksBf16Device(Q, kCache, vCache, sinks, O, dBlockTables, dContextLens,
      batchSize, numHeads, numKvHeads, headDim, blockSize, maxBlocksPerSeq, softmaxScale);
  }

  /**
   * Ensure MoE scratch buffers are allocated for the given number of tokens.
   * Buffers are pre-allocated to avoid cudaMalloc/cudaFree per kernel call.
   */
  private ensureMoeBuffers(
    deviceCtx: DeviceContext,
    numTokens: number,
    intermediateSize: number,
    topK: number
  ): void {
    // Check if we need to allocate or reallocate
    if (numTokens <= this.moeMaxTokens && this.moeGateUpBuffers.has(deviceCtx.device)) {
      return;  // Already have sufficient buffers
    }

    // Free old buffers if they exist
    const oldGateUp = this.moeGateUpBuffers.get(deviceCtx.device);
    const oldActivated = this.moeActivatedBuffers.get(deviceCtx.device);
    if (oldGateUp) this.cuda.free(oldGateUp);
    if (oldActivated) this.cuda.free(oldActivated);

    // Allocate new buffers with some headroom
    const bufferTokens = Math.max(numTokens, 64) * 2;  // 2x headroom
    const gateUpSize = bufferTokens * topK * intermediateSize * 2 * 2;  // [tokens, topK, intermediate*2] bf16
    const activatedSize = bufferTokens * topK * intermediateSize * 2;   // [tokens, topK, intermediate] bf16

    deviceCtx.setActive();
    const gateUpBuffer = this.cuda.malloc(gateUpSize);
    const activatedBuffer = this.cuda.malloc(activatedSize);

    this.moeGateUpBuffers.set(deviceCtx.device, gateUpBuffer);
    this.moeActivatedBuffers.set(deviceCtx.device, activatedBuffer);
    this.moeMaxTokens = bufferTokens;
  }

  /**
   * Ensure cuBLAS MoE scratch buffer is allocated for given parameters
   */
  private ensureMoeCublasScratch(
    deviceCtx: DeviceContext,
    numTokens: number,
    hiddenSize: number,
    intermediateSize: number,
    numExperts: number,
    topK: number
  ): bigint {
    // Calculate required scratch size
    const requiredSize = this.cuda.moeCublasScratchSize(
      numTokens, hiddenSize, intermediateSize, numExperts, topK
    );

    const existingScratch = this.moeCublasScratch.get(deviceCtx.device);
    const existingSize = this.moeCublasScratchSize.get(deviceCtx.device) ?? BigInt(0);

    // Reallocate if needed (with 2x headroom)
    if (!existingScratch || requiredSize > existingSize) {
      if (existingScratch) {
        this.cuda.free(existingScratch);
      }

      deviceCtx.setActive();
      const allocSize = requiredSize * BigInt(2);  // 2x headroom
      const scratch = this.cuda.malloc(Number(allocSize));

      this.moeCublasScratch.set(deviceCtx.device, scratch);
      this.moeCublasScratchSize.set(deviceCtx.device, allocSize);

      return scratch;
    }

    return existingScratch;
  }

  /**
   * Ensure grouped GEMM MoE scratch buffer is allocated for given parameters
   */
  private ensureMoeGroupedScratch(
    deviceCtx: DeviceContext,
    numTokens: number,
    hiddenSize: number,
    intermediateSize: number,
    numExperts: number,
    topK: number
  ): bigint {
    // Calculate required scratch size
    const requiredSize = this.cuda.moeCutlassScratchSize(
      numTokens, hiddenSize, intermediateSize, numExperts, topK
    );

    const existingScratch = this.moeCublasScratch.get(deviceCtx.device);
    const existingSize = this.moeCublasScratchSize.get(deviceCtx.device) ?? BigInt(0);

    // Reallocate if needed (with 2x headroom)
    if (!existingScratch || requiredSize > existingSize) {
      if (existingScratch) {
        this.cuda.free(existingScratch);
      }

      deviceCtx.setActive();
      const allocSize = requiredSize * BigInt(2);  // 2x headroom
      const scratch = this.cuda.malloc(Number(allocSize));

      this.moeCublasScratch.set(deviceCtx.device, scratch);
      this.moeCublasScratchSize.set(deviceCtx.device, allocSize);

      return scratch;
    }

    return existingScratch;
  }

  /**
   * Ensure grouped GEMM MoE scratch buffer is allocated for a given size
   */
  private ensureMoeGroupedScratchForSize(
    deviceCtx: DeviceContext,
    requiredSize: bigint
  ): bigint {
    const existingScratch = this.moeCublasScratch.get(deviceCtx.device);
    const existingSize = this.moeCublasScratchSize.get(deviceCtx.device) ?? BigInt(0);

    // Reallocate if needed (with 2x headroom)
    if (!existingScratch || requiredSize > existingSize) {
      if (existingScratch) {
        this.cuda.free(existingScratch);
      }

      deviceCtx.setActive();
      const allocSize = requiredSize * BigInt(2);  // 2x headroom
      const scratch = this.cuda.malloc(Number(allocSize));

      this.moeCublasScratch.set(deviceCtx.device, scratch);
      this.moeCublasScratchSize.set(deviceCtx.device, allocSize);

      return scratch;
    }

    return existingScratch;
  }

  /**
   * Ensure grouped GEMM MoE scratch buffer is allocated for EP (expert parallelism)
   */
  private ensureMoeGroupedScratchEP(
    deviceCtx: DeviceContext,
    numTokens: number,
    hiddenSize: number,
    intermediateSize: number,
    expertsPerRank: number,
    topK: number
  ): bigint {
    // Calculate required scratch size for EP version
    const requiredSize = this.cuda.moeCutlassScratchSizeEP(
      numTokens, hiddenSize, intermediateSize, expertsPerRank, topK
    );

    const existingScratch = this.moeCublasScratch.get(deviceCtx.device);
    const existingSize = this.moeCublasScratchSize.get(deviceCtx.device) ?? BigInt(0);

    // Reallocate if needed (with 2x headroom)
    if (!existingScratch || requiredSize > existingSize) {
      if (existingScratch) {
        this.cuda.free(existingScratch);
      }

      deviceCtx.setActive();
      const allocSize = requiredSize * BigInt(2);  // 2x headroom
      const scratch = this.cuda.malloc(Number(allocSize));

      this.moeCublasScratch.set(deviceCtx.device, scratch);
      this.moeCublasScratchSize.set(deviceCtx.device, allocSize);

      return scratch;
    }

    return existingScratch;
  }

  /**
   * Ensure Marlin-style MoE scratch buffer is allocated for EP (expert parallelism)
   * Uses work-stealing to process all experts in single kernel launches
   */
  private moeMarlinScratch = new Map<number, bigint>();
  private moeMarlinScratchSize = new Map<number, bigint>();

  private ensureMoeMarlinScratchEP(
    deviceCtx: DeviceContext,
    requiredSize: bigint
  ): bigint {
    const existingScratch = this.moeMarlinScratch.get(deviceCtx.device);
    const existingSize = this.moeMarlinScratchSize.get(deviceCtx.device) ?? BigInt(0);

    // Reallocate if needed (with 2x headroom)
    if (!existingScratch || requiredSize > existingSize) {
      if (existingScratch) {
        this.cuda.free(existingScratch);
      }

      deviceCtx.setActive();
      const allocSize = requiredSize * BigInt(2);  // 2x headroom
      const scratch = this.cuda.malloc(Number(allocSize));

      this.moeMarlinScratch.set(deviceCtx.device, scratch);
      this.moeMarlinScratchSize.set(deviceCtx.device, allocSize);

      return scratch;
    }

    return existingScratch;
  }

  /**
   * Initialize RoPE cache for given max sequence length on all devices
   */
  private initRopeCache(maxSeqLen: number): void {
    if (this.ropeInitialized && maxSeqLen <= this.ropeMaxSeqLen) {
      return;
    }

    const { headDim, ropeTheta } = this.ctx.config;
    const halfDim = headDim / 2;
    const theta = ropeTheta || 10000.0;

    // Generate position-dependent cos/sin tables in float32
    const cosF32 = new Float32Array(maxSeqLen * halfDim);
    const sinF32 = new Float32Array(maxSeqLen * halfDim);

    for (let pos = 0; pos < maxSeqLen; pos++) {
      for (let i = 0; i < halfDim; i++) {
        const freq = 1.0 / Math.pow(theta, (2 * i) / headDim);
        const angle = pos * freq;
        cosF32[pos * halfDim + i] = Math.cos(angle);
        sinF32[pos * halfDim + i] = Math.sin(angle);
      }
    }

    // Convert to BF16/FP16 depending on model dtype
    const cos = new Uint16Array(maxSeqLen * halfDim);
    const sin = new Uint16Array(maxSeqLen * halfDim);
    const floatView = new Float32Array(1);
    const int32View = new Int32Array(floatView.buffer);

    for (let i = 0; i < cosF32.length; i++) {
      // BF16: just take upper 16 bits of float32
      floatView[0] = cosF32[i];
      cos[i] = (int32View[0] >> 16) & 0xffff;
      floatView[0] = sinF32[i];
      sin[i] = (int32View[0] >> 16) & 0xffff;
    }

    const byteSize = cos.byteLength;

    // Free old caches and allocate new ones on each device
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      const oldCache = this.ropeCaches.get(deviceCtx.device);
      if (oldCache) {
        this.cuda.free(oldCache.cos);
        this.cuda.free(oldCache.sin);
      }

      const ropeCos = this.cuda.malloc(byteSize);
      const ropeSin = this.cuda.malloc(byteSize);
      this.cuda.memcpyH2D(ropeCos, cos.buffer, byteSize);
      this.cuda.memcpyH2D(ropeSin, sin.buffer, byteSize);

      this.ropeCaches.set(deviceCtx.device, { cos: ropeCos, sin: ropeSin });
    }

    this.ropeMaxSeqLen = maxSeqLen;
    this.ropeInitialized = true;
    this.cuda.setDevice(0);
  }

  /**
   * Get KV pool for a device
   */
  private getKvPool(device: number): PagedKVPool {
    return this.kvPools.get(device)!;
  }

  /**
   * Allocate blocks for a new sequence (on all devices)
   */
  allocateSequence(seqId: number, numTokens: number): SequenceKVState | null {
    const numBlocks = PagedKVPool.blocksForTokens(numTokens);

    // Check if we can allocate on all devices (they should have same block count)
    const pool0 = this.kvPools.get(0)!;
    if (!pool0.canAllocate(numBlocks)) {
      return null;  // OOM
    }

    // Allocate on all devices (same block IDs for consistent addressing)
    const blockIds: number[] = [];
    for (const [device, pool] of this.kvPools) {
      const ids = pool.allocateBlocks(seqId, numBlocks);
      if (ids === null) {
        // Rollback allocations on other devices
        for (const [d, p] of this.kvPools) {
          if (d < device) p.freeSequence(seqId);
        }
        return null;
      }
      if (device === 0) blockIds.push(...ids);
    }

    const state = new SequenceKVState(seqId);
    state.addBlocks(blockIds);
    state.numTokens = 0;  // Start at 0; caller updates after prefill
    return state;
  }

  /**
   * Extend a sequence with more tokens
   */
  extendSequence(state: SequenceKVState, numNewTokens: number): boolean {
    const newTotal = state.numTokens + numNewTokens;
    const blocksNeeded = PagedKVPool.blocksForTokens(newTotal);
    const blocksToAlloc = blocksNeeded - state.numBlocks;

    if (blocksToAlloc > 0) {
      const pool0 = this.kvPools.get(0)!;
      if (!pool0.canAllocate(blocksToAlloc)) {
        return false;  // OOM
      }

      // Allocate on all devices
      const newBlockIds: number[] = [];
      for (const [device, pool] of this.kvPools) {
        const ids = pool.allocateBlocks(state.seqId, blocksToAlloc);
        if (ids === null) return false;
        if (device === 0) newBlockIds.push(...ids);
      }
      state.addBlocks(newBlockIds);
    }

    state.numTokens = newTotal;
    return true;
  }

  /**
   * Free a sequence's blocks on all devices
   */
  freeSequence(state: SequenceKVState): void {
    for (const pool of this.kvPools.values()) {
      pool.freeSequence(state.seqId);
    }
    state.blockIds = [];
    state.numTokens = 0;
  }

  /**
   * Check if we can allocate blocks for a given token count
   */
  canAllocate(numTokens: number): boolean {
    const numBlocks = PagedKVPool.blocksForTokens(numTokens);
    return this.kvPools.get(0)!.canAllocate(numBlocks);
  }

  /**
   * Build block tables and context lengths for paged attention
   */
  private buildBlockTablesAndContextLens(
    kvStates: SequenceKVState[],
    queryLengths: number[]
  ): { blockTables: Int32Array; contextLens: Int32Array; maxContextLen: number } {
    const batchSize = kvStates.length;
    const blockTables = new Int32Array(batchSize * this.maxBlocksPerSeq);
    const contextLens = new Int32Array(batchSize);
    let maxContextLen = 0;

    for (let i = 0; i < batchSize; i++) {
      const contextLen = kvStates[i].numTokens;
      contextLens[i] = contextLen;
      maxContextLen = Math.max(maxContextLen, contextLen);

      const blockIds = kvStates[i].blockIds;
      for (let j = 0; j < blockIds.length; j++) {
        blockTables[i * this.maxBlocksPerSeq + j] = blockIds[j];
      }
    }

    return { blockTables, contextLens, maxContextLen };
  }

  /**
   * Forward pass through one transformer layer with paged attention and TP support
   */
  private forwardLayerPaged(
    hiddenStates: Map<number, Tensor>,  // device -> hidden tensor
    layerIdx: number,
    kvStates: SequenceKVState[],
    queryLengths: number[],      // Tokens per sequence in this batch
    positions: number[],         // Starting position per sequence
    isPrefill: boolean,
    positionsGpu?: Map<number, Tensor>,  // Pre-uploaded positions per device (for decode)
    decodeBuffers?: {
      blockTables: bigint;       // Pre-uploaded block tables (device ptr)
      kvContextLens: bigint;     // Pre-uploaded context lens for KV update (device ptr)
      attnContextLens: bigint;   // Pre-uploaded context lens for attention (device ptr)
    }
  ): void {
    const p = this.profiler;
    const {
      hiddenSize,
      headDim,
      rmsNormEps,
      intermediateSize,
    } = this.ctx.config;

    const batchSize = kvStates.length;
    const totalTokens = queryLengths.reduce((a, b) => a + b, 0);
    const prefix = `model.layers.${layerIdx}`;

    // Per-rank dimensions (sharded for TP)
    const qSize = this.ctx.numHeadsPerRank * headDim;
    const kvSize = this.ctx.numKvHeadsPerRank * headDim;

    // Build block tables and context lengths for paged attention
    const { blockTables, contextLens, maxContextLen } = this.buildBlockTablesAndContextLens(kvStates, queryLengths);

    // Pre-calculate norm stats (same for input_norm and post_norm)
    const normNumel = totalTokens * hiddenSize;
    const normFlops = 5 * normNumel;
    const normBytes = 2 * (normNumel + hiddenSize + normNumel); // input + weight + output

    // Pre-calculate allreduce bytes
    const allReduceBytes = this.ctx.worldSize > 1 ? 2 * totalTokens * hiddenSize * 2 : 0;

    // ========== Attention Sublayer (parallel across devices) ==========
    const afterAttn: Map<number, Tensor> = new Map();

    for (const deviceCtx of this.ctx.devices) {
      this.switchDeviceForCapture(deviceCtx.device);
      deviceCtx.setActive();
      const hidden = hiddenStates.get(deviceCtx.device)!;
      const kvPool = this.getKvPool(deviceCtx.device);
      const ropeCache = deviceCtx.ropeCache!;

      // 1. Input LayerNorm
      const inputNormSpan = p.startSpan("input_norm", deviceCtx.device);
      const inputNormWeight = deviceCtx.getWeight(`${prefix}.input_layernorm.weight`);
      const normedInput = deviceCtx.alloc([totalTokens, hiddenSize]);
      this.rmsnorm(hidden.ptr, inputNormWeight.ptr, normedInput.ptr, 1, totalTokens, hiddenSize, rmsNormEps);
      // RMSNorm: ~5 FLOPs per element (variance, rsqrt, scale), read input+weight, write output
      p.endSpan(inputNormSpan, normFlops, normBytes);
      this.trackOp(normFlops, normBytes);

      // 2. QKV projections (column parallel - partitioned output)
      const qkvSpan = p.startSpan("qkv_proj", deviceCtx.device);
      const qWeight = deviceCtx.getWeight(`${prefix}.self_attn.q_proj.weight`);
      const kWeight = deviceCtx.getWeight(`${prefix}.self_attn.k_proj.weight`);
      const vWeight = deviceCtx.getWeight(`${prefix}.self_attn.v_proj.weight`);

      const q = deviceCtx.alloc([totalTokens, qSize]);
      const k = deviceCtx.alloc([totalTokens, kvSize]);
      const v = deviceCtx.alloc([totalTokens, kvSize]);

      this.gemmTransB(normedInput.ptr, qWeight.ptr, q.ptr, totalTokens, qSize, hiddenSize);
      this.gemmTransB(normedInput.ptr, kWeight.ptr, k.ptr, totalTokens, kvSize, hiddenSize);
      this.gemmTransB(normedInput.ptr, vWeight.ptr, v.ptr, totalTokens, kvSize, hiddenSize);

      // Add QKV biases if present (batched - single kernel per tensor)
      if (deviceCtx.hasWeight(`${prefix}.self_attn.q_proj.bias`)) {
        const qBias = deviceCtx.getWeight(`${prefix}.self_attn.q_proj.bias`);
        const kBias = deviceCtx.getWeight(`${prefix}.self_attn.k_proj.bias`);
        const vBias = deviceCtx.getWeight(`${prefix}.self_attn.v_proj.bias`);
        this.addBias(q.ptr, qBias.ptr, totalTokens, qSize);
        this.addBias(k.ptr, kBias.ptr, totalTokens, kvSize);
        this.addBias(v.ptr, vBias.ptr, totalTokens, kvSize);
      }

      // 2b. QK-norm (Qwen3 style)
      if (this.ctx.config.useQkNorm) {
        const qNormWeight = deviceCtx.getWeight(`${prefix}.self_attn.q_norm.weight`);
        const kNormWeight = deviceCtx.getWeight(`${prefix}.self_attn.k_norm.weight`);
        this.rmsnorm(q.ptr, qNormWeight.ptr, q.ptr, totalTokens * this.ctx.numHeadsPerRank, 1, headDim, rmsNormEps);
        this.rmsnorm(k.ptr, kNormWeight.ptr, k.ptr, totalTokens * this.ctx.numKvHeadsPerRank, 1, headDim, rmsNormEps);
      }
      // QKV FLOPs: 3 GEMMs (Q, K, V)
      const qkvFlops = this.gemmFlops(totalTokens, qSize, hiddenSize) +
        this.gemmFlops(totalTokens, kvSize, hiddenSize) * 2;
      const qkvBytes = this.gemmBytes(totalTokens, qSize, hiddenSize) +
        this.gemmBytes(totalTokens, kvSize, hiddenSize) * 2;
      p.endSpan(qkvSpan, qkvFlops, qkvBytes);
      this.trackOp(qkvFlops, qkvBytes);

      // 3. RoPE - apply rotary embeddings
      const ropeSpan = p.startSpan("rope", deviceCtx.device);

      // For decode mode with BF16, use batched kernel with GPU positions
      // This is required for CUDA graphs (positions read from device buffer, not captured as constants)
      if (!isPrefill && this.isBf16 && positionsGpu && positionsGpu.has(deviceCtx.device)) {
        const positionsPtr = positionsGpu.get(deviceCtx.device)!;

        this.rotaryEmbeddingDecode(
          q.ptr, k.ptr, ropeCache.cos.ptr, ropeCache.sin.ptr,
          positionsPtr.ptr, batchSize, this.ctx.numHeadsPerRank, this.ctx.numKvHeadsPerRank, headDim
        );
      } else {
        // Fall back to per-sequence for prefill or small batches
        let tokenOffset = 0;
        for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
          const seqLen = queryLengths[seqIdx];
          const posOffset = positions[seqIdx];

          const qSeqPtr = q.ptr + BigInt(tokenOffset * qSize * 2);
          const kSeqPtr = k.ptr + BigInt(tokenOffset * kvSize * 2);

          this.rotaryEmbedding(
            qSeqPtr, kSeqPtr, ropeCache.cos.ptr, ropeCache.sin.ptr,
            1, seqLen, this.ctx.numHeadsPerRank, this.ctx.numKvHeadsPerRank, headDim, posOffset
          );

          tokenOffset += seqLen;
        }
      }
      // RoPE: 6 FLOPs per element (2 sin/cos muls + 2 adds per dim pair), read/write Q and K
      const ropeQNumel = totalTokens * qSize;
      const ropeKNumel = totalTokens * kvSize;
      const ropeFlops = 6 * (ropeQNumel + ropeKNumel);
      const ropeBytes = 2 * 2 * (ropeQNumel + ropeKNumel); // read + write for Q and K
      p.endSpan(ropeSpan, ropeFlops, ropeBytes);
      this.trackOp(ropeFlops, ropeBytes);

      // 4. Update paged KV cache with new K/V values
      const kvUpdateSpan = p.startSpan("kv_update", deviceCtx.device);

      // For decode mode (each sequence has 1 token), batch all sequences together
      // Use device buffers if provided (required for CUDA graph capture)
      if (!isPrefill && decodeBuffers) {
        // Use pre-uploaded device buffers (fast path - no per-layer malloc)
        this.pagedKvCacheUpdateDevice(
          kvPool.getKCachePtr(layerIdx),
          kvPool.getVCachePtr(layerIdx),
          k.ptr,
          v.ptr,
          decodeBuffers.blockTables,
          decodeBuffers.kvContextLens,
          batchSize,
          1,  // num_new_tokens = 1 for decode
          this.ctx.numKvHeadsPerRank,
          headDim,
          this.maxBlocksPerSeq
        );
      } else if (!isPrefill && batchSize > 1) {
        // Fallback for batched decode without pre-uploaded buffers: build and upload per-layer
        const batchedBlockTables = new Int32Array(batchSize * this.maxBlocksPerSeq);
        const batchedContextLens = new Int32Array(batchSize);

        for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
          batchedContextLens[seqIdx] = positions[seqIdx];
          const blockIds = kvStates[seqIdx].blockIds;
          for (let j = 0; j < blockIds.length; j++) {
            batchedBlockTables[seqIdx * this.maxBlocksPerSeq + j] = blockIds[j];
          }
        }

        this.pagedKvCacheUpdate(
          kvPool.getKCachePtr(layerIdx),
          kvPool.getVCachePtr(layerIdx),
          k.ptr,
          v.ptr,
          batchedBlockTables,
          batchedContextLens,
          batchSize,
          1,  // num_new_tokens = 1 for decode
          this.ctx.numKvHeadsPerRank,
          headDim,
          this.maxBlocksPerSeq
        );
      } else {
        // For prefill or single sequence, use per-sequence loop
        let tokenOffset = 0;
        for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
          const seqLen = queryLengths[seqIdx];
          const kSeqPtr = k.ptr + BigInt(tokenOffset * kvSize * 2);
          const vSeqPtr = v.ptr + BigInt(tokenOffset * kvSize * 2);

          const seqBlockTable = new Int32Array(this.maxBlocksPerSeq);
          for (let j = 0; j < kvStates[seqIdx].blockIds.length; j++) {
            seqBlockTable[j] = kvStates[seqIdx].blockIds[j];
          }
          const seqContextLen = new Int32Array([positions[seqIdx]]);

          this.pagedKvCacheUpdate(
            kvPool.getKCachePtr(layerIdx),
            kvPool.getVCachePtr(layerIdx),
            kSeqPtr,
            vSeqPtr,
            seqBlockTable,
            seqContextLen,
            1,
            seqLen,
            this.ctx.numKvHeadsPerRank,
            headDim,
            this.maxBlocksPerSeq
          );

          tokenOffset += seqLen;
        }
      }
      // KV cache update: read new K/V, write to cache
      const kvUpdateBytes = 2 * 2 * totalTokens * kvSize * 2; // read K+V, write K+V
      p.endSpan(kvUpdateSpan, 0, kvUpdateBytes);
      this.trackOp(0, kvUpdateBytes);

      // 5. Attention
      const attnSpan = p.startSpan("attention", deviceCtx.device);
      const attnOut = deviceCtx.alloc([totalTokens, qSize]);

      if (!isPrefill) {
        // Decode: one token per sequence, use paged attention
        const softmaxScale = 1.0 / Math.sqrt(headDim);

        // Check if model has attention sinks (GPT-OSS)
        const sinksKey = `${prefix}.self_attn.sinks`;
        const hasSinks = deviceCtx.hasWeight(sinksKey);

        if (decodeBuffers) {
          // Fast path: use pre-uploaded device buffers
          if (hasSinks) {
            const sinks = deviceCtx.getWeight(sinksKey);
            this.pagedAttentionWithSinksDevice(
              q.ptr,
              kvPool.getKCachePtr(layerIdx),
              kvPool.getVCachePtr(layerIdx),
              sinks.ptr,
              attnOut.ptr,
              decodeBuffers.blockTables,
              decodeBuffers.attnContextLens,
              batchSize,
              this.ctx.numHeadsPerRank,
              this.ctx.numKvHeadsPerRank,
              headDim,
              BLOCK_SIZE,
              this.maxBlocksPerSeq,
              softmaxScale
            );
          } else {
            this.pagedAttentionDevice(
              q.ptr,
              kvPool.getKCachePtr(layerIdx),
              kvPool.getVCachePtr(layerIdx),
              attnOut.ptr,
              decodeBuffers.blockTables,
              decodeBuffers.attnContextLens,
              batchSize,
              this.ctx.numHeadsPerRank,
              this.ctx.numKvHeadsPerRank,
              headDim,
              BLOCK_SIZE,
              this.maxBlocksPerSeq,
              softmaxScale
            );
          }
        } else {
          // Fallback: build context lens per layer (slow path)
          const newContextLens = new Int32Array(batchSize);
          for (let i = 0; i < batchSize; i++) {
            newContextLens[i] = positions[i] + 1;
          }

          if (hasSinks) {
            const sinks = deviceCtx.getWeight(sinksKey);
            this.pagedAttentionWithSinks(
              q.ptr,
              kvPool.getKCachePtr(layerIdx),
              kvPool.getVCachePtr(layerIdx),
              sinks.ptr,
              attnOut.ptr,
              blockTables,
              newContextLens,
              batchSize,
              this.ctx.numHeadsPerRank,
              this.ctx.numKvHeadsPerRank,
              headDim,
              BLOCK_SIZE,
              this.maxBlocksPerSeq,
              softmaxScale
            );
          } else {
            this.pagedAttention(
              q.ptr,
              kvPool.getKCachePtr(layerIdx),
              kvPool.getVCachePtr(layerIdx),
              attnOut.ptr,
              blockTables,
              newContextLens,
              batchSize,
              this.ctx.numHeadsPerRank,
              this.ctx.numKvHeadsPerRank,
              headDim,
              BLOCK_SIZE,
              this.maxBlocksPerSeq,
              softmaxScale
            );
          }
        }
      } else {
        // Prefill: use standard flash attention per sequence
        const softmaxScale = 1.0 / Math.sqrt(headDim);
        const sinksKey = `${prefix}.self_attn.sinks`;
        const hasSinks = deviceCtx.hasWeight(sinksKey);
        const sinks = hasSinks ? deviceCtx.getWeight(sinksKey) : null;

        let tokenOffset = 0;
        for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
          const seqLen = queryLengths[seqIdx];
          const qSeqPtr = q.ptr + BigInt(tokenOffset * qSize * 2);
          const kSeqPtr = k.ptr + BigInt(tokenOffset * kvSize * 2);
          const vSeqPtr = v.ptr + BigInt(tokenOffset * kvSize * 2);
          const oSeqPtr = attnOut.ptr + BigInt(tokenOffset * qSize * 2);

          if (hasSinks && sinks) {
            this.attentionWithSinks(
              qSeqPtr, kSeqPtr, vSeqPtr, sinks.ptr, oSeqPtr,
              1, seqLen, seqLen, seqLen, 0,
              this.ctx.numHeadsPerRank, this.ctx.numKvHeadsPerRank, headDim, softmaxScale, true
            );
          } else {
            this.flashAttention(
              qSeqPtr, kSeqPtr, vSeqPtr, oSeqPtr,
              1, seqLen, seqLen, seqLen, 0,
              this.ctx.numHeadsPerRank, this.ctx.numKvHeadsPerRank, headDim, softmaxScale, true
            );
          }

          tokenOffset += seqLen;
        }
      }
      // Attention FLOPs: 2 * batch * heads * seq^2 * head_dim for QK^T and softmax*V (approx)
      // For decode: seq=1, context_len varies; for prefill: both are seq_len
      // Memory: read Q, K, V, write O
      let attnFlops = 0;
      let attnBytes = 0;
      if (isPrefill) {
        // Prefill: each seq has seq_len^2 attention
        for (const seqLen of queryLengths) {
          attnFlops += 4 * this.ctx.numHeadsPerRank * seqLen * seqLen * headDim;
        }
        attnBytes = 2 * (totalTokens * qSize + totalTokens * kvSize * 2 + totalTokens * qSize);
      } else {
        // Decode: 1 query token attends to context_len KV
        for (let i = 0; i < batchSize; i++) {
          const contextLen = positions[i] + 1;
          attnFlops += 4 * this.ctx.numHeadsPerRank * contextLen * headDim;
        }
        attnBytes = 2 * (batchSize * qSize + maxContextLen * kvSize * 2 * batchSize + batchSize * qSize);
      }
      p.endSpan(attnSpan, attnFlops, attnBytes);
      this.trackOp(attnFlops, attnBytes);

      // 6. Output projection (row parallel - needs all-reduce)
      const oProjSpan = p.startSpan("o_proj", deviceCtx.device);
      const oWeight = deviceCtx.getWeight(`${prefix}.self_attn.o_proj.weight`);
      const attnProjected = deviceCtx.alloc([totalTokens, hiddenSize]);
      this.gemmTransB(attnOut.ptr, oWeight.ptr, attnProjected.ptr, totalTokens, hiddenSize, qSize);

      // Add o_proj bias if present - only on rank 0 to avoid multiplying bias by worldSize
      if (deviceCtx.device === 0 && deviceCtx.hasWeight(`${prefix}.self_attn.o_proj.bias`)) {
        const oBias = deviceCtx.getWeight(`${prefix}.self_attn.o_proj.bias`);
        this.addBias(attnProjected.ptr, oBias.ptr, totalTokens, hiddenSize);
      }

      afterAttn.set(deviceCtx.device, attnProjected);
      // o_proj FLOPs
      const oProjFlops = this.gemmFlops(totalTokens, hiddenSize, qSize);
      const oProjBytes = this.gemmBytes(totalTokens, hiddenSize, qSize);
      p.endSpan(oProjSpan, oProjFlops, oProjBytes);
      this.trackOp(oProjFlops, oProjBytes);

      // Cleanup attention tensors
      deviceCtx.free(normedInput);
      deviceCtx.free(q);
      deviceCtx.free(k);
      deviceCtx.free(v);
      deviceCtx.free(attnOut);
    }

    // All-reduce attention outputs (no-op for TP=1)
    const attnAllReduceSpan = p.startSpan("attn_allreduce", 0);
    if (this.ctx.worldSize > 1) {
      this.ctx.allReduceSum(afterAttn);
      // Skip synchronization during graph capture - stream ordering handles it
      if (!this.isCapturing) {
        this.ctx.synchronize();
      }
    }
    p.endSpan(attnAllReduceSpan, 0, allReduceBytes);
    this.trackOp(0, allReduceBytes);

    // ========== MLP Sublayer ==========
    const afterMlp: Map<number, Tensor> = new Map();

    for (const deviceCtx of this.ctx.devices) {
      this.switchDeviceForCapture(deviceCtx.device);
      deviceCtx.setActive();
      const hidden = hiddenStates.get(deviceCtx.device)!;
      const attnOut = afterAttn.get(deviceCtx.device)!;

      // 7. Residual connection
      const residualSpan = p.startSpan("residual", deviceCtx.device);
      const residual1 = deviceCtx.alloc([totalTokens, hiddenSize]);
      this.add(hidden.ptr, attnOut.ptr, residual1.ptr, totalTokens * hiddenSize);
      deviceCtx.free(attnOut);
      // Residual: 1 FLOP per element, read 2 inputs + write output
      const residualNumel = totalTokens * hiddenSize;
      const residualFlops = residualNumel;
      const residualBytes = 2 * 3 * residualNumel; // read a, read b, write c
      p.endSpan(residualSpan, residualFlops, residualBytes);
      this.trackOp(residualFlops, residualBytes);

      // 8. Post-attention LayerNorm
      const postNormSpan = p.startSpan("post_norm", deviceCtx.device);
      const postNormWeight = deviceCtx.getWeight(`${prefix}.post_attention_layernorm.weight`);
      const normedAfterAttn = deviceCtx.alloc([totalTokens, hiddenSize]);
      this.rmsnorm(residual1.ptr, postNormWeight.ptr, normedAfterAttn.ptr, 1, totalTokens, hiddenSize, rmsNormEps);
      // Same as input_norm
      p.endSpan(postNormSpan, normFlops, normBytes);
      this.trackOp(normFlops, normBytes);

      // 9. MLP (MoE or Dense)
      let mlpProjected: Tensor;

      if (this.isMoE) {
        // MoE: Mixture of Experts (replicated weights, no all-reduce needed)
        // Calculate MoE FLOPs and bytes for profiling
        const topK = this.ctx.config.numExpertsPerToken ?? 2;
        const numExperts = this.ctx.config.numLocalExperts ?? 8;
        const interSize = this.ctx.config.intermediateSize;

        // Router FLOPs: hidden @ router_weight^T = 2 * tokens * hidden * experts
        const routerFlops = 2 * totalTokens * hiddenSize * numExperts;

        // Gate+Up projection: 2 * tokens * topK * hidden * (intermediate * 2)
        // The "2" is for matmul, intermediate*2 is for gate and up combined
        const gateUpFlops = 2 * totalTokens * topK * hiddenSize * interSize * 2;

        // Activation: ~4 FLOPs per element (sigmoid, mul, add, mul for GPT-OSS activation)
        const activationFlops = 4 * totalTokens * topK * interSize;

        // Down projection: 2 * tokens * topK * intermediate * hidden
        const downFlops = 2 * totalTokens * topK * interSize * hiddenSize;

        const moeFlops = routerFlops + gateUpFlops + activationFlops + downFlops;

        // Bytes calculation for MXFP4 weights:
        // MXFP4: 4-bit per element = 0.5 bytes, plus E8M0 scale (1 byte per 32 elements)
        // Gate+Up weight: [numExperts, intermediate*2, hidden] but only topK experts used per token
        // Down weight: [numExperts, hidden, intermediate]
        // For effective bytes, we count what's actually read:
        const numBlocks = Math.ceil(hiddenSize / 32);
        const numBlocksInter = Math.ceil(interSize / 32);

        // Gate+Up weights read per token: topK * (intermediate*2 * hidden / 2 + scales)
        const gateUpWeightBytes = totalTokens * topK * (interSize * 2 * numBlocks * 16 + interSize * 2 * numBlocks);
        // Down weights read: topK * (hidden * intermediate / 2 + scales)
        const downWeightBytes = totalTokens * topK * (hiddenSize * numBlocksInter * 16 + hiddenSize * numBlocksInter);

        // Hidden/output tensors (bf16 = 2 bytes)
        const hiddenBytes = totalTokens * hiddenSize * 2;  // read
        const outputBytes = totalTokens * hiddenSize * 2;  // write
        // Intermediate buffers (internal to kernel)
        const intermediateBytes = totalTokens * topK * interSize * 2 * 3;  // gate_up + activated + down intermediate

        const moeBytes = gateUpWeightBytes + downWeightBytes + hiddenBytes + outputBytes + intermediateBytes;

        const moeSpan = p.startSpan("moe", deviceCtx.device);
        mlpProjected = this.forwardMoEBatched(deviceCtx, normedAfterAttn, prefix, layerIdx, totalTokens);
        p.endSpan(moeSpan, moeFlops, moeBytes);
        this.trackOp(moeFlops, moeBytes);
      } else {
        // Dense MLP (column parallel gate/up, row parallel down)
        const mlpGateUpSpan = p.startSpan("mlp_gate_up", deviceCtx.device);
        const gateWeight = deviceCtx.getWeight(`${prefix}.mlp.gate_proj.weight`);
        const upWeight = deviceCtx.getWeight(`${prefix}.mlp.up_proj.weight`);
        const downWeight = deviceCtx.getWeight(`${prefix}.mlp.down_proj.weight`);

        const gate = deviceCtx.alloc([totalTokens, this.ctx.intermediateSizePerRank]);
        const up = deviceCtx.alloc([totalTokens, this.ctx.intermediateSizePerRank]);

        this.gemmTransB(normedAfterAttn.ptr, gateWeight.ptr, gate.ptr, totalTokens, this.ctx.intermediateSizePerRank, hiddenSize);
        this.gemmTransB(normedAfterAttn.ptr, upWeight.ptr, up.ptr, totalTokens, this.ctx.intermediateSizePerRank, hiddenSize);
        // gate/up FLOPs
        const gateUpFlops = this.gemmFlops(totalTokens, this.ctx.intermediateSizePerRank, hiddenSize) * 2;
        const gateUpBytes = this.gemmBytes(totalTokens, this.ctx.intermediateSizePerRank, hiddenSize) * 2;
        p.endSpan(mlpGateUpSpan, gateUpFlops, gateUpBytes);
        this.trackOp(gateUpFlops, gateUpBytes);

        // SwiGLU activation
        const swigluSpan = p.startSpan("swiglu", deviceCtx.device);
        const mlpOut = deviceCtx.alloc([totalTokens, this.ctx.intermediateSizePerRank]);
        this.swiglu(gate.ptr, up.ptr, mlpOut.ptr, totalTokens * this.ctx.intermediateSizePerRank);
        // SwiGLU: ~4 FLOPs per element (sigmoid, mul, mul), read gate+up, write output
        const swigluNumel = totalTokens * this.ctx.intermediateSizePerRank;
        const swigluFlops = 4 * swigluNumel;
        const swigluBytes = 2 * (swigluNumel * 2 + swigluNumel); // read gate+up, write output
        p.endSpan(swigluSpan, swigluFlops, swigluBytes);
        this.trackOp(swigluFlops, swigluBytes);

        // Down projection
        const mlpDownSpan = p.startSpan("mlp_down", deviceCtx.device);
        mlpProjected = deviceCtx.alloc([totalTokens, hiddenSize]);
        this.gemmTransB(mlpOut.ptr, downWeight.ptr, mlpProjected.ptr, totalTokens, hiddenSize, this.ctx.intermediateSizePerRank);
        // down FLOPs
        const downFlops = this.gemmFlops(totalTokens, hiddenSize, this.ctx.intermediateSizePerRank);
        const downBytes = this.gemmBytes(totalTokens, hiddenSize, this.ctx.intermediateSizePerRank);
        p.endSpan(mlpDownSpan, downFlops, downBytes);
        this.trackOp(downFlops, downBytes);

        deviceCtx.free(gate);
        deviceCtx.free(up);
        deviceCtx.free(mlpOut);
      }

      afterMlp.set(deviceCtx.device, mlpProjected);

      // Store residual for final addition
      hiddenStates.set(deviceCtx.device, residual1);

      // Cleanup
      deviceCtx.free(normedAfterAttn);
    }

    // All-reduce MLP outputs (only needed for multi-GPU)
    // For dense MLP: sharded weights need reduce
    // For MoE with EP (worldSize > 1): each GPU has partial expert results, need reduce
    // For MoE with TP=1: no reduce needed (single GPU has all experts)
    const mlpAllReduceSpan = p.startSpan("mlp_allreduce", 0);
    if (this.ctx.worldSize > 1) {
      this.ctx.allReduceSum(afterMlp);
      // Skip synchronization during graph capture - stream ordering handles it
      if (!this.isCapturing) {
        this.ctx.synchronize();
      }
    }
    p.endSpan(mlpAllReduceSpan, 0, allReduceBytes);
    this.trackOp(0, allReduceBytes);

    // ========== Final Residual ==========
    for (const deviceCtx of this.ctx.devices) {
      this.switchDeviceForCapture(deviceCtx.device);
      deviceCtx.setActive();
      const residual1 = hiddenStates.get(deviceCtx.device)!;
      const mlpOut = afterMlp.get(deviceCtx.device)!;

      const output = deviceCtx.alloc([totalTokens, hiddenSize]);
      this.add(residual1.ptr, mlpOut.ptr, output.ptr, totalTokens * hiddenSize);

      deviceCtx.free(residual1);
      deviceCtx.free(mlpOut);
      hiddenStates.set(deviceCtx.device, output);
    }
  }

  /**
   * MoE forward pass for batched tokens.
   * Processes tokens through Mixture of Experts using MXFP4 quantized weights.
   */
  private forwardMoEBatched(
    deviceCtx: DeviceContext,
    hidden: Tensor,
    prefix: string,
    layerIdx: number,
    numTokens: number
  ): Tensor {
    const {
      hiddenSize,
      intermediateSize,
      numLocalExperts,
      numExpertsPerToken,
    } = this.ctx.config;
    const topK = numExpertsPerToken ?? 2;

    // Initialize MXFP4 tables on first MoE layer (must be done on each device for constant memory)
    if (layerIdx === 0) {
      const deviceId = deviceCtx.device;
      if (!this.mxfp4InitializedDevices.has(deviceId)) {
        deviceCtx.setActive();
        this.cuda.initMxfp4Tables();
        this.cuda.initMoeCutlassTables();
        this.mxfp4InitializedDevices.add(deviceId);
      }
    }

    // 1. Router: compute expert selection
    const routerWeight = deviceCtx.getWeight(`${prefix}.mlp.router.weight`);
    const routerBias = deviceCtx.hasWeight(`${prefix}.mlp.router.bias`)
      ? deviceCtx.getWeight(`${prefix}.mlp.router.bias`)
      : null;

    const expertIndices = deviceCtx.alloc([numTokens, topK], DType.Int32);
    const expertWeights = deviceCtx.alloc([numTokens, topK], deviceCtx.dtype);

    this.cuda.moeRouterTopK(
      hidden.ptr,
      routerWeight.ptr,
      routerBias?.ptr ?? BigInt(0),
      expertIndices.ptr,
      expertWeights.ptr,
      1,
      numTokens,
      hiddenSize,
      numLocalExperts!,
      topK
    );

    // 2. Get MXFP4 quantized expert weights
    const gateUpBlocks = deviceCtx.getWeight(`${prefix}.mlp.experts.gate_up_proj_blocks`);
    const gateUpScales = deviceCtx.getWeight(`${prefix}.mlp.experts.gate_up_proj_scales`);
    const gateUpBias = deviceCtx.hasWeight(`${prefix}.mlp.experts.gate_up_proj_bias`)
      ? deviceCtx.getWeight(`${prefix}.mlp.experts.gate_up_proj_bias`)
      : null;
    const downBlocks = deviceCtx.getWeight(`${prefix}.mlp.experts.down_proj_blocks`);
    const downScales = deviceCtx.getWeight(`${prefix}.mlp.experts.down_proj_scales`);
    const downBias = deviceCtx.hasWeight(`${prefix}.mlp.experts.down_proj_bias`)
      ? deviceCtx.getWeight(`${prefix}.mlp.experts.down_proj_bias`)
      : null;

    // For single-token per-sequence batched decode, use batched kernel for efficiency
    // The optimized MoE kernel with tensor cores handles any batch size efficiently
    if (numTokens <= 16384) {  // Increased from 256 - tensor core kernels handle large batches well
      const output = deviceCtx.alloc([numTokens, hiddenSize], deviceCtx.dtype);
      const worldSize = this.ctx.worldSize;

      // Ensure MoE scratch buffers are allocated
      this.ensureMoeBuffers(deviceCtx, numTokens, intermediateSize, topK);
      const gateUpBuffer = this.moeGateUpBuffers.get(deviceCtx.device)!;
      const activatedBuffer = this.moeActivatedBuffers.get(deviceCtx.device)!;

      // Expert parallelism: each GPU has expertsPerRank experts
      if (worldSize > 1) {
        const expertsPerRank = gateUpBlocks.shape[0]; // Local experts on this rank
        const rank = deviceCtx.device;

        // EP kernel selection:
        // - Single token decode: use optimized per-token kernel (best latency)
        // - Batched (numTokens > 1): Marlin work-stealing is default (best throughput)
        // - Set MOE_CUTLASS=1 to use CUTLASS grouped GEMM instead of Marlin
        const useMarlin = numTokens > 1 && process.env.MOE_CUTLASS !== "1";
        const useCutlass = numTokens > 1 && process.env.MOE_CUTLASS === "1";

        if (useMarlin) {
          // Use Marlin-style kernel with work-stealing (single kernel launch for all experts)
          const scratchSize = this.cuda.moeMarlinScratchSizeEP(numTokens, expertsPerRank, hiddenSize, intermediateSize, topK);
          const scratch = this.ensureMoeMarlinScratchEP(deviceCtx, scratchSize);

          this.cuda.moeMarlinForwardEP(
            hidden.ptr,
            gateUpBlocks.ptr,
            gateUpScales.ptr,
            gateUpBias?.ptr ?? BigInt(0),
            downBlocks.ptr,
            downScales.ptr,
            downBias?.ptr ?? BigInt(0),
            expertIndices.ptr,
            expertWeights.ptr,
            output.ptr,
            scratch,
            scratchSize,
            numTokens,
            hiddenSize,
            intermediateSize,
            expertsPerRank,
            topK,
            rank,
            worldSize
          );
        } else if (useCutlass) {
          // Use CUTLASS grouped GEMM kernel for EP (high throughput batched inference)
          const scratch = this.ensureMoeGroupedScratchEP(deviceCtx, numTokens, hiddenSize, intermediateSize, expertsPerRank, topK);
          const scratchSize = this.moeCublasScratchSize.get(deviceCtx.device)!;

          this.cuda.moeCutlassForwardEP(
            hidden.ptr,
            gateUpBlocks.ptr,
            gateUpScales.ptr,
            gateUpBias?.ptr ?? BigInt(0),
            downBlocks.ptr,
            downScales.ptr,
            downBias?.ptr ?? BigInt(0),
            expertIndices.ptr,
            expertWeights.ptr,
            output.ptr,
            scratch,
            scratchSize,
            numTokens,
            hiddenSize,
            intermediateSize,
            expertsPerRank,
            topK,
            rank,
            worldSize
          );
        } else {
          // Use optimized per-token kernel for single token decode (low latency)
          this.cuda.moeOptimizedForwardEP(
            hidden.ptr,
            gateUpBlocks.ptr,
            gateUpScales.ptr,
            gateUpBias?.ptr ?? BigInt(0),
            downBlocks.ptr,
            downScales.ptr,
            downBias?.ptr ?? BigInt(0),
            expertIndices.ptr,
            expertWeights.ptr,
            output.ptr,
            gateUpBuffer,
            activatedBuffer,
            numTokens,
            hiddenSize,
            intermediateSize,
            expertsPerRank,
            topK,
            rank,
            worldSize
          );
        }
      } else {
        // Single GPU MoE kernel selection:
        // - For single token decode: use optimized per-token kernel (best latency)
        // - For batched inference (numTokens > 1): use CUTLASS grouped GEMM (best throughput)
        // - Set MOE_OPTIMIZED=1 to force per-token kernel for all batch sizes
        const useGrouped = numTokens > 1 && process.env.MOE_OPTIMIZED !== "1";

        if (useGrouped) {
          // Use CUTLASS grouped GEMM kernel for batched inference
          const scratch = this.ensureMoeGroupedScratch(deviceCtx, numTokens, hiddenSize, intermediateSize, numLocalExperts!, topK);
          const scratchSize = this.moeCublasScratchSize.get(deviceCtx.device)!;

          this.cuda.moeCutlassForward(
            hidden.ptr,
            gateUpBlocks.ptr,
            gateUpScales.ptr,
            gateUpBias?.ptr ?? BigInt(0),
            downBlocks.ptr,
            downScales.ptr,
            downBias?.ptr ?? BigInt(0),
            expertIndices.ptr,
            expertWeights.ptr,
            output.ptr,
            scratch,
            scratchSize,
            numTokens,
            hiddenSize,
            intermediateSize,
            numLocalExperts!,
            topK
          );
        } else {
          // Default for single token: optimized kernel with tensor cores
          this.cuda.moeOptimizedForward(
            hidden.ptr,
            gateUpBlocks.ptr,
            gateUpScales.ptr,
            gateUpBias?.ptr ?? BigInt(0),
            downBlocks.ptr,
            downScales.ptr,
            downBias?.ptr ?? BigInt(0),
            expertIndices.ptr,
            expertWeights.ptr,
            output.ptr,
            gateUpBuffer,
            activatedBuffer,
            numTokens,
            hiddenSize,
            intermediateSize,
            numLocalExperts!,
            topK
          );
        }
      }

      deviceCtx.free(expertIndices);
      deviceCtx.free(expertWeights);
      return output;
    }

    // For larger batches (prefill), fall back to per-expert batching
    this.cuda.synchronize();
    const indicesBuffer = new ArrayBuffer(numTokens * topK * 4);
    this.cuda.memcpyD2H(indicesBuffer, expertIndices.ptr, indicesBuffer.byteLength);
    const indicesArray = new Int32Array(indicesBuffer);

    const weightsBuffer = new ArrayBuffer(numTokens * topK * 2);
    this.cuda.memcpyD2H(weightsBuffer, expertWeights.ptr, weightsBuffer.byteLength);
    const weightsUint16 = new Uint16Array(weightsBuffer);

    // Convert BF16 weights to float
    const weightsFloat = new Float32Array(numTokens * topK);
    for (let i = 0; i < weightsUint16.length; i++) {
      const bf16 = weightsUint16[i];
      const f32Buffer = new Float32Array(1);
      const i32View = new Int32Array(f32Buffer.buffer);
      i32View[0] = bf16 << 16;
      weightsFloat[i] = f32Buffer[0];
    }

    // Group tokens by expert
    const tokensByExpert = new Map<number, number[]>();
    for (let t = 0; t < numTokens; t++) {
      for (let k = 0; k < topK; k++) {
        const expertIdx = indicesArray[t * topK + k];
        if (expertIdx >= 0) {
          if (!tokensByExpert.has(expertIdx)) {
            tokensByExpert.set(expertIdx, []);
          }
          tokensByExpert.get(expertIdx)!.push(t * topK + k);
        }
      }
    }

    const numBlocksGateUp = gateUpBlocks.shape[2];
    const numBlocksDown = downBlocks.shape[2];

    // Dequantize experts
    const dequantedGateUp = new Map<number, bigint>();
    const dequantedDown = new Map<number, bigint>();

    const uniqueExperts = Array.from(tokensByExpert.keys());
    for (const expertIdx of uniqueExperts) {
      const gateUpBuf = deviceCtx.alloc([intermediateSize * 2, hiddenSize], deviceCtx.dtype);
      const downBuf = deviceCtx.alloc([hiddenSize, intermediateSize], deviceCtx.dtype);

      this.cuda.mxfp4DequantSingleExpert(
        gateUpBlocks.ptr,
        gateUpScales.ptr,
        gateUpBuf.ptr,
        expertIdx,
        numLocalExperts!,
        intermediateSize * 2,
        numBlocksGateUp,
        hiddenSize
      );

      this.cuda.mxfp4DequantSingleExpert(
        downBlocks.ptr,
        downScales.ptr,
        downBuf.ptr,
        expertIdx,
        numLocalExperts!,
        hiddenSize,
        numBlocksDown,
        intermediateSize
      );

      dequantedGateUp.set(expertIdx, gateUpBuf.ptr);
      dequantedDown.set(expertIdx, downBuf.ptr);
    }

    // Output accumulator
    const output = deviceCtx.alloc([numTokens, hiddenSize], deviceCtx.dtype);
    this.cuda.memset(output.ptr, 0, numTokens * hiddenSize * 2);

    // Process each active expert
    for (const [expertIdx, tokenKs] of tokensByExpert) {
      const expertGateUpPtr = dequantedGateUp.get(expertIdx)!;
      const expertDownPtr = dequantedDown.get(expertIdx)!;

      for (const tk of tokenKs) {
        const tokenIdx = Math.floor(tk / topK);
        const weight = weightsFloat[tk];

        const tokenHiddenPtr = hidden.ptr + BigInt(tokenIdx * hiddenSize * 2);

        const gateUp = deviceCtx.alloc([1, intermediateSize * 2], deviceCtx.dtype);
        const swigled = deviceCtx.alloc([1, intermediateSize], deviceCtx.dtype);
        const down = deviceCtx.alloc([1, hiddenSize], deviceCtx.dtype);

        this.gemmTransB(tokenHiddenPtr, expertGateUpPtr, gateUp.ptr, 1, intermediateSize * 2, hiddenSize);

        if (gateUpBias) {
          const expertGateUpBiasPtr = gateUpBias.ptr + BigInt(expertIdx * intermediateSize * 2 * 2);
          this.add(gateUp.ptr, expertGateUpBiasPtr, gateUp.ptr, intermediateSize * 2);
        }

        if (this.ctx.config.modelType === "gpt_oss") {
          this.cuda.gptOssActivation(gateUp.ptr, swigled.ptr, 1, intermediateSize, 1.702, 7.0);
        } else {
          this.cuda.moeSwiglu(gateUp.ptr, swigled.ptr, 1, intermediateSize);
        }

        this.gemmTransB(swigled.ptr, expertDownPtr, down.ptr, 1, hiddenSize, intermediateSize);

        if (downBias) {
          const expertDownBiasPtr = downBias.ptr + BigInt(expertIdx * hiddenSize * 2);
          this.add(down.ptr, expertDownBiasPtr, down.ptr, hiddenSize);
        }

        const outputTokenPtr = output.ptr + BigInt(tokenIdx * hiddenSize * 2);
        this.cuda.scaleAddBf16(down.ptr, outputTokenPtr, weight, hiddenSize);

        deviceCtx.free(gateUp);
        deviceCtx.free(swigled);
        deviceCtx.free(down);
      }
    }

    // Cleanup
    deviceCtx.free(expertIndices);
    deviceCtx.free(expertWeights);
    for (const ptr of dequantedGateUp.values()) {
      this.cuda.free(ptr);
    }
    for (const ptr of dequantedDown.values()) {
      this.cuda.free(ptr);
    }

    return output;
  }

  /**
   * Batched prefill - process multiple variable-length prompts together with TP support
   */
  async prefillBatch(
    inputTokens: number[][],
    kvStates: SequenceKVState[]
  ): Promise<BatchedForwardResult> {
    const batchSize = inputTokens.length;
    if (batchSize === 0) {
      return { logits: [] };
    }

    const { hiddenSize, numHiddenLayers, rmsNormEps, vocabSize } = this.ctx.config;

    // Calculate total tokens and query lengths
    const queryLengths = inputTokens.map(t => t.length);
    const totalTokens = queryLengths.reduce((a, b) => a + b, 0);

    // Track prefill tokens for profiling
    if (this.profilingEnabled) {
      this.prefillTokens += totalTokens;
    }

    // Positions for prefill: start at 0 for each sequence
    const positions = new Array(batchSize).fill(0);

    // Concatenate all input tokens
    const allInputs = new Int32Array(totalTokens);
    let offset = 0;
    for (const tokens of inputTokens) {
      allInputs.set(tokens, offset);
      offset += tokens.length;
    }

    // Track hidden states per device
    const hiddenStates: Map<number, Tensor> = new Map();

    // 1. Embedding lookup (replicated on all devices)
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();

      const inputPtr = this.cuda.malloc(totalTokens * 4);
      this.cuda.memcpyH2D(inputPtr, allInputs.buffer, allInputs.byteLength);

      const embedWeight = deviceCtx.getWeight("model.embed_tokens.weight");
      const hidden = deviceCtx.alloc([totalTokens, hiddenSize]);

      this.embedding(embedWeight.ptr, inputPtr, hidden.ptr, 1, totalTokens, vocabSize, hiddenSize);

      this.cuda.free(inputPtr);
      hiddenStates.set(deviceCtx.device, hidden);
    }

    this.ctx.synchronize();

    // 2. Process transformer layers
    for (let layer = 0; layer < numHiddenLayers; layer++) {
      this.forwardLayerPaged(hiddenStates, layer, kvStates, queryLengths, positions, true);
    }

    // 3. Final norm (replicated on all devices)
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      const hidden = hiddenStates.get(deviceCtx.device)!;
      const normWeight = deviceCtx.getWeight("model.norm.weight");
      const normed = deviceCtx.alloc([totalTokens, hiddenSize]);

      this.rmsnorm(hidden.ptr, normWeight.ptr, normed.ptr, 1, totalTokens, hiddenSize, rmsNormEps);

      deviceCtx.free(hidden);
      hiddenStates.set(deviceCtx.device, normed);
    }

    this.ctx.synchronize();

    // 4. Compute logits (only last position per sequence) on device 0
    const deviceCtx0 = this.ctx.devices[0];
    deviceCtx0.setActive();
    const normed = hiddenStates.get(0)!;

    const lmHeadWeight = this.ctx.config.tieWordEmbeddings
      ? deviceCtx0.getWeight("model.embed_tokens.weight")
      : deviceCtx0.getWeight("lm_head.weight");

    const logitsArr: Float32Array[] = [];

    let hiddenOffset = 0;
    for (let i = 0; i < batchSize; i++) {
      const seqLen = queryLengths[i];
      const lastPosOffset = (hiddenOffset + seqLen - 1) * hiddenSize * 2;

      const logits = deviceCtx0.alloc([1, vocabSize]);
      this.gemmTransB(normed.ptr + BigInt(lastPosOffset), lmHeadWeight.ptr, logits.ptr, 1, vocabSize, hiddenSize);

      logitsArr.push(logits.toArray());
      deviceCtx0.free(logits);
      hiddenOffset += seqLen;
    }

    // Cleanup on all devices
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      deviceCtx.free(hiddenStates.get(deviceCtx.device)!);
    }

    this.cuda.setDevice(0);
    return { logits: logitsArr };
  }

  /**
   * Batched decode - process one token per sequence with TP support
   */
  async decodeBatch(
    tokens: number[],
    kvStates: SequenceKVState[]
  ): Promise<BatchedForwardResult> {
    // If CUDA graphs enabled and batch size is small, try graph-accelerated path
    if (this.cudaGraphsEnabled && tokens.length <= 8) {
      return this.decodeBatchGraphed(tokens, kvStates);
    }
    return this.decodeBatchImpl(tokens, kvStates);
  }

  /**
   * Graph-accelerated decode for small batch sizes (returns logits).
   * Uses power-of-2 padding to minimize graph variants.
   *
   * Flow:
   * 1. Warmup phase (first 2 runs): Execute normally to stabilize memory pool
   * 2. Capture phase (3rd run): Allocate persistent buffers and capture graph
   * 3. Replay phase (subsequent runs): Update input buffers and launch graph
   */
  private async decodeBatchGraphed(
    tokens: number[],
    kvStates: SequenceKVState[]
  ): Promise<BatchedForwardResult> {
    const actualBatchSize = tokens.length;
    const paddedBatchSize = this.getPaddedBatchSize(actualBatchSize);

    // Check if we have a captured graph for this padded size
    const graphInfo = this.graphCache.get(paddedBatchSize);

    if (graphInfo) {
      // === GRAPH REPLAY PATH ===
      return this.decodeBatchReplay(tokens, kvStates, graphInfo, paddedBatchSize);
    }

    // Check if we have pre-allocated buffers (allocated on first warmup)
    let buffers = this.graphBuffersCache.get(paddedBatchSize);

    // Track warmup runs for this batch size
    const warmupCount = this.graphWarmupCount.get(paddedBatchSize) || 0;
    this.graphWarmupCount.set(paddedBatchSize, warmupCount + 1);

    // We need 5 warmup runs to stabilize memory pool, then capture on 6th run
    const WARMUP_THRESHOLD = 5;

    if (warmupCount === 0 && !buffers) {
      // === FIRST WARMUP: Allocate persistent buffers ===
      buffers = this.allocateGraphBuffers(paddedBatchSize);
      this.graphBuffersCache.set(paddedBatchSize, buffers);
    }

    if (warmupCount < WARMUP_THRESHOLD) {
      // === WARMUP RUN: Use persistent buffers but don't capture ===
      return this.decodeBatchWithBuffers(tokens, kvStates, buffers!, paddedBatchSize);
    }

    if (warmupCount === WARMUP_THRESHOLD) {
      // === CAPTURE RUN: Use persistent buffers and capture ===
      return this.decodeBatchCapture(tokens, kvStates, buffers!, paddedBatchSize);
    }

    // After capture, we should have a graph - but if capture failed, fall back
    return this.decodeBatchWithBuffers(tokens, kvStates, buffers!, paddedBatchSize);
  }

  /**
   * Allocate persistent buffers for CUDA graph capture.
   */
  private allocateGraphBuffers(paddedBatchSize: number): {
    inputIds: Map<number, bigint>;
    positions: Map<number, bigint>;
    blockTables: Map<number, bigint>;
    kvContextLens: Map<number, bigint>;
    attnContextLens: Map<number, bigint>;
    hiddenStates: Map<number, bigint>;
    outputLogits: Map<number, bigint>;
    outputTokens: Map<number, bigint>;
  } {
    const { hiddenSize, vocabSize } = this.ctx.config;

    const inputIds: Map<number, bigint> = new Map();
    const positions: Map<number, bigint> = new Map();
    const blockTables: Map<number, bigint> = new Map();
    const kvContextLens: Map<number, bigint> = new Map();
    const attnContextLens: Map<number, bigint> = new Map();
    const hiddenStates: Map<number, bigint> = new Map();

    // Allocate persistent buffers on each device
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      const device = deviceCtx.device;

      inputIds.set(device, this.cuda.malloc(paddedBatchSize * 4));
      positions.set(device, this.cuda.malloc(paddedBatchSize * 4));
      blockTables.set(device, this.cuda.malloc(paddedBatchSize * this.maxBlocksPerSeq * 4));
      kvContextLens.set(device, this.cuda.malloc(paddedBatchSize * 4));
      attnContextLens.set(device, this.cuda.malloc(paddedBatchSize * 4));
      hiddenStates.set(device, this.cuda.malloc(paddedBatchSize * hiddenSize * 2)); // bf16
    }

    // Output buffers on device 0
    this.ctx.devices[0].setActive();
    const outputLogits: Map<number, bigint> = new Map();
    const outputTokens: Map<number, bigint> = new Map();
    outputLogits.set(0, this.cuda.malloc(paddedBatchSize * vocabSize * 2)); // bf16
    outputTokens.set(0, this.cuda.malloc(paddedBatchSize * 4));

    // Pre-allocate MoE buffers for CUDA graph capture (avoids cudaMalloc during capture)
    if (this.isMoE) {
      const { intermediateSize, numExpertsPerToken } = this.ctx.config;
      const topK = numExpertsPerToken ?? 2;
      for (const deviceCtx of this.ctx.devices) {
        deviceCtx.setActive();
        // Pre-allocate gate_up and activated buffers
        this.ensureMoeBuffers(deviceCtx, paddedBatchSize, intermediateSize, topK);
        // Pre-allocate down accumulator buffer in CUDA kernel
        this.cuda.moePreallocateBuffers(paddedBatchSize, hiddenSize);
      }
    }

    this.cuda.synchronize();

    return { inputIds, positions, blockTables, kvContextLens, attnContextLens, hiddenStates, outputLogits, outputTokens };
  }

  /**
   * Run decode using pre-allocated buffers (warmup path - same as capture but no graph recording).
   */
  private async decodeBatchWithBuffers(
    tokens: number[],
    kvStates: SequenceKVState[],
    buffers: {
      inputIds: Map<number, bigint>;
      positions: Map<number, bigint>;
      blockTables: Map<number, bigint>;
      kvContextLens: Map<number, bigint>;
      attnContextLens: Map<number, bigint>;
      hiddenStates: Map<number, bigint>;
      outputLogits: Map<number, bigint>;
      outputTokens: Map<number, bigint>;
    },
    paddedBatchSize: number
  ): Promise<BatchedForwardResult> {
    const batchSize = tokens.length;
    const { hiddenSize, numHiddenLayers, rmsNormEps, vocabSize } = this.ctx.config;
    const kvPositions = kvStates.map(s => s.numTokens - 1);

    // Build padded input arrays
    const inputArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) inputArr[i] = tokens[i];

    const positionsArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) positionsArr[i] = kvPositions[i];

    const blockTablesArr = new Int32Array(paddedBatchSize * this.maxBlocksPerSeq);
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      const blockIds = kvStates[seqIdx].blockIds;
      for (let j = 0; j < blockIds.length; j++) {
        blockTablesArr[seqIdx * this.maxBlocksPerSeq + j] = blockIds[j];
      }
    }

    const kvContextLensArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) kvContextLensArr[i] = kvPositions[i];

    const attnContextLensArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) attnContextLensArr[i] = kvPositions[i] + 1;

    // Upload to persistent buffers
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      const device = deviceCtx.device;
      this.cuda.memcpyH2D(buffers.inputIds.get(device)!, inputArr.buffer, paddedBatchSize * 4);
      this.cuda.memcpyH2D(buffers.positions.get(device)!, positionsArr.buffer, paddedBatchSize * 4);
      this.cuda.memcpyH2D(buffers.blockTables.get(device)!, blockTablesArr.buffer, paddedBatchSize * this.maxBlocksPerSeq * 4);
      this.cuda.memcpyH2D(buffers.kvContextLens.get(device)!, kvContextLensArr.buffer, paddedBatchSize * 4);
      this.cuda.memcpyH2D(buffers.attnContextLens.get(device)!, attnContextLensArr.buffer, paddedBatchSize * 4);
    }

    // Track decode tokens for profiling
    if (this.profilingEnabled) {
      this.decodeTokens += batchSize;
    }

    // 1. Embedding lookup (replicated on all devices)
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      const embedWeight = deviceCtx.getWeight("model.embed_tokens.weight");
      const inputPtr = buffers.inputIds.get(deviceCtx.device)!;
      const hiddenPtr = buffers.hiddenStates.get(deviceCtx.device)!;
      this.embedding(embedWeight.ptr, inputPtr, hiddenPtr, 1, paddedBatchSize, vocabSize, hiddenSize);
    }

    // 2. Process transformer layers with persistent buffers
    const hiddenStates: Map<number, Tensor> = new Map();
    const positionsGpu: Map<number, Tensor> = new Map();

    for (const deviceCtx of this.ctx.devices) {
      const hidden = Tensor.fromPtr(buffers.hiddenStates.get(deviceCtx.device)!, [paddedBatchSize, hiddenSize], DType.BFloat16, deviceCtx.device);
      hiddenStates.set(deviceCtx.device, hidden);

      const posPtr = Tensor.fromPtr(buffers.positions.get(deviceCtx.device)!, [paddedBatchSize], DType.Int32, deviceCtx.device);
      positionsGpu.set(deviceCtx.device, posPtr);
    }

    const decodeBuffers = {
      blockTables: buffers.blockTables.get(0)!,
      kvContextLens: buffers.kvContextLens.get(0)!,
      attnContextLens: buffers.attnContextLens.get(0)!,
    };

    const queryLengths = new Array(paddedBatchSize).fill(1);
    const paddedPositions = Array.from(positionsArr);

    for (let layer = 0; layer < numHiddenLayers; layer++) {
      this.forwardLayerPaged(hiddenStates, layer, kvStates, queryLengths, paddedPositions, false, positionsGpu, decodeBuffers);
    }

    // 3. Final norm and lm_head on device 0
    const deviceCtx0 = this.ctx.devices[0];
    deviceCtx0.setActive();

    const normWeight = deviceCtx0.getWeight("model.norm.weight");
    const hidden = hiddenStates.get(0)!;
    const normed = deviceCtx0.alloc([paddedBatchSize, hiddenSize]);
    this.rmsnorm(hidden.ptr, normWeight.ptr, normed.ptr, 1, paddedBatchSize, hiddenSize, rmsNormEps);

    const lmHeadWeight = this.ctx.config.tieWordEmbeddings
      ? deviceCtx0.getWeight("model.embed_tokens.weight")
      : deviceCtx0.getWeight("lm_head.weight");

    const logitsPtr = buffers.outputLogits.get(0)!;
    this.gemmTransB(normed.ptr, lmHeadWeight.ptr, logitsPtr, paddedBatchSize, vocabSize, hiddenSize);

    deviceCtx0.free(normed);

    // Read logits (convert bf16 → float32)
    this.cuda.synchronize();
    const logitsTensor = Tensor.fromPtr(logitsPtr, [batchSize, vocabSize], DType.BFloat16, 0);
    const logitsData = logitsTensor.toArray();

    const logitsArr: Float32Array[] = [];
    for (let i = 0; i < batchSize; i++) {
      const seqLogits = new Float32Array(vocabSize);
      seqLogits.set(logitsData.subarray(i * vocabSize, (i + 1) * vocabSize));
      logitsArr.push(seqLogits);
    }

    return { logits: logitsArr };
  }

  /**
   * Capture a CUDA graph for decode (logits path).
   * Uses pre-allocated buffers and captures the forward pass into a graph.
   */
  private async decodeBatchCapture(
    tokens: number[],
    kvStates: SequenceKVState[],
    buffers: {
      inputIds: Map<number, bigint>;
      positions: Map<number, bigint>;
      blockTables: Map<number, bigint>;
      kvContextLens: Map<number, bigint>;
      attnContextLens: Map<number, bigint>;
      hiddenStates: Map<number, bigint>;
      outputLogits: Map<number, bigint>;
      outputTokens: Map<number, bigint>;
    },
    paddedBatchSize: number
  ): Promise<BatchedForwardResult> {
    const batchSize = tokens.length;
    const { hiddenSize, numHiddenLayers, rmsNormEps, vocabSize } = this.ctx.config;
    const kvPositions = kvStates.map(s => s.numTokens - 1);

    // Build padded input arrays
    const inputArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) inputArr[i] = tokens[i];

    const positionsArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) positionsArr[i] = kvPositions[i];

    const blockTablesArr = new Int32Array(paddedBatchSize * this.maxBlocksPerSeq);
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      const blockIds = kvStates[seqIdx].blockIds;
      for (let j = 0; j < blockIds.length; j++) {
        blockTablesArr[seqIdx * this.maxBlocksPerSeq + j] = blockIds[j];
      }
    }

    const kvContextLensArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) kvContextLensArr[i] = kvPositions[i];

    const attnContextLensArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) attnContextLensArr[i] = kvPositions[i] + 1;

    // Upload to persistent buffers
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      const device = deviceCtx.device;
      this.cuda.memcpyH2D(buffers.inputIds.get(device)!, inputArr.buffer, paddedBatchSize * 4);
      this.cuda.memcpyH2D(buffers.positions.get(device)!, positionsArr.buffer, paddedBatchSize * 4);
      this.cuda.memcpyH2D(buffers.blockTables.get(device)!, blockTablesArr.buffer, paddedBatchSize * this.maxBlocksPerSeq * 4);
      this.cuda.memcpyH2D(buffers.kvContextLens.get(device)!, kvContextLensArr.buffer, paddedBatchSize * 4);
      this.cuda.memcpyH2D(buffers.attnContextLens.get(device)!, attnContextLensArr.buffer, paddedBatchSize * 4);
    }

    this.cuda.synchronize();

    // Create graph and begin capture
    // Use MultiDeviceCudaGraph for tensor parallelism (TP > 1)
    const isMultiDevice = this.ctx.worldSize > 1;
    const graph = isMultiDevice
      ? new MultiDeviceCudaGraph(this.cuda, this.ctx.worldSize)
      : new CudaGraph(this.cuda);

    try {
      this.isCapturing = true;  // Skip synchronization during capture
      if (isMultiDevice) {
        this.capturingGraph = graph as MultiDeviceCudaGraph;
      }
      graph.beginCapture();

      // === CAPTURED FORWARD PASS ===

      // 1. Embedding lookup (replicated on all devices)
      for (const deviceCtx of this.ctx.devices) {
        this.switchDeviceForCapture(deviceCtx.device);
        deviceCtx.setActive();
        const embedWeight = deviceCtx.getWeight("model.embed_tokens.weight");
        const inputPtr = buffers.inputIds.get(deviceCtx.device)!;
        const hiddenPtr = buffers.hiddenStates.get(deviceCtx.device)!;
        this.embedding(embedWeight.ptr, inputPtr, hiddenPtr, 1, paddedBatchSize, vocabSize, hiddenSize);
      }

      // 2. Process transformer layers with persistent buffers
      const hiddenStates: Map<number, Tensor> = new Map();
      const positionsGpu: Map<number, Tensor> = new Map();

      for (const deviceCtx of this.ctx.devices) {
        const hidden = Tensor.fromPtr(buffers.hiddenStates.get(deviceCtx.device)!, [paddedBatchSize, hiddenSize], DType.BFloat16, deviceCtx.device);
        hiddenStates.set(deviceCtx.device, hidden);

        const posPtr = Tensor.fromPtr(buffers.positions.get(deviceCtx.device)!, [paddedBatchSize], DType.Int32, deviceCtx.device);
        positionsGpu.set(deviceCtx.device, posPtr);
      }

      const decodeBuffers = {
        blockTables: buffers.blockTables.get(0)!,
        kvContextLens: buffers.kvContextLens.get(0)!,
        attnContextLens: buffers.attnContextLens.get(0)!,
      };

      const queryLengths = new Array(paddedBatchSize).fill(1);
      const paddedPositions = Array.from(positionsArr);

      for (let layer = 0; layer < numHiddenLayers; layer++) {
        this.forwardLayerPaged(hiddenStates, layer, kvStates, queryLengths, paddedPositions, false, positionsGpu, decodeBuffers);
      }

      // 3. Final norm and lm_head on device 0
      this.switchDeviceForCapture(0);
      const deviceCtx0 = this.ctx.devices[0];
      deviceCtx0.setActive();

      const normWeight = deviceCtx0.getWeight("model.norm.weight");
      const hidden = hiddenStates.get(0)!;

      // Allocate temp buffer for normed output (reuses same address via memory pool)
      const normed = deviceCtx0.alloc([paddedBatchSize, hiddenSize]);
      this.rmsnorm(hidden.ptr, normWeight.ptr, normed.ptr, 1, paddedBatchSize, hiddenSize, rmsNormEps);

      const lmHeadWeight = this.ctx.config.tieWordEmbeddings
        ? deviceCtx0.getWeight("model.embed_tokens.weight")
        : deviceCtx0.getWeight("lm_head.weight");

      const logitsPtr = buffers.outputLogits.get(0)!;
      this.gemmTransB(normed.ptr, lmHeadWeight.ptr, logitsPtr, paddedBatchSize, vocabSize, hiddenSize);

      // Free intermediate normed buffer (returns to pool for graph stability)
      deviceCtx0.free(normed);

      // End capture
      graph.endCapture();
      this.isCapturing = false;
      this.capturingGraph = null;

      // Store graph with the pre-allocated buffers
      this.graphCache.set(paddedBatchSize, {
        graph,
        inputIds: buffers.inputIds,
        positions: buffers.positions,
        blockTables: buffers.blockTables,
        kvContextLens: buffers.kvContextLens,
        attnContextLens: buffers.attnContextLens,
        hiddenStates: buffers.hiddenStates,
        outputLogits: buffers.outputLogits,
        outputTokens: buffers.outputTokens,
      });

      // Launch graph to execute the captured operations and get valid output
      graph.launch();
      graph.sync();

      // Read logits from this capture run (convert bf16 → float32)
      const logitsTensor = Tensor.fromPtr(logitsPtr, [batchSize, vocabSize], DType.BFloat16, 0);
      const logitsData = logitsTensor.toArray();

      const logitsArr: Float32Array[] = [];
      for (let i = 0; i < batchSize; i++) {
        const seqLogits = new Float32Array(vocabSize);
        seqLogits.set(logitsData.subarray(i * vocabSize, (i + 1) * vocabSize));
        logitsArr.push(seqLogits);
      }

      return { logits: logitsArr };

    } catch (e) {
      this.isCapturing = false;
      this.capturingGraph = null;
      console.error(`[CUDA Graphs] Capture failed for batch size ${paddedBatchSize} (logits path):`, e);
      graph.destroy();

      // Fall back to normal execution
      return this.decodeBatchWithBuffers(tokens, kvStates, buffers, paddedBatchSize);
    }
  }

  /**
   * Replay a captured CUDA graph for decode (logits path).
   * Updates persistent input buffers with new data and launches the graph.
   */
  private async decodeBatchReplay(
    tokens: number[],
    kvStates: SequenceKVState[],
    graphInfo: {
      graph: CudaGraph;
      inputIds: Map<number, bigint>;
      positions: Map<number, bigint>;
      blockTables: Map<number, bigint>;
      kvContextLens: Map<number, bigint>;
      attnContextLens: Map<number, bigint>;
      hiddenStates: Map<number, bigint>;
      outputLogits: Map<number, bigint>;
      outputTokens: Map<number, bigint>;
    },
    paddedBatchSize: number
  ): Promise<BatchedForwardResult> {
    const batchSize = tokens.length;
    const { vocabSize } = this.ctx.config;
    const positions = kvStates.map(s => s.numTokens - 1);

    // Build padded input arrays
    const inputArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) inputArr[i] = tokens[i];

    const positionsArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) positionsArr[i] = positions[i];

    const blockTablesArr = new Int32Array(paddedBatchSize * this.maxBlocksPerSeq);
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      const blockIds = kvStates[seqIdx].blockIds;
      for (let j = 0; j < blockIds.length; j++) {
        blockTablesArr[seqIdx * this.maxBlocksPerSeq + j] = blockIds[j];
      }
    }

    const kvContextLensArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) kvContextLensArr[i] = positions[i];

    const attnContextLensArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) attnContextLensArr[i] = positions[i] + 1;

    // Update persistent input buffers on ALL devices (needed for multi-device graphs)
    // Each device does its own embedding lookup from its local inputIds buffer
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      const device = deviceCtx.device;
      this.cuda.memcpyH2D(graphInfo.inputIds.get(device)!, inputArr.buffer, paddedBatchSize * 4);
      this.cuda.memcpyH2D(graphInfo.positions.get(device)!, positionsArr.buffer, paddedBatchSize * 4);
    }
    // Block tables and context lens only need to be on device 0 (used by attention)
    this.ctx.devices[0].setActive();
    this.cuda.memcpyH2D(graphInfo.blockTables.get(0)!, blockTablesArr.buffer, paddedBatchSize * this.maxBlocksPerSeq * 4);
    this.cuda.memcpyH2D(graphInfo.kvContextLens.get(0)!, kvContextLensArr.buffer, paddedBatchSize * 4);
    this.cuda.memcpyH2D(graphInfo.attnContextLens.get(0)!, attnContextLensArr.buffer, paddedBatchSize * 4);

    // Track decode tokens for profiling
    if (this.profilingEnabled) {
      this.decodeTokens += batchSize;
    }

    // Launch the captured graph
    graphInfo.graph.launch();

    // Sync and read logits (convert bf16 → float32)
    // For multi-device graphs, use graph.sync() to synchronize all device streams
    graphInfo.graph.sync();
    const logitsTensor = Tensor.fromPtr(graphInfo.outputLogits.get(0)!, [batchSize, vocabSize], DType.BFloat16, 0);
    const logitsData = logitsTensor.toArray();

    const logitsArr: Float32Array[] = [];
    for (let i = 0; i < batchSize; i++) {
      const seqLogits = new Float32Array(vocabSize);
      seqLogits.set(logitsData.subarray(i * vocabSize, (i + 1) * vocabSize));
      logitsArr.push(seqLogits);
    }

    return { logits: logitsArr };
  }

  /**
   * Core decode implementation (non-graphed path, returns logits)
   */
  private async decodeBatchImpl(
    tokens: number[],
    kvStates: SequenceKVState[]
  ): Promise<BatchedForwardResult> {
    const batchSize = tokens.length;
    if (batchSize === 0) {
      return { logits: [] };
    }

    const { hiddenSize, numHiddenLayers, rmsNormEps, vocabSize } = this.ctx.config;

    const queryLengths = tokens.map(() => 1);
    const totalTokens = batchSize;

    // Track decode tokens for profiling
    if (this.profilingEnabled) {
      this.decodeTokens += totalTokens;
    }

    // Positions: numTokens - 1 because extendSequence has already been called
    const positions = kvStates.map(s => s.numTokens - 1);

    // Track hidden states per device
    const hiddenStates: Map<number, Tensor> = new Map();

    // 1. Embedding lookup (replicated on all devices)
    const inputArr = new Int32Array(tokens);
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();

      const inputPtr = this.cuda.malloc(totalTokens * 4);
      this.cuda.memcpyH2D(inputPtr, inputArr.buffer, inputArr.byteLength);

      const embedWeight = deviceCtx.getWeight("model.embed_tokens.weight");
      const hidden = deviceCtx.alloc([totalTokens, hiddenSize]);

      this.embedding(embedWeight.ptr, inputPtr, hidden.ptr, 1, totalTokens, vocabSize, hiddenSize);

      this.cuda.free(inputPtr);
      hiddenStates.set(deviceCtx.device, hidden);
    }

    this.ctx.synchronize();

    // Pre-upload positions for all devices (used across all layers in decode)
    // For CUDA graphs, we need this even for batchSize=1 to match the capture code path
    const positionsGpu: Map<number, Tensor> = new Map();
    if (this.isBf16 && (batchSize > 1 || this.cudaGraphsEnabled)) {
      const positionsArr = new Int32Array(positions);
      for (const deviceCtx of this.ctx.devices) {
        deviceCtx.setActive();
        const posPtr = deviceCtx.alloc([batchSize], DType.Int32);
        this.cuda.memcpyH2D(posPtr.ptr, positionsArr.buffer, positionsArr.byteLength);
        positionsGpu.set(deviceCtx.device, posPtr);
      }
    }

    // Pre-upload block tables and context lens for all layers (avoids 40x cudaMalloc per layer)
    // For CUDA graphs, we need this even for batchSize=1 to match the capture code path
    let decodeBuffers: { blockTables: bigint; kvContextLens: bigint; attnContextLens: bigint } | undefined;
    if (this.isBf16 && (batchSize > 1 || this.cudaGraphsEnabled)) {
      // Build block tables (same for all layers)
      const blockTablesArr = new Int32Array(batchSize * this.maxBlocksPerSeq);
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        const blockIds = kvStates[seqIdx].blockIds;
        for (let j = 0; j < blockIds.length; j++) {
          blockTablesArr[seqIdx * this.maxBlocksPerSeq + j] = blockIds[j];
        }
      }

      // Build context lens for KV update (positions before new token)
      const kvContextLensArr = new Int32Array(batchSize);
      for (let i = 0; i < batchSize; i++) {
        kvContextLensArr[i] = positions[i];
      }

      // Build context lens for attention (positions after new token)
      const attnContextLensArr = new Int32Array(batchSize);
      for (let i = 0; i < batchSize; i++) {
        attnContextLensArr[i] = positions[i] + 1;
      }

      // Upload to GPU
      const blockTablesSize = batchSize * this.maxBlocksPerSeq * 4;
      const contextLensSize = batchSize * 4;

      const blockTablesPtr = this.cuda.malloc(blockTablesSize);
      const kvContextLensPtr = this.cuda.malloc(contextLensSize);
      const attnContextLensPtr = this.cuda.malloc(contextLensSize);

      this.cuda.memcpyH2D(blockTablesPtr, blockTablesArr.buffer, blockTablesSize);
      this.cuda.memcpyH2D(kvContextLensPtr, kvContextLensArr.buffer, contextLensSize);
      this.cuda.memcpyH2D(attnContextLensPtr, attnContextLensArr.buffer, contextLensSize);

      decodeBuffers = {
        blockTables: blockTablesPtr,
        kvContextLens: kvContextLensPtr,
        attnContextLens: attnContextLensPtr,
      };
    }

    // 2. Process transformer layers
    for (let layer = 0; layer < numHiddenLayers; layer++) {
      this.forwardLayerPaged(hiddenStates, layer, kvStates, queryLengths, positions, false, positionsGpu, decodeBuffers);
    }

    // Free pre-uploaded decode buffers
    if (decodeBuffers) {
      this.cuda.free(decodeBuffers.blockTables);
      this.cuda.free(decodeBuffers.kvContextLens);
      this.cuda.free(decodeBuffers.attnContextLens);
    }

    // Free pre-uploaded positions
    for (const [device, posPtr] of positionsGpu) {
      const deviceCtx = this.ctx.devices.find(d => d.device === device)!;
      deviceCtx.setActive();
      deviceCtx.free(posPtr);
    }

    // 3. Final norm (replicated)
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      const hidden = hiddenStates.get(deviceCtx.device)!;
      const normWeight = deviceCtx.getWeight("model.norm.weight");
      const normed = deviceCtx.alloc([totalTokens, hiddenSize]);

      this.rmsnorm(hidden.ptr, normWeight.ptr, normed.ptr, 1, totalTokens, hiddenSize, rmsNormEps);

      deviceCtx.free(hidden);
      hiddenStates.set(deviceCtx.device, normed);
    }

    this.ctx.synchronize();

    // 4. Compute logits for all sequences on device 0
    const deviceCtx0 = this.ctx.devices[0];
    deviceCtx0.setActive();
    const normed = hiddenStates.get(0)!;

    const lmHeadWeight = this.ctx.config.tieWordEmbeddings
      ? deviceCtx0.getWeight("model.embed_tokens.weight")
      : deviceCtx0.getWeight("lm_head.weight");

    const allLogits = deviceCtx0.alloc([totalTokens, vocabSize]);
    this.gemmTransB(normed.ptr, lmHeadWeight.ptr, allLogits.ptr, totalTokens, vocabSize, hiddenSize);

    const logitsData = allLogits.toArray();
    const logitsArr: Float32Array[] = [];
    for (let i = 0; i < batchSize; i++) {
      const seqLogits = new Float32Array(vocabSize);
      seqLogits.set(logitsData.subarray(i * vocabSize, (i + 1) * vocabSize));
      logitsArr.push(seqLogits);
    }

    deviceCtx0.free(allLogits);

    // Cleanup on all devices
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      deviceCtx.free(hiddenStates.get(deviceCtx.device)!);
    }

    this.cuda.setDevice(0);
    return { logits: logitsArr };
  }

  /**
   * Batched decode with GPU greedy sampling - returns token IDs directly
   * Avoids expensive D2H copy of full logits tensor
   */
  async decodeBatchGreedy(
    tokens: number[],
    kvStates: SequenceKVState[]
  ): Promise<{ tokenIds: number[] }> {
    // If CUDA graphs enabled and batch size is small, try graph-accelerated path
    if (this.cudaGraphsEnabled && tokens.length <= 8) {
      return this.decodeBatchGreedyGraphed(tokens, kvStates);
    }
    return this.decodeBatchGreedyImpl(tokens, kvStates);
  }

  /**
   * Graph-accelerated decode for small batch sizes.
   * Uses power-of-2 padding to minimize graph variants.
   *
   * Flow:
   * 1. Warmup phase (first 2 runs): Execute normally to stabilize memory pool
   * 2. Capture phase (3rd run): Allocate persistent buffers and capture graph
   * 3. Replay phase (subsequent runs): Update input buffers and launch graph
   */
  private async decodeBatchGreedyGraphed(
    tokens: number[],
    kvStates: SequenceKVState[]
  ): Promise<{ tokenIds: number[] }> {
    const actualBatchSize = tokens.length;
    const paddedBatchSize = this.getPaddedBatchSize(actualBatchSize);

    // Check if we have a captured graph for this padded size
    const graphInfo = this.graphCache.get(paddedBatchSize);

    if (graphInfo) {
      // === GRAPH REPLAY PATH ===
      return this.decodeBatchGreedyReplay(tokens, kvStates, graphInfo, paddedBatchSize);
    }

    // Track warmup runs for this batch size
    const warmupCount = this.graphWarmupCount.get(paddedBatchSize) || 0;
    this.graphWarmupCount.set(paddedBatchSize, warmupCount + 1);

    // We need 5 warmup runs to stabilize memory pool, then capture on 6th run
    const WARMUP_THRESHOLD = 5;

    if (warmupCount < WARMUP_THRESHOLD) {
      // Warmup run - just execute normally to populate memory pool
      return this.decodeBatchGreedyImpl(tokens, kvStates);
    }

    if (warmupCount === WARMUP_THRESHOLD) {
      // === CAPTURE RUN ===
      return this.decodeBatchGreedyCapture(tokens, kvStates, paddedBatchSize);
    }

    // After capture, we should have a graph - but if capture failed, fall back
    return this.decodeBatchGreedyImpl(tokens, kvStates);
  }

  /**
   * Capture a CUDA graph for greedy decode.
   * Allocates persistent buffers and captures the forward pass into a graph.
   */
  private async decodeBatchGreedyCapture(
    tokens: number[],
    kvStates: SequenceKVState[],
    paddedBatchSize: number
  ): Promise<{ tokenIds: number[] }> {
    const batchSize = tokens.length;
    const { hiddenSize, numHiddenLayers, rmsNormEps, vocabSize } = this.ctx.config;
    const positions = kvStates.map(s => s.numTokens - 1);

    // Allocate PERSISTENT input buffers (per device) - these stay allocated for replay
    const persistentInputIds: Map<number, bigint> = new Map();
    const persistentPositions: Map<number, bigint> = new Map();
    const persistentBlockTables: Map<number, bigint> = new Map();
    const persistentKvContextLens: Map<number, bigint> = new Map();
    const persistentAttnContextLens: Map<number, bigint> = new Map();
    const persistentHidden: Map<number, bigint> = new Map();

    // Build input arrays with padding to paddedBatchSize
    const inputArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) inputArr[i] = tokens[i];

    const positionsArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) positionsArr[i] = positions[i];

    const blockTablesArr = new Int32Array(paddedBatchSize * this.maxBlocksPerSeq);
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      const blockIds = kvStates[seqIdx].blockIds;
      for (let j = 0; j < blockIds.length; j++) {
        blockTablesArr[seqIdx * this.maxBlocksPerSeq + j] = blockIds[j];
      }
    }

    const kvContextLensArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) kvContextLensArr[i] = positions[i];

    const attnContextLensArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) attnContextLensArr[i] = positions[i] + 1;

    // Allocate persistent buffers on each device
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      const device = deviceCtx.device;

      // Input IDs buffer
      const inputIdsPtr = this.cuda.malloc(paddedBatchSize * 4);
      this.cuda.memcpyH2D(inputIdsPtr, inputArr.buffer, paddedBatchSize * 4);
      persistentInputIds.set(device, inputIdsPtr);

      // Positions buffer
      const posPtr = this.cuda.malloc(paddedBatchSize * 4);
      this.cuda.memcpyH2D(posPtr, positionsArr.buffer, paddedBatchSize * 4);
      persistentPositions.set(device, posPtr);

      // Block tables buffer
      const blockTablesPtr = this.cuda.malloc(paddedBatchSize * this.maxBlocksPerSeq * 4);
      this.cuda.memcpyH2D(blockTablesPtr, blockTablesArr.buffer, paddedBatchSize * this.maxBlocksPerSeq * 4);
      persistentBlockTables.set(device, blockTablesPtr);

      // Context lens buffers
      const kvContextLensPtr = this.cuda.malloc(paddedBatchSize * 4);
      this.cuda.memcpyH2D(kvContextLensPtr, kvContextLensArr.buffer, paddedBatchSize * 4);
      persistentKvContextLens.set(device, kvContextLensPtr);

      const attnContextLensPtr = this.cuda.malloc(paddedBatchSize * 4);
      this.cuda.memcpyH2D(attnContextLensPtr, attnContextLensArr.buffer, paddedBatchSize * 4);
      persistentAttnContextLens.set(device, attnContextLensPtr);

      // Hidden states buffer (will be written by embedding)
      const hiddenPtr = this.cuda.malloc(paddedBatchSize * hiddenSize * 2); // bf16
      persistentHidden.set(device, hiddenPtr);
    }

    // Allocate persistent OUTPUT buffer on device 0
    const persistentOutputTokens: Map<number, bigint> = new Map();
    this.ctx.devices[0].setActive();
    const outputTokensPtr = this.cuda.malloc(paddedBatchSize * 4);
    persistentOutputTokens.set(0, outputTokensPtr);

    // Allocate persistent logits buffer on device 0 (for argmax input)
    const persistentLogits: Map<number, bigint> = new Map();
    const logitsPtr = this.cuda.malloc(paddedBatchSize * vocabSize * 2); // bf16
    persistentLogits.set(0, logitsPtr);

    this.cuda.synchronize();

    // Create graph and begin capture
    // Use MultiDeviceCudaGraph for tensor parallelism (TP > 1)
    const isMultiDevice = this.ctx.worldSize > 1;
    const graph = isMultiDevice
      ? new MultiDeviceCudaGraph(this.cuda, this.ctx.worldSize)
      : new CudaGraph(this.cuda);

    try {
      this.isCapturing = true;  // Skip synchronization during capture
      if (isMultiDevice) {
        this.capturingGraph = graph as MultiDeviceCudaGraph;
      }
      graph.beginCapture();

      // === CAPTURED FORWARD PASS ===

      // 1. Embedding lookup (replicated on all devices)
      for (const deviceCtx of this.ctx.devices) {
        this.switchDeviceForCapture(deviceCtx.device);
        deviceCtx.setActive();
        const embedWeight = deviceCtx.getWeight("model.embed_tokens.weight");
        const inputPtr = persistentInputIds.get(deviceCtx.device)!;
        const hiddenPtr = persistentHidden.get(deviceCtx.device)!;
        this.embedding(embedWeight.ptr, inputPtr, hiddenPtr, 1, paddedBatchSize, vocabSize, hiddenSize);
      }

      // 2. Process transformer layers with persistent buffers
      const hiddenStates: Map<number, Tensor> = new Map();
      const positionsGpu: Map<number, Tensor> = new Map();

      for (const deviceCtx of this.ctx.devices) {
        // Wrap persistent pointers in Tensor objects for the layer functions
        const hidden = Tensor.fromPtr(persistentHidden.get(deviceCtx.device)!, [paddedBatchSize, hiddenSize], DType.BFloat16, deviceCtx.device);
        hiddenStates.set(deviceCtx.device, hidden);

        const posPtr = Tensor.fromPtr(persistentPositions.get(deviceCtx.device)!, [paddedBatchSize], DType.Int32, deviceCtx.device);
        positionsGpu.set(deviceCtx.device, posPtr);
      }

      // Use device 0's persistent buffers for decode (all devices share same block tables during capture)
      const decodeBuffers = {
        blockTables: persistentBlockTables.get(0)!,
        kvContextLens: persistentKvContextLens.get(0)!,
        attnContextLens: persistentAttnContextLens.get(0)!,
      };

      const queryLengths = new Array(paddedBatchSize).fill(1);
      const paddedPositions = Array.from(positionsArr);

      for (let layer = 0; layer < numHiddenLayers; layer++) {
        this.forwardLayerPaged(hiddenStates, layer, kvStates, queryLengths, paddedPositions, false, positionsGpu, decodeBuffers);
      }

      // 3. Final norm and lm_head on device 0
      this.switchDeviceForCapture(0);
      const deviceCtx0 = this.ctx.devices[0];
      deviceCtx0.setActive();

      const normWeight = deviceCtx0.getWeight("model.norm.weight");
      const hidden = hiddenStates.get(0)!;

      // Allocate temp buffer for normed output (needed for graph - reuses same address via memory pool)
      const normed = deviceCtx0.alloc([paddedBatchSize, hiddenSize]);
      this.rmsnorm(hidden.ptr, normWeight.ptr, normed.ptr, 1, paddedBatchSize, hiddenSize, rmsNormEps);

      const lmHeadWeight = this.ctx.config.tieWordEmbeddings
        ? deviceCtx0.getWeight("model.embed_tokens.weight")
        : deviceCtx0.getWeight("lm_head.weight");

      this.gemmTransB(normed.ptr, lmHeadWeight.ptr, logitsPtr, paddedBatchSize, vocabSize, hiddenSize);

      // 4. GPU argmax
      this.cuda.argmaxBf16(logitsPtr, outputTokensPtr, paddedBatchSize, vocabSize);

      // Free intermediate normed buffer (returns to pool for graph stability)
      deviceCtx0.free(normed);

      // End capture
      graph.endCapture();
      this.isCapturing = false;
      this.capturingGraph = null;

      // Store graph and persistent buffers
      this.graphCache.set(paddedBatchSize, {
        graph,
        inputIds: persistentInputIds,
        positions: persistentPositions,
        blockTables: persistentBlockTables,
        kvContextLens: persistentKvContextLens,
        attnContextLens: persistentAttnContextLens,
        hiddenStates: persistentHidden,
        outputLogits: persistentLogits,
        outputTokens: persistentOutputTokens,
      });

      // Launch graph to execute the captured operations and get valid output
      graph.launch();
      graph.sync();

      // Read output tokens from this capture run
      const tokenIdsBuffer = new ArrayBuffer(batchSize * 4);
      this.cuda.memcpyD2H(tokenIdsBuffer, outputTokensPtr, batchSize * 4);
      const tokenIds = Array.from(new Int32Array(tokenIdsBuffer));

      return { tokenIds };

    } catch (e) {
      this.isCapturing = false;
      this.capturingGraph = null;
      console.error(`[CUDA Graphs] Capture failed for batch size ${paddedBatchSize}:`, e);

      // Cleanup persistent buffers on failure
      for (const [_, ptr] of persistentInputIds) this.cuda.free(ptr);
      for (const [_, ptr] of persistentPositions) this.cuda.free(ptr);
      for (const [_, ptr] of persistentBlockTables) this.cuda.free(ptr);
      for (const [_, ptr] of persistentKvContextLens) this.cuda.free(ptr);
      for (const [_, ptr] of persistentAttnContextLens) this.cuda.free(ptr);
      for (const [_, ptr] of persistentHidden) this.cuda.free(ptr);
      for (const [_, ptr] of persistentOutputTokens) this.cuda.free(ptr);
      for (const [_, ptr] of persistentLogits) this.cuda.free(ptr);
      graph.destroy();

      // Fall back to normal execution
      return this.decodeBatchGreedyImpl(tokens, kvStates);
    }
  }

  /**
   * Replay a captured CUDA graph for greedy decode.
   * Updates persistent input buffers with new data and launches the graph.
   */
  private async decodeBatchGreedyReplay(
    tokens: number[],
    kvStates: SequenceKVState[],
    graphInfo: {
      graph: CudaGraph;
      inputIds: Map<number, bigint>;
      positions: Map<number, bigint>;
      blockTables: Map<number, bigint>;
      kvContextLens: Map<number, bigint>;
      attnContextLens: Map<number, bigint>;
      hiddenStates: Map<number, bigint>;
      outputLogits: Map<number, bigint>;
      outputTokens: Map<number, bigint>;
    },
    paddedBatchSize: number
  ): Promise<{ tokenIds: number[] }> {
    const batchSize = tokens.length;
    const positions = kvStates.map(s => s.numTokens - 1);

    // Build padded input arrays
    const inputArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) inputArr[i] = tokens[i];

    const positionsArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) positionsArr[i] = positions[i];

    const blockTablesArr = new Int32Array(paddedBatchSize * this.maxBlocksPerSeq);
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      const blockIds = kvStates[seqIdx].blockIds;
      for (let j = 0; j < blockIds.length; j++) {
        blockTablesArr[seqIdx * this.maxBlocksPerSeq + j] = blockIds[j];
      }
    }

    const kvContextLensArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) kvContextLensArr[i] = positions[i];

    const attnContextLensArr = new Int32Array(paddedBatchSize);
    for (let i = 0; i < batchSize; i++) attnContextLensArr[i] = positions[i] + 1;

    // Update persistent input buffers on ALL devices (needed for multi-device graphs)
    // Each device does its own embedding lookup from its local inputIds buffer
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      const device = deviceCtx.device;
      this.cuda.memcpyH2D(graphInfo.inputIds.get(device)!, inputArr.buffer, paddedBatchSize * 4);
      this.cuda.memcpyH2D(graphInfo.positions.get(device)!, positionsArr.buffer, paddedBatchSize * 4);
    }
    // Block tables and context lens only need to be on device 0 (used by attention)
    this.ctx.devices[0].setActive();
    this.cuda.memcpyH2D(graphInfo.blockTables.get(0)!, blockTablesArr.buffer, paddedBatchSize * this.maxBlocksPerSeq * 4);
    this.cuda.memcpyH2D(graphInfo.kvContextLens.get(0)!, kvContextLensArr.buffer, paddedBatchSize * 4);
    this.cuda.memcpyH2D(graphInfo.attnContextLens.get(0)!, attnContextLensArr.buffer, paddedBatchSize * 4);

    // Track decode tokens for profiling
    if (this.profilingEnabled) {
      this.decodeTokens += batchSize;
    }

    // Launch the captured graph
    graphInfo.graph.launch();

    // Sync and read output
    // For multi-device graphs, use graph.sync() to synchronize all device streams
    graphInfo.graph.sync();
    const tokenIdsBuffer = new ArrayBuffer(batchSize * 4);
    this.cuda.memcpyD2H(tokenIdsBuffer, graphInfo.outputTokens.get(0)!, batchSize * 4);
    const tokenIds = Array.from(new Int32Array(tokenIdsBuffer));

    return { tokenIds };
  }

  /**
   * Core decode implementation (non-graphed path)
   */
  private async decodeBatchGreedyImpl(
    tokens: number[],
    kvStates: SequenceKVState[]
  ): Promise<{ tokenIds: number[] }> {
    const batchSize = tokens.length;
    if (batchSize === 0) {
      return { tokenIds: [] };
    }

    const { hiddenSize, numHiddenLayers, rmsNormEps, vocabSize } = this.ctx.config;

    const queryLengths = tokens.map(() => 1);
    const totalTokens = batchSize;

    // Track decode tokens for profiling
    if (this.profilingEnabled) {
      this.decodeTokens += totalTokens;
    }

    // Positions: numTokens - 1 because extendSequence has already been called
    const positions = kvStates.map(s => s.numTokens - 1);

    // Track hidden states per device
    const hiddenStates: Map<number, Tensor> = new Map();

    // 1. Embedding lookup (replicated on all devices)
    const inputArr = new Int32Array(tokens);
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();

      const inputPtr = this.cuda.malloc(totalTokens * 4);
      this.cuda.memcpyH2D(inputPtr, inputArr.buffer, inputArr.byteLength);

      const embedWeight = deviceCtx.getWeight("model.embed_tokens.weight");
      const hidden = deviceCtx.alloc([totalTokens, hiddenSize]);

      this.embedding(embedWeight.ptr, inputPtr, hidden.ptr, 1, totalTokens, vocabSize, hiddenSize);

      this.cuda.free(inputPtr);
      hiddenStates.set(deviceCtx.device, hidden);
    }

    this.ctx.synchronize();

    // Pre-upload positions for all devices (used across all layers in decode)
    // For CUDA graphs, we need this even for batchSize=1 to match the capture code path
    const positionsGpu: Map<number, Tensor> = new Map();
    if (this.isBf16 && (batchSize > 1 || this.cudaGraphsEnabled)) {
      const positionsArr = new Int32Array(positions);
      for (const deviceCtx of this.ctx.devices) {
        deviceCtx.setActive();
        const posPtr = deviceCtx.alloc([batchSize], DType.Int32);
        this.cuda.memcpyH2D(posPtr.ptr, positionsArr.buffer, positionsArr.byteLength);
        positionsGpu.set(deviceCtx.device, posPtr);
      }
    }

    // Pre-upload block tables and context lens for all layers (avoids 40x cudaMalloc per layer)
    // For CUDA graphs, we need this even for batchSize=1 to match the capture code path
    let decodeBuffers: { blockTables: bigint; kvContextLens: bigint; attnContextLens: bigint } | undefined;
    if (this.isBf16 && (batchSize > 1 || this.cudaGraphsEnabled)) {
      // Build block tables (same for all layers)
      const blockTablesArr = new Int32Array(batchSize * this.maxBlocksPerSeq);
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        const blockIds = kvStates[seqIdx].blockIds;
        for (let j = 0; j < blockIds.length; j++) {
          blockTablesArr[seqIdx * this.maxBlocksPerSeq + j] = blockIds[j];
        }
      }

      // Build context lens for KV update (positions before new token)
      const kvContextLensArr = new Int32Array(batchSize);
      for (let i = 0; i < batchSize; i++) {
        kvContextLensArr[i] = positions[i];
      }

      // Build context lens for attention (positions after new token)
      const attnContextLensArr = new Int32Array(batchSize);
      for (let i = 0; i < batchSize; i++) {
        attnContextLensArr[i] = positions[i] + 1;
      }

      // Upload to GPU
      const blockTablesSize = batchSize * this.maxBlocksPerSeq * 4;
      const contextLensSize = batchSize * 4;

      const blockTablesPtr = this.cuda.malloc(blockTablesSize);
      const kvContextLensPtr = this.cuda.malloc(contextLensSize);
      const attnContextLensPtr = this.cuda.malloc(contextLensSize);

      this.cuda.memcpyH2D(blockTablesPtr, blockTablesArr.buffer, blockTablesSize);
      this.cuda.memcpyH2D(kvContextLensPtr, kvContextLensArr.buffer, contextLensSize);
      this.cuda.memcpyH2D(attnContextLensPtr, attnContextLensArr.buffer, contextLensSize);

      decodeBuffers = {
        blockTables: blockTablesPtr,
        kvContextLens: kvContextLensPtr,
        attnContextLens: attnContextLensPtr,
      };
    }

    // 2. Process transformer layers
    for (let layer = 0; layer < numHiddenLayers; layer++) {
      this.forwardLayerPaged(hiddenStates, layer, kvStates, queryLengths, positions, false, positionsGpu, decodeBuffers);
    }

    // Free pre-uploaded decode buffers
    if (decodeBuffers) {
      this.cuda.free(decodeBuffers.blockTables);
      this.cuda.free(decodeBuffers.kvContextLens);
      this.cuda.free(decodeBuffers.attnContextLens);
    }

    // Free pre-uploaded positions
    for (const [device, posPtr] of positionsGpu) {
      const deviceCtx = this.ctx.devices.find(d => d.device === device)!;
      deviceCtx.setActive();
      deviceCtx.free(posPtr);
    }

    // 3. Final norm (replicated)
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      const hidden = hiddenStates.get(deviceCtx.device)!;
      const normWeight = deviceCtx.getWeight("model.norm.weight");
      const normed = deviceCtx.alloc([totalTokens, hiddenSize]);

      this.rmsnorm(hidden.ptr, normWeight.ptr, normed.ptr, 1, totalTokens, hiddenSize, rmsNormEps);

      deviceCtx.free(hidden);
      hiddenStates.set(deviceCtx.device, normed);
    }

    this.ctx.synchronize();

    // 4. Compute logits and sample on GPU
    const deviceCtx0 = this.ctx.devices[0];
    deviceCtx0.setActive();
    const normed = hiddenStates.get(0)!;

    const lmHeadWeight = this.ctx.config.tieWordEmbeddings
      ? deviceCtx0.getWeight("model.embed_tokens.weight")
      : deviceCtx0.getWeight("lm_head.weight");

    const allLogits = deviceCtx0.alloc([totalTokens, vocabSize]);
    this.gemmTransB(normed.ptr, lmHeadWeight.ptr, allLogits.ptr, totalTokens, vocabSize, hiddenSize);

    // GPU argmax instead of D2H copy
    const outputTokensPtr = this.cuda.malloc(batchSize * 4);  // int32 per token
    this.cuda.argmaxBf16(allLogits.ptr, outputTokensPtr, batchSize, vocabSize);

    // Small D2H copy: just batchSize * 4 bytes instead of batchSize * vocabSize * 2
    this.cuda.synchronize();
    const tokenIdsBuffer = new ArrayBuffer(batchSize * 4);
    this.cuda.memcpyD2H(tokenIdsBuffer, outputTokensPtr, batchSize * 4);
    const tokenIds = Array.from(new Int32Array(tokenIdsBuffer));

    this.cuda.free(outputTokensPtr);
    deviceCtx0.free(allLogits);

    // Cleanup on all devices
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      deviceCtx.free(hiddenStates.get(deviceCtx.device)!);
    }

    this.cuda.setDevice(0);
    return { tokenIds };
  }

  /**
   * Get KV pool stats
   */
  getStats(): { totalBlocks: number; freeBlocks: number; usedBlocks: number } {
    const pool0 = this.kvPools.get(0)!;
    const total = pool0.getTotalBlocks();
    const free = pool0.getNumFreeBlocks();
    return {
      totalBlocks: total,
      freeBlocks: free,
      usedBlocks: total - free,
    };
  }

  // ============================================================================
  // Compatibility methods for index.ts/chat interface (LegacyInferenceEngine API)
  // ============================================================================

  // Sequence ID counter for single-sequence generate()
  private nextSeqId = 1;

  /**
   * Reset engine state (no-op for paged KV - each sequence manages its own state)
   */
  reset(): void {
    // Paged attention doesn't need explicit reset - each sequence has its own KV state
  }

  /**
   * Enable profiling for timing and performance analysis
   */
  enableProfiling(): void {
    this.profilingEnabled = true;
    this.profiler.enable();
    this.flopsAccum = 0;
    this.bytesAccum = 0;
    this.prefillTokens = 0;
    this.decodeTokens = 0;
    // Enable MoE sub-operation profiling
    this.ctx.cuda.moeEnableProfiling(true);
  }

  /**
   * Disable profiling
   */
  disableProfiling(): void {
    this.profilingEnabled = false;
    this.profiler.disable();
    this.ctx.cuda.moeEnableProfiling(false);
  }

  /**
   * Enable CUDA graph capture for decode loop (experimental).
   * When enabled, the first decode step is captured and replayed for subsequent steps.
   * This reduces CPU overhead for kernel launches.
   * Supports multi-GPU tensor parallelism (TP > 1) using stream events for device handoffs.
   */
  enableCudaGraphs(): void {
    this.cudaGraphsEnabled = true;
    // Graphs will be captured lazily on first decode of each batch size
  }

  /**
   * Get the padded batch size (next power of 2) to reduce graph count.
   */
  private getPaddedBatchSize(batchSize: number): number {
    if (batchSize <= 1) return 1;
    if (batchSize <= 2) return 2;
    if (batchSize <= 4) return 4;
    if (batchSize <= 8) return 8;
    if (batchSize <= 16) return 16;
    if (batchSize <= 32) return 32;
    if (batchSize <= 64) return 64;
    return 128; // Max supported
  }

  /**
   * Get profiling metrics including timing breakdown
   */
  getProfileMetrics(): Map<string, number> {
    const metrics = this.profiler.collectMetrics();

    // Add FLOPs and bandwidth stats
    if (this.flopsAccum > 0) {
      metrics.set("__flops__", this.flopsAccum);
    }
    if (this.bytesAccum > 0) {
      metrics.set("__bytes__", this.bytesAccum);
    }
    if (this.prefillTokens > 0) {
      metrics.set("__prefill_tokens__", this.prefillTokens);
    }
    if (this.decodeTokens > 0) {
      metrics.set("__decode_tokens__", this.decodeTokens);
    }

    // Add MoE sub-operation timing
    const moeProfiling = this.ctx.cuda.moeGetProfiling();
    if (moeProfiling.callCount > 0) {
      metrics.set("__moe_gate_up_ms__", moeProfiling.gateUpMs);
      metrics.set("__moe_activation_ms__", moeProfiling.activationMs);
      metrics.set("__moe_down_ms__", moeProfiling.downMs);
      metrics.set("__moe_call_count__", moeProfiling.callCount);
    }

    // Reset accumulators
    this.flopsAccum = 0;
    this.bytesAccum = 0;
    this.prefillTokens = 0;
    this.decodeTokens = 0;

    return metrics;
  }

  /**
   * Calculate FLOPs for a GEMM operation: 2 * M * N * K
   */
  private gemmFlops(M: number, N: number, K: number): number {
    return 2 * M * N * K;
  }

  /**
   * Calculate memory bytes for a GEMM: reading A, B, writing C
   * A: M x K, B: K x N, C: M x N (all in fp16/bf16 = 2 bytes)
   */
  private gemmBytes(M: number, N: number, K: number): number {
    return 2 * (M * K + K * N + M * N);
  }

  /**
   * Track FLOPs and bytes for profiling
   */
  private trackOp(flops: number, bytes: number): void {
    if (this.profilingEnabled) {
      this.flopsAccum += flops;
      this.bytesAccum += bytes;
    }
  }

  /**
   * Generate tokens from a prompt (compatibility wrapper for single-sequence generation).
   * Uses the batched APIs internally with a single-sequence batch.
   */
  async generate(
    inputIds: number[],
    maxNewTokens: number,
    onToken?: (token: number) => void | boolean,
    samplingParams?: Partial<SamplingParams>
  ): Promise<number[]> {
    const seqId = this.nextSeqId++;
    const generatedTokens: number[] = [];
    const sampler = new Sampler(samplingParams ?? { temperature: 0 });

    // Allocate KV state for this sequence
    const kvState = this.allocateSequence(seqId, inputIds.length + maxNewTokens);
    if (!kvState) {
      throw new Error("Failed to allocate KV cache for sequence");
    }

    try {
      // Prefill phase
      const prefillResult = await this.prefillBatch([inputIds], [kvState]);

      // Update numTokens after prefill
      kvState.numTokens = inputIds.length;

      // Sample first token
      let nextToken = sampler.sample(prefillResult.logits[0], generatedTokens);
      generatedTokens.push(nextToken);

      // Check if callback requests stop
      if (onToken && onToken(nextToken) === false) {
        return generatedTokens;
      }

      // Check for EOS
      if (nextToken === this.ctx.config.eosTokenId) {
        return generatedTokens;
      }

      // Decode loop
      for (let i = 1; i < maxNewTokens; i++) {
        // Yield to event loop periodically to allow signal processing
        if (i % 4 === 0) {
          await new Promise(resolve => setImmediate(resolve));
        }

        // Extend sequence for the new token (must be called before decodeBatch)
        this.extendSequence(kvState, 1);

        // Decode step
        const decodeResult = await this.decodeBatch([nextToken], [kvState]);

        // Sample next token
        nextToken = sampler.sample(decodeResult.logits[0], generatedTokens);
        generatedTokens.push(nextToken);

        // Check if callback requests stop
        if (onToken && onToken(nextToken) === false) {
          break;
        }

        // Check for EOS
        if (nextToken === this.ctx.config.eosTokenId) {
          break;
        }
      }

      return generatedTokens;
    } finally {
      // Always free the KV state
      this.freeSequence(kvState);
    }
  }

  /**
   * Dispose engine resources
   */
  dispose(): void {
    for (const pool of this.kvPools.values()) {
      pool.dispose();
    }
    for (const [device, cache] of this.ropeCaches) {
      this.cuda.setDevice(device);
      this.cuda.free(cache.cos);
      this.cuda.free(cache.sin);
    }
    this.cuda.setDevice(0);
  }
}

/**
 * Create a batched inference engine
 */
export function createBatchedEngine(
  ctx: TensorParallelContext,
  options: {
    maxKvMemoryGB?: number;
    maxSeqLen?: number;
    maxBatchSize?: number;
    quiet?: boolean;
    cudaGraphs?: boolean;
  } = {}
): BatchedInferenceEngine {
  const engine = new BatchedInferenceEngine(
    ctx,
    options.maxKvMemoryGB ?? 10,
    options.maxSeqLen ?? 4096,
    options.maxBatchSize ?? 64,
    options.quiet ?? false
  );
  if (options.cudaGraphs) {
    engine.enableCudaGraphs();
  }
  return engine;
}
