// Continuous Batching Scheduler
// Dynamically batches requests for maximum throughput

import { Request, RequestStatus, RequestQueue, generateRequestId } from "./request";
import { PagedKVCache, PagedKVCacheManager, BLOCK_SIZE } from "../kv/paged";
import { LlamaConfig } from "../model/config";
import { Tensor, DType } from "../tensor/tensor";
import { getCudaBackend, CudaBackend } from "../backend/cuda/bindings";
import { Sampler, SamplingParams } from "../engine/sampler";

export interface SchedulerConfig {
  maxBatchSize: number;          // Maximum sequences in a batch
  maxTokensPerBatch: number;     // Maximum total tokens per iteration
  maxSeqLen: number;             // Maximum sequence length
  kvCacheMemoryGB: number;       // GPU memory for KV cache
}

export const DEFAULT_SCHEDULER_CONFIG: SchedulerConfig = {
  maxBatchSize: 64,
  maxTokensPerBatch: 4096,
  maxSeqLen: 4096,
  kvCacheMemoryGB: 10,
};

interface ScheduledRequest {
  request: Request;
  kvCache: PagedKVCache;
}

/**
 * Continuous batching scheduler.
 * Manages request queue, KV cache allocation, and batch formation.
 */
export class Scheduler {
  private config: SchedulerConfig;
  private modelConfig: LlamaConfig;
  private kvManager: PagedKVCacheManager;
  private cuda: CudaBackend;

  private pendingQueue: RequestQueue;
  private runningRequests: Map<string, ScheduledRequest>;

  private sampler: Sampler;
  private isRunning: boolean = false;

  constructor(
    modelConfig: LlamaConfig,
    schedulerConfig: Partial<SchedulerConfig> = {}
  ) {
    this.config = { ...DEFAULT_SCHEDULER_CONFIG, ...schedulerConfig };
    this.modelConfig = modelConfig;
    this.cuda = getCudaBackend();

    // Initialize KV cache manager
    this.kvManager = new PagedKVCacheManager(
      modelConfig.numHiddenLayers,
      modelConfig.numKeyValueHeads,
      modelConfig.headDim,
      this.config.kvCacheMemoryGB
    );

    this.pendingQueue = new RequestQueue();
    this.runningRequests = new Map();

    this.sampler = new Sampler({});
  }

  /**
   * Submit a new request for generation.
   */
  submitRequest(
    prompt: string,
    promptTokenIds: number[],
    params: Partial<SamplingParams & { maxNewTokens?: number; stopTokenIds?: number[] }> = {}
  ): Request {
    const request = new Request(
      generateRequestId(),
      prompt,
      promptTokenIds,
      {
        maxNewTokens: params.maxNewTokens ?? 256,
        temperature: params.temperature ?? 0.7,
        topK: params.topK ?? 50,
        topP: params.topP ?? 0.9,
        stopTokenIds: params.stopTokenIds ?? [this.modelConfig.eosTokenId],
      }
    );

    this.pendingQueue.add(request);
    return request;
  }

  /**
   * Try to schedule pending requests into the running batch.
   */
  private scheduleNewRequests(): void {
    while (
      !this.pendingQueue.isEmpty() &&
      this.runningRequests.size < this.config.maxBatchSize
    ) {
      const request = this.pendingQueue.peek()!;

      // Check if we have enough memory for this request
      const blocksNeeded = Math.ceil(
        (request.promptTokenIds.length + request.params.maxNewTokens) / BLOCK_SIZE
      );

      if (!this.kvManager.canAllocate(blocksNeeded * BLOCK_SIZE)) {
        // Not enough memory, try to preempt or wait
        break;
      }

      // Remove from pending queue
      this.pendingQueue.pop();

      // Allocate KV cache
      const kvCache = this.kvManager.createCache();
      const allocated = kvCache.allocateTokens(request.promptTokenIds.length);

      if (!allocated) {
        console.warn(`Failed to allocate KV cache for request ${request.id}`);
        request.markFailed("Failed to allocate KV cache");
        continue;
      }

      request.status = RequestStatus.RUNNING;
      this.runningRequests.set(request.id, { request, kvCache });
    }
  }

  /**
   * Get requests that need prefill (haven't started yet).
   */
  getPrefillRequests(): ScheduledRequest[] {
    return Array.from(this.runningRequests.values()).filter(
      (sr) => !sr.request.prefillComplete
    );
  }

  /**
   * Get requests that need decode (prefill complete, still generating).
   */
  getDecodeRequests(): ScheduledRequest[] {
    return Array.from(this.runningRequests.values()).filter(
      (sr) => sr.request.prefillComplete && !sr.request.isFinished()
    );
  }

  /**
   * Remove completed requests and free their resources.
   */
  private cleanupCompletedRequests(): Request[] {
    const completed: Request[] = [];

    for (const [id, sr] of this.runningRequests) {
      if (sr.request.isFinished()) {
        this.kvManager.removeCache(sr.kvCache.sequenceId);
        this.runningRequests.delete(id);
        completed.push(sr.request);
      }
    }

    return completed;
  }

  /**
   * Check if a request should finish (stop token or max length).
   */
  checkRequestCompletion(request: Request): void {
    if (request.hasReachedMaxTokens() || request.hasStopToken()) {
      request.markCompleted();
    }
  }

  /**
   * Get batch data for prefill phase.
   * Returns input tensors for batched prefill.
   */
  preparePrefillBatch(requests: ScheduledRequest[]): {
    inputIds: Tensor;
    positions: Tensor;
    blockTables: Int32Array;
    contextLens: Int32Array;
    seqLens: number[];
    maxSeqLen: number;
    maxBlocksPerSeq: number;
  } | null {
    if (requests.length === 0) return null;

    const numSeqs = requests.length;
    const seqLens = requests.map((sr) => sr.request.promptTokenIds.length);
    const maxSeqLen = Math.max(...seqLens);
    const maxBlocksPerSeq = Math.ceil(maxSeqLen / BLOCK_SIZE);

    // Flatten all input IDs (padded to max length)
    const totalTokens = seqLens.reduce((a, b) => a + b, 0);
    const inputIdsArray = new Int32Array(totalTokens);
    const positionsArray = new Int32Array(totalTokens);

    let offset = 0;
    for (let i = 0; i < numSeqs; i++) {
      const tokenIds = requests[i].request.promptTokenIds;
      for (let j = 0; j < tokenIds.length; j++) {
        inputIdsArray[offset] = tokenIds[j];
        positionsArray[offset] = j;
        offset++;
      }
    }

    // Build block tables
    const blockTables = new Int32Array(numSeqs * maxBlocksPerSeq);
    const contextLens = new Int32Array(numSeqs);

    for (let i = 0; i < numSeqs; i++) {
      const sr = requests[i];
      const table = sr.kvCache.getBlockTableArray(maxBlocksPerSeq);
      blockTables.set(table, i * maxBlocksPerSeq);
      contextLens[i] = sr.kvCache.seqLen;
    }

    return {
      inputIds: Tensor.fromArray(inputIdsArray, [totalTokens], DType.Int32),
      positions: Tensor.fromArray(positionsArray, [totalTokens], DType.Int32),
      blockTables,
      contextLens,
      seqLens,
      maxSeqLen,
      maxBlocksPerSeq,
    };
  }

  /**
   * Get batch data for decode phase.
   * Each sequence generates one token.
   */
  prepareDecodeBatch(requests: ScheduledRequest[]): {
    inputIds: Tensor;
    positions: Tensor;
    blockTables: Int32Array;
    contextLens: Int32Array;
    maxBlocksPerSeq: number;
  } | null {
    if (requests.length === 0) return null;

    const numSeqs = requests.length;

    // Find max context length to determine max blocks
    let maxContextLen = 0;
    for (const sr of requests) {
      maxContextLen = Math.max(maxContextLen, sr.request.totalTokens);
    }
    const maxBlocksPerSeq = Math.ceil(maxContextLen / BLOCK_SIZE);

    // Each sequence inputs its last generated token
    const inputIdsArray = new Int32Array(numSeqs);
    const positionsArray = new Int32Array(numSeqs);

    for (let i = 0; i < numSeqs; i++) {
      const req = requests[i].request;
      const lastToken = req.generatedTokenIds.length > 0
        ? req.generatedTokenIds[req.generatedTokenIds.length - 1]
        : req.promptTokenIds[req.promptTokenIds.length - 1];
      inputIdsArray[i] = lastToken;
      positionsArray[i] = req.totalTokens - 1;
    }

    // Build block tables
    const blockTables = new Int32Array(numSeqs * maxBlocksPerSeq);
    const contextLens = new Int32Array(numSeqs);

    for (let i = 0; i < numSeqs; i++) {
      const sr = requests[i];
      const table = sr.kvCache.getBlockTableArray(maxBlocksPerSeq);
      blockTables.set(table, i * maxBlocksPerSeq);
      contextLens[i] = sr.request.totalTokens;
    }

    return {
      inputIds: Tensor.fromArray(inputIdsArray, [numSeqs, 1], DType.Int32),
      positions: Tensor.fromArray(positionsArray, [numSeqs], DType.Int32),
      blockTables,
      contextLens,
      maxBlocksPerSeq,
    };
  }

  /**
   * Allocate blocks for new tokens after decode.
   */
  allocateNewTokens(requests: ScheduledRequest[]): boolean {
    for (const sr of requests) {
      if (!sr.kvCache.allocateTokens(1)) {
        console.warn(`Failed to allocate token for request ${sr.request.id}`);
        sr.request.markFailed("Out of memory");
        return false;
      }
    }
    return true;
  }

  /**
   * Run one scheduler iteration.
   * Returns completed requests.
   */
  step(): {
    prefillBatch: ScheduledRequest[] | null;
    decodeBatch: ScheduledRequest[] | null;
    completed: Request[];
  } {
    // Clean up finished requests
    const completed = this.cleanupCompletedRequests();

    // Schedule new requests
    this.scheduleNewRequests();

    // Get prefill and decode batches
    const prefillBatch = this.getPrefillRequests();
    const decodeBatch = this.getDecodeRequests();

    return {
      prefillBatch: prefillBatch.length > 0 ? prefillBatch : null,
      decodeBatch: decodeBatch.length > 0 ? decodeBatch : null,
      completed,
    };
  }

  /**
   * Mark prefill requests as complete after processing.
   */
  completePrefill(requests: ScheduledRequest[]): void {
    for (const sr of requests) {
      sr.request.markPrefillComplete();
    }
  }

  /**
   * Get scheduler stats.
   */
  getStats(): {
    pendingRequests: number;
    runningRequests: number;
    kvCacheStats: { totalBlocks: number; freeBlocks: number; usedBlocks: number };
  } {
    return {
      pendingRequests: this.pendingQueue.length,
      runningRequests: this.runningRequests.size,
      kvCacheStats: this.kvManager.getStats(),
    };
  }

  /**
   * Check if scheduler has work to do.
   */
  hasWork(): boolean {
    return !this.pendingQueue.isEmpty() || this.runningRequests.size > 0;
  }

  /**
   * Dispose all resources.
   */
  dispose(): void {
    this.kvManager.dispose();
    this.runningRequests.clear();
  }
}
