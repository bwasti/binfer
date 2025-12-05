// Legacy Inference Engine (DEPRECATED)
// Use BatchedInferenceEngine instead for production workloads.
// This engine is kept for backwards compatibility but does not support continuous batching.
// Unified engine supporting single-GPU (TP=1) and multi-GPU (TP>1) tensor parallelism

import { Tensor, DType } from "../tensor/tensor";
import { LlamaConfig } from "../model/config";
import { KVCache, createKVCache } from "../kv/manager";
import { Sampler, SamplingParams, StoppingCriteria, shouldStop } from "./sampler";
import { getCudaBackend, CudaBackend } from "../backend/cuda/bindings";
import { getFA3Backend, FA3Backend } from "../backend/cuda/fa3_bindings";
import { TensorParallelContext } from "./tp_context";
import { DeviceContext } from "./device_context";
import { CudaProfiler, getProfiler } from "../profiler/profiler";
import { GpuTracer, getTracer } from "../profiler/tracer";

export interface GenerationConfig {
  maxNewTokens: number;
  sampling: Partial<SamplingParams>;
  streamCallback?: (token: number, text?: string) => void;
}

export const DEFAULT_GENERATION_CONFIG: GenerationConfig = {
  maxNewTokens: 256,
  sampling: {
    temperature: 0.7,
    topK: 50,
    topP: 0.9,
  },
};

/**
 * Legacy Inference Engine for LLM generation.
 * @deprecated Use BatchedInferenceEngine instead for production workloads.
 *
 * Supports both single-GPU and multi-GPU tensor parallelism.
 * For TP=1, all-reduce operations are no-ops.
 * For TP>1, uses NCCL for inter-GPU communication.
 */
export class LegacyInferenceEngine {
  private ctx: TensorParallelContext;
  private cuda: CudaBackend;
  private fa3: FA3Backend;
  private profiler: CudaProfiler;
  private tracer: GpuTracer;

  constructor(ctx: TensorParallelContext) {
    this.ctx = ctx;
    this.cuda = getCudaBackend();
    this.fa3 = getFA3Backend();
    this.profiler = getProfiler();
    this.tracer = getTracer();

  }

  get config(): LlamaConfig {
    return this.ctx.config;
  }

  get worldSize(): number {
    return this.ctx.worldSize;
  }

  /**
   * Enable profiling for this engine.
   */
  enableProfiling(): void {
    this.profiler.enable();
  }

  /**
   * Disable profiling.
   */
  disableProfiling(): void {
    this.profiler.disable();
  }

  /**
   * Get profiling metrics and reset.
   */
  getProfileMetrics(): Map<string, number> {
    return this.profiler.collectMetrics();
  }

  /**
   * Enable tracing for detailed timeline analysis.
   */
  enableTracing(): void {
    this.tracer.enable();
  }

  /**
   * Disable tracing.
   */
  disableTracing(): void {
    this.tracer.disable();
  }

  /**
   * Get the tracer for trace finalization.
   */
  getTracer(): GpuTracer {
    return this.tracer;
  }

  /**
   * Reset the engine state for a new generation.
   * Clears KV caches so the next generate() call starts fresh.
   */
  reset(): void {
    for (const deviceCtx of this.ctx.devices) {
      if (deviceCtx.kvCache) {
        deviceCtx.kvCache.reset();
      }
    }
  }

  // ============================================================================
  // Dtype-aware operation dispatchers
  // These call either F16 or BF16 variants based on config.dtype
  // ============================================================================

  private get isBf16(): boolean {
    return this.ctx.config.dtype === "bfloat16";
  }

  private gemmTransB(A: bigint, B: bigint, C: bigint, M: number, N: number, K: number): void {
    if (this.isBf16) {
      this.cuda.gemmBf16TransB(A, B, C, M, N, K);
    } else {
      this.cuda.gemmF16TransB(A, B, C, M, N, K);
    }
  }

  private gemm(A: bigint, B: bigint, C: bigint, M: number, N: number, K: number): void {
    if (this.isBf16) {
      this.cuda.gemmBf16(A, B, C, M, N, K);
    } else {
      this.cuda.gemmF16(A, B, C, M, N, K);
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

  private scaleAdd(input: bigint, output: bigint, scale: number, numel: number): void {
    // output = output + scale * input
    if (this.isBf16) {
      this.cuda.scaleAddBf16(input, output, scale, numel);
    } else {
      // For FP16, fall back to a temporary buffer approach
      // (We only have BF16 version, but MoE models are typically BF16)
      this.cuda.scaleAddBf16(input, output, scale, numel);
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

  private softmax(
    input: bigint, output: bigint, batchSize: number, seqLen: number,
    vocabSize: number, temperature: number
  ): void {
    if (this.isBf16) {
      this.cuda.softmaxBf16(input, output, batchSize, seqLen, vocabSize, temperature);
    } else {
      this.cuda.softmaxF16(input, output, batchSize, seqLen, vocabSize, temperature);
    }
  }

  private flashAttention(
    Q: bigint, K: bigint, V: bigint, O: bigint,
    batchSize: number, seqQ: number, seqKV: number, kvStride: number, qOffset: number,
    numHeads: number, numKvHeads: number, headDim: number, softmaxScale: number, isCausal: boolean
  ): void {
    // FA3 integration is WIP - disabled for now due to param complexity
    // FA3 requires proper TMA setup and scheduler configuration
    // For now, use custom BF16/FP16 flash attention kernels

    // Custom flash attention kernels (support cached KV with stride)
    if (this.isBf16) {
      this.cuda.flashAttentionBf16(Q, K, V, O, batchSize, seqQ, seqKV, kvStride, qOffset,
        numHeads, numKvHeads, headDim, softmaxScale, isCausal);
    } else {
      this.cuda.flashAttentionF16(Q, K, V, O, batchSize, seqQ, seqKV, kvStride, qOffset,
        numHeads, numKvHeads, headDim, softmaxScale, isCausal);
    }
  }

  /**
   * Prefill phase - process prompt tokens on all GPUs.
   * Returns hidden states for the last position (used for logits computation).
   */
  private prefill(inputIds: number[]): Map<number, Tensor> {
    const seqLen = inputIds.length;
    const { hiddenSize, numHiddenLayers, rmsNormEps } = this.ctx.config;

    const inputArray = new Int32Array(inputIds);
    const hiddenStates: Map<number, Tensor> = new Map();

    // 1. Embedding lookup (replicated on all devices)
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();

      // Allocate input tensor
      const inputSize = seqLen * 4; // int32 = 4 bytes
      const inputPtr = this.cuda.malloc(inputSize);
      this.cuda.memcpyH2D(inputPtr, inputArray.buffer, inputSize);

      const embedWeight = deviceCtx.getWeight("model.embed_tokens.weight");
      const hidden = deviceCtx.alloc([1, seqLen, hiddenSize]);

      this.embedding(
        embedWeight.ptr,
        inputPtr,
        hidden.ptr,
        1,
        seqLen,
        this.ctx.config.vocabSize,
        hiddenSize
      );

      this.cuda.free(inputPtr);
      hiddenStates.set(deviceCtx.device, hidden);
    }

    this.ctx.synchronize();

    // 2. Process transformer layers
    for (let layer = 0; layer < numHiddenLayers; layer++) {
      this.forwardLayer(hiddenStates, layer, seqLen, true, 0);
    }

    // Advance KV cache positions after all layers
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.kvCache!.advanceSeqLen(seqLen);
    }

    // 3. Final norm (replicated)
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      const hidden = hiddenStates.get(deviceCtx.device)!;
      const normWeight = deviceCtx.getWeight("model.norm.weight");
      const normed = deviceCtx.alloc([1, seqLen, hiddenSize]);

      this.rmsnorm(
        hidden.ptr,
        normWeight.ptr,
        normed.ptr,
        1,
        seqLen,
        hiddenSize,
        rmsNormEps
      );

      deviceCtx.free(hidden);
      hiddenStates.set(deviceCtx.device, normed);
    }

    this.ctx.synchronize();
    return hiddenStates;
  }

  /**
   * Decode step - generate one token.
   */
  private decode(tokenId: number, position: number): Map<number, Tensor> {
    const { hiddenSize, numHiddenLayers, rmsNormEps } = this.ctx.config;

    const inputArray = new Int32Array([tokenId]);
    const hiddenStates: Map<number, Tensor> = new Map();

    // 1. Embedding lookup (replicated)
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();

      const inputSize = 4; // single int32
      const inputPtr = this.cuda.malloc(inputSize);
      this.cuda.memcpyH2D(inputPtr, inputArray.buffer, inputSize);

      const embedWeight = deviceCtx.getWeight("model.embed_tokens.weight");
      const hidden = deviceCtx.alloc([1, 1, hiddenSize]);

      this.embedding(
        embedWeight.ptr,
        inputPtr,
        hidden.ptr,
        1,
        1,
        this.ctx.config.vocabSize,
        hiddenSize
      );

      this.cuda.free(inputPtr);
      hiddenStates.set(deviceCtx.device, hidden);
    }

    this.ctx.synchronize();

    // 2. Process transformer layers
    for (let layer = 0; layer < numHiddenLayers; layer++) {
      this.forwardLayer(hiddenStates, layer, 1, false, position);
    }

    // Advance KV cache positions after all layers
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.kvCache!.advanceSeqLen(1);
    }

    // 3. Final norm (replicated)
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      const hidden = hiddenStates.get(deviceCtx.device)!;
      const normWeight = deviceCtx.getWeight("model.norm.weight");
      const normed = deviceCtx.alloc([1, 1, hiddenSize]);

      this.rmsnorm(
        hidden.ptr,
        normWeight.ptr,
        normed.ptr,
        1,
        1,
        hiddenSize,
        rmsNormEps
      );

      deviceCtx.free(hidden);
      hiddenStates.set(deviceCtx.device, normed);
    }

    this.ctx.synchronize();
    return hiddenStates;
  }

  /**
   * Forward through one transformer layer.
   * Unified logic for both prefill and decode.
   */
  private forwardLayer(
    hiddenStates: Map<number, Tensor>,
    layerIdx: number,
    seqLen: number,
    isPrefill: boolean,
    position: number
  ): void {
    const {
      hiddenSize,
      headDim,
      rmsNormEps,
      useQkNorm,
    } = this.ctx.config;

    const prefix = `model.layers.${layerIdx}`;
    const p = this.profiler;
    const t = this.tracer;

    // ========== Attention Sublayer ==========
    const afterAttn: Map<number, Tensor> = new Map();

    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      const hidden = hiddenStates.get(deviceCtx.device)!;
      const kvCache = deviceCtx.kvCache!;
      const ropeCache = deviceCtx.ropeCache!;

      // 1. Input LayerNorm
      const inputNormSpan = p.startSpan("input_norm", deviceCtx.device);
      t.begin("input_norm", "gpu", layerIdx);
      const inputNormWeight = deviceCtx.getWeight(`${prefix}.input_layernorm.weight`);
      const normedInput = deviceCtx.alloc([1, seqLen, hiddenSize]);
      this.rmsnorm(
        hidden.ptr,
        inputNormWeight.ptr,
        normedInput.ptr,
        1,
        seqLen,
        hiddenSize,
        rmsNormEps
      );
      t.end("input_norm", layerIdx);
      p.endSpan(inputNormSpan);

      // 2. QKV projections (column parallel - partitioned output)
      const qkvSpan = p.startSpan("qkv_proj", deviceCtx.device);
      t.begin("qkv_proj", "gpu", layerIdx);
      const qWeight = deviceCtx.getWeight(`${prefix}.self_attn.q_proj.weight`);
      const kWeight = deviceCtx.getWeight(`${prefix}.self_attn.k_proj.weight`);
      const vWeight = deviceCtx.getWeight(`${prefix}.self_attn.v_proj.weight`);

      const qSize = this.ctx.numHeadsPerRank * headDim;
      const kvSize = this.ctx.numKvHeadsPerRank * headDim;

      const q = deviceCtx.alloc([1, seqLen, qSize]);
      const k = deviceCtx.alloc([1, seqLen, kvSize]);
      const v = deviceCtx.alloc([1, seqLen, kvSize]);

      this.gemmTransB(normedInput.ptr, qWeight.ptr, q.ptr, seqLen, qSize, hiddenSize);
      this.gemmTransB(normedInput.ptr, kWeight.ptr, k.ptr, seqLen, kvSize, hiddenSize);
      this.gemmTransB(normedInput.ptr, vWeight.ptr, v.ptr, seqLen, kvSize, hiddenSize);

      // Add QKV biases if present (GPT-OSS has attention_bias=True)
      if (deviceCtx.hasWeight(`${prefix}.self_attn.q_proj.bias`)) {
        const qBias = deviceCtx.getWeight(`${prefix}.self_attn.q_proj.bias`);
        const kBias = deviceCtx.getWeight(`${prefix}.self_attn.k_proj.bias`);
        const vBias = deviceCtx.getWeight(`${prefix}.self_attn.v_proj.bias`);
        // Broadcast bias to each token position
        for (let pos = 0; pos < seqLen; pos++) {
          this.add(q.ptr + BigInt(pos * qSize * 2), qBias.ptr, q.ptr + BigInt(pos * qSize * 2), qSize);
          this.add(k.ptr + BigInt(pos * kvSize * 2), kBias.ptr, k.ptr + BigInt(pos * kvSize * 2), kvSize);
          this.add(v.ptr + BigInt(pos * kvSize * 2), vBias.ptr, v.ptr + BigInt(pos * kvSize * 2), kvSize);
        }
      }
      t.end("qkv_proj", layerIdx);
      p.endSpan(qkvSpan);

      // 2b. QK-norm (Qwen3 style)
      if (useQkNorm) {
        const qkNormSpan = p.startSpan("qk_norm", deviceCtx.device);
        t.begin("qk_norm", "gpu", layerIdx);
        const qNormWeight = deviceCtx.getWeight(`${prefix}.self_attn.q_norm.weight`);
        const kNormWeight = deviceCtx.getWeight(`${prefix}.self_attn.k_norm.weight`);
        this.rmsnorm(q.ptr, qNormWeight.ptr, q.ptr, seqLen * this.ctx.numHeadsPerRank, 1, headDim, rmsNormEps);
        this.rmsnorm(k.ptr, kNormWeight.ptr, k.ptr, seqLen * this.ctx.numKvHeadsPerRank, 1, headDim, rmsNormEps);
        t.end("qk_norm", layerIdx);
        p.endSpan(qkNormSpan);
      }

      // 3. RoPE
      const ropeSpan = p.startSpan("rope", deviceCtx.device);
      t.begin("rope", "gpu", layerIdx);
      const posOffset = isPrefill ? 0 : position;
      this.rotaryEmbedding(
        q.ptr,
        k.ptr,
        ropeCache.cos.ptr,
        ropeCache.sin.ptr,
        1,
        seqLen,
        this.ctx.numHeadsPerRank,
        this.ctx.numKvHeadsPerRank,
        headDim,
        posOffset
      );
      t.end("rope", layerIdx);
      p.endSpan(ropeSpan);

      // 4. Update KV cache
      const kvUpdateSpan = p.startSpan("kv_update", deviceCtx.device);
      t.begin("kv_update", "gpu", layerIdx);
      const layerCache = kvCache.update(layerIdx, k, v);
      t.end("kv_update", layerIdx);
      p.endSpan(kvUpdateSpan);

      // 5. Attention (local to this rank's heads)
      const attnSpan = p.startSpan("attention", deviceCtx.device);
      t.begin("attention", "gpu", layerIdx);
      const attnOut = deviceCtx.alloc([1, seqLen, qSize]);
      const softmaxScale = 1.0 / Math.sqrt(headDim);

      // Check if this layer has attention sinks (GPT-OSS)
      const sinksKey = `${prefix}.self_attn.sinks`;
      if (deviceCtx.hasWeight(sinksKey)) {
        const sinks = deviceCtx.getWeight(sinksKey);
        this.cuda.attentionWithSinksBf16(
          q.ptr,
          layerCache.k.ptr,
          layerCache.v.ptr,
          sinks.ptr,
          attnOut.ptr,
          1,
          seqLen,
          kvCache.seqLen + seqLen,
          kvCache.maxSeqLen,
          isPrefill ? 0 : position,
          this.ctx.numHeadsPerRank,
          this.ctx.numKvHeadsPerRank,
          headDim,
          softmaxScale,
          true
        );
      } else {
        this.flashAttention(
          q.ptr,
          layerCache.k.ptr,
          layerCache.v.ptr,
          attnOut.ptr,
          1,
          seqLen,
          kvCache.seqLen + seqLen,
          kvCache.maxSeqLen,
          isPrefill ? 0 : position,
          this.ctx.numHeadsPerRank,
          this.ctx.numKvHeadsPerRank,
          headDim,
          softmaxScale,
          true
        );
      }
      t.end("attention", layerIdx);
      p.endSpan(attnSpan);

      // 6. Output projection (row parallel - needs all-reduce)
      const oProjSpan = p.startSpan("o_proj", deviceCtx.device);
      t.begin("o_proj", "gpu", layerIdx);
      const oWeight = deviceCtx.getWeight(`${prefix}.self_attn.o_proj.weight`);
      const attnProjected = deviceCtx.alloc([1, seqLen, hiddenSize]);
      this.gemmTransB(attnOut.ptr, oWeight.ptr, attnProjected.ptr, seqLen, hiddenSize, qSize);

      // Add o_proj bias if present
      if (deviceCtx.hasWeight(`${prefix}.self_attn.o_proj.bias`)) {
        const oBias = deviceCtx.getWeight(`${prefix}.self_attn.o_proj.bias`);
        for (let pos = 0; pos < seqLen; pos++) {
          this.add(attnProjected.ptr + BigInt(pos * hiddenSize * 2), oBias.ptr, attnProjected.ptr + BigInt(pos * hiddenSize * 2), hiddenSize);
        }
      }
      t.end("o_proj", layerIdx);
      p.endSpan(oProjSpan);

      afterAttn.set(deviceCtx.device, attnProjected);

      // Cleanup
      deviceCtx.free(normedInput);
      deviceCtx.free(q);
      deviceCtx.free(k);
      deviceCtx.free(v);
      deviceCtx.free(attnOut);
    }

    // All-reduce attention outputs (no-op for TP=1)
    const attnAllReduceSpan = p.startSpan("attn_allreduce", 0);
    this.ctx.allReduceSum(afterAttn);
    this.ctx.synchronize();
    p.endSpan(attnAllReduceSpan);

    // ========== MLP Sublayer ==========
    const afterMlp: Map<number, Tensor> = new Map();

    // Check if this is an MoE model (used for all-reduce decision)
    const isMoE = this.ctx.config.numLocalExperts > 0;

    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      const hidden = hiddenStates.get(deviceCtx.device)!;
      const attnOut = afterAttn.get(deviceCtx.device)!;

      // 7. Residual connection
      const residualSpan = p.startSpan("residual", deviceCtx.device);
      const residual1 = deviceCtx.alloc([1, seqLen, hiddenSize]);
      this.add(hidden.ptr, attnOut.ptr, residual1.ptr, seqLen * hiddenSize);
      deviceCtx.free(attnOut);
      p.endSpan(residualSpan);

      // 8. Post-attention LayerNorm
      const postNormSpan = p.startSpan("post_norm", deviceCtx.device);
      const postNormWeight = deviceCtx.getWeight(`${prefix}.post_attention_layernorm.weight`);
      const normedAfterAttn = deviceCtx.alloc([1, seqLen, hiddenSize]);
      this.rmsnorm(
        residual1.ptr,
        postNormWeight.ptr,
        normedAfterAttn.ptr,
        1,
        seqLen,
        hiddenSize,
        rmsNormEps
      );
      p.endSpan(postNormSpan);

      // 9. MLP (Dense or MoE)
      let mlpProjected: Tensor;

      if (isMoE) {
        // MoE: Mixture of Experts MLP
        mlpProjected = this.forwardMoE(
          deviceCtx,
          normedAfterAttn,
          prefix,
          layerIdx,
          seqLen,
          p
        );

        // Store residual for final addition
        hiddenStates.set(deviceCtx.device, residual1);
        afterMlp.set(deviceCtx.device, mlpProjected);

        // Cleanup
        deviceCtx.free(normedAfterAttn);
      } else {
        // Dense MLP: gate/up (column parallel)
        const mlpGateUpSpan = p.startSpan("mlp_gate_up", deviceCtx.device);
        const gateWeight = deviceCtx.getWeight(`${prefix}.mlp.gate_proj.weight`);
        const upWeight = deviceCtx.getWeight(`${prefix}.mlp.up_proj.weight`);

        const gate = deviceCtx.alloc([1, seqLen, this.ctx.intermediateSizePerRank]);
        const up = deviceCtx.alloc([1, seqLen, this.ctx.intermediateSizePerRank]);

        this.gemmTransB(
          normedAfterAttn.ptr,
          gateWeight.ptr,
          gate.ptr,
          seqLen,
          this.ctx.intermediateSizePerRank,
          hiddenSize
        );
        this.gemmTransB(
          normedAfterAttn.ptr,
          upWeight.ptr,
          up.ptr,
          seqLen,
          this.ctx.intermediateSizePerRank,
          hiddenSize
        );
        p.endSpan(mlpGateUpSpan);

        // SwiGLU
        const swigluSpan = p.startSpan("swiglu", deviceCtx.device);
        const mlpOut = deviceCtx.alloc([1, seqLen, this.ctx.intermediateSizePerRank]);
        this.swiglu(gate.ptr, up.ptr, mlpOut.ptr, seqLen * this.ctx.intermediateSizePerRank);
        p.endSpan(swigluSpan);

        // 10. MLP down (row parallel - needs all-reduce)
        const mlpDownSpan = p.startSpan("mlp_down", deviceCtx.device);
        const downWeight = deviceCtx.getWeight(`${prefix}.mlp.down_proj.weight`);
        mlpProjected = deviceCtx.alloc([1, seqLen, hiddenSize]);
        this.gemmTransB(
          mlpOut.ptr,
          downWeight.ptr,
          mlpProjected.ptr,
          seqLen,
          hiddenSize,
          this.ctx.intermediateSizePerRank
        );
        p.endSpan(mlpDownSpan);

        afterMlp.set(deviceCtx.device, mlpProjected);

        // Store residual for final addition
        hiddenStates.set(deviceCtx.device, residual1);

        // Cleanup
        deviceCtx.free(normedAfterAttn);
        deviceCtx.free(gate);
        deviceCtx.free(up);
        deviceCtx.free(mlpOut);
      }
    }

    // All-reduce MLP outputs (only for dense MLP with sharded weights)
    // For MoE, weights are replicated so each GPU has the full result - no reduce needed
    const mlpAllReduceSpan = p.startSpan("mlp_allreduce", 0);
    if (!isMoE) {
      this.ctx.allReduceSum(afterMlp);
      this.ctx.synchronize();
    }
    p.endSpan(mlpAllReduceSpan);

    // ========== Final Residual ==========
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      const residual1 = hiddenStates.get(deviceCtx.device)!;
      const mlpOut = afterMlp.get(deviceCtx.device)!;

      const output = deviceCtx.alloc([1, seqLen, hiddenSize]);
      this.add(residual1.ptr, mlpOut.ptr, output.ptr, seqLen * hiddenSize);

      deviceCtx.free(residual1);
      deviceCtx.free(mlpOut);
      hiddenStates.set(deviceCtx.device, output);
    }
  }

  /**
   * MoE forward pass: expert routing + weighted expert computation.
   */
  private forwardMoE(
    deviceCtx: DeviceContext,
    hidden: Tensor,
    prefix: string,
    layerIdx: number,
    seqLen: number,
    profiler: CudaProfiler
  ): Tensor {
    // Ensure we're on the correct device for all CUDA operations
    deviceCtx.setActive();

    const config = this.ctx.config;
    const {
      hiddenSize,
      intermediateSize,
      numLocalExperts,
      numExpertsPerToken,
    } = config;
    const topK = numExpertsPerToken;

    // Initialize MXFP4 tables on first MoE layer
    if (layerIdx === 0 && !this._mxfp4Initialized) {
      this.cuda.initMxfp4Tables();
      this._mxfp4Initialized = true;
    }

    const moeSpan = profiler.startSpan("moe", deviceCtx.device);

    // 1. Router: compute expert selection
    const routerSpan = profiler.startSpan("moe_router", deviceCtx.device);
    const routerWeight = deviceCtx.getWeight(`${prefix}.mlp.router.weight`);
    const routerBias = deviceCtx.hasWeight(`${prefix}.mlp.router.bias`)
      ? deviceCtx.getWeight(`${prefix}.mlp.router.bias`)
      : null;

    const numTokens = seqLen;
    const expertIndices = deviceCtx.alloc([numTokens, topK], DType.Int32);
    const expertWeights = deviceCtx.alloc([numTokens, topK], deviceCtx.dtype);

    this.cuda.moeRouterTopK(
      hidden.ptr,
      routerWeight.ptr,
      routerBias?.ptr ?? BigInt(0),
      expertIndices.ptr,
      expertWeights.ptr,
      1, // batch_size
      seqLen,
      hiddenSize,
      numLocalExperts,
      topK
    );
    profiler.endSpan(routerSpan);

    // 2. Get MXFP4 quantized expert weights
    const gateUpBlocks = deviceCtx.getWeight(`${prefix}.mlp.experts.gate_up_proj_blocks`);
    const gateUpScales = deviceCtx.getWeight(`${prefix}.mlp.experts.gate_up_proj_scales`);
    const downBlocks = deviceCtx.getWeight(`${prefix}.mlp.experts.down_proj_blocks`);
    const downScales = deviceCtx.getWeight(`${prefix}.mlp.experts.down_proj_scales`);

    // For single-token decode with TP=1, use fused kernel to avoid CPU sync
    // This keeps expert indices on GPU and does dequant+GEMM in one pass
    // Note: Fused kernel doesn't support TP>1 (sharded weights across GPUs)
    if (numTokens === 1 && this.ctx.worldSize === 1) {
      // Allocate output tensor
      const output = deviceCtx.alloc([1, 1, hiddenSize], deviceCtx.dtype);

      // Call fused MoE kernel
      this.cuda.moeFusedForward(
        hidden.ptr,
        gateUpBlocks.ptr,
        gateUpScales.ptr,
        downBlocks.ptr,
        downScales.ptr,
        expertIndices.ptr,
        expertWeights.ptr,
        output.ptr,
        hiddenSize,
        intermediateSize,
        numLocalExperts,
        topK
      );

      profiler.endSpan(routerSpan);
      return output.view([1, 1, hiddenSize]);
    }

    // Get biases for multi-token path
    const gateUpBias = deviceCtx.hasWeight(`${prefix}.mlp.experts.gate_up_proj_bias`)
      ? deviceCtx.getWeight(`${prefix}.mlp.experts.gate_up_proj_bias`)
      : null;
    const downBias = deviceCtx.hasWeight(`${prefix}.mlp.experts.down_proj_bias`)
      ? deviceCtx.getWeight(`${prefix}.mlp.experts.down_proj_bias`)
      : null;

    const numBlocksGateUp = gateUpBlocks.shape[2];
    const numBlocksDown = downBlocks.shape[2];

    // 3. Read expert indices back to CPU to determine which experts to dequant
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

    // Group tokens by expert for efficient batching
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

    // 4. Dequantize only the selected experts (on-demand)
    const dequantSpan = profiler.startSpan("moe_dequant", deviceCtx.device);

    // Allocate single-expert dequant buffers (reused across experts)
    const expertGateUpDequant = deviceCtx.alloc(
      [1, intermediateSize * 2, hiddenSize],
      deviceCtx.dtype
    );
    const expertDownDequant = deviceCtx.alloc(
      [1, hiddenSize, intermediateSize],
      deviceCtx.dtype
    );

    // Map to cache dequantized expert weights for this forward pass
    const dequantedGateUp = new Map<number, bigint>();
    const dequantedDown = new Map<number, bigint>();

    // Pre-allocate buffers for all unique experts
    const uniqueExperts = Array.from(tokensByExpert.keys());
    for (const expertIdx of uniqueExperts) {
      // Allocate persistent buffers for this expert
      const gateUpBuf = deviceCtx.alloc([intermediateSize * 2, hiddenSize], deviceCtx.dtype);
      const downBuf = deviceCtx.alloc([hiddenSize, intermediateSize], deviceCtx.dtype);

      // Dequantize gate_up for this expert
      this.cuda.mxfp4DequantSingleExpert(
        gateUpBlocks.ptr,
        gateUpScales.ptr,
        gateUpBuf.ptr,
        expertIdx,
        numLocalExperts,
        intermediateSize * 2,
        numBlocksGateUp,
        hiddenSize
      );

      // Dequantize down for this expert
      this.cuda.mxfp4DequantSingleExpert(
        downBlocks.ptr,
        downScales.ptr,
        downBuf.ptr,
        expertIdx,
        numLocalExperts,
        hiddenSize,
        numBlocksDown,
        intermediateSize
      );

      dequantedGateUp.set(expertIdx, gateUpBuf.ptr);
      dequantedDown.set(expertIdx, downBuf.ptr);
    }
    profiler.endSpan(dequantSpan);

    // 5. Compute expert outputs
    const expertSpan = profiler.startSpan("moe_expert", deviceCtx.device);

    // Output accumulator (initialized to zeros)
    const output = deviceCtx.alloc([1, seqLen, hiddenSize], deviceCtx.dtype);
    // Zero the output buffer before accumulating
    this.cuda.memset(output.ptr, 0, seqLen * hiddenSize * 2);  // 2 bytes per bf16

    // Process each active expert
    for (const [expertIdx, tokenKs] of tokensByExpert) {
      const expertGateUpPtr = dequantedGateUp.get(expertIdx)!;
      const expertDownPtr = dequantedDown.get(expertIdx)!;

      for (const tk of tokenKs) {
        const tokenIdx = Math.floor(tk / topK);
        const weight = weightsFloat[tk];

        // Get token hidden state
        const tokenHiddenPtr = hidden.ptr + BigInt(tokenIdx * hiddenSize * 2);

        // Allocate intermediate tensors
        const gateUp = deviceCtx.alloc([1, intermediateSize * 2], deviceCtx.dtype);
        const swigled = deviceCtx.alloc([1, intermediateSize], deviceCtx.dtype);
        const down = deviceCtx.alloc([1, hiddenSize], deviceCtx.dtype);

        // gate_up = hidden @ gateUpWeight^T
        this.gemmTransB(
          tokenHiddenPtr,
          expertGateUpPtr,
          gateUp.ptr,
          1,
          intermediateSize * 2,
          hiddenSize
        );

        // Add gate_up bias if present
        if (gateUpBias) {
          const expertGateUpBiasPtr = gateUpBias.ptr + BigInt(expertIdx * intermediateSize * 2 * 2);
          this.add(gateUp.ptr, expertGateUpBiasPtr, gateUp.ptr, intermediateSize * 2);
        }

        // Activation: GPT-OSS uses custom interleaved activation, others use SwiGLU
        if (this.ctx.config.modelType === "gpt_oss") {
          this.cuda.gptOssActivation(gateUp.ptr, swigled.ptr, 1, intermediateSize, 1.702, 7.0);
        } else {
          this.cuda.moeSwiglu(gateUp.ptr, swigled.ptr, 1, intermediateSize);
        }

        // down = swigled @ downWeight^T
        this.gemmTransB(
          swigled.ptr,
          expertDownPtr,
          down.ptr,
          1,
          hiddenSize,
          intermediateSize
        );

        // Add down bias if present
        if (downBias) {
          const expertDownBiasPtr = downBias.ptr + BigInt(expertIdx * hiddenSize * 2);
          this.add(down.ptr, expertDownBiasPtr, down.ptr, hiddenSize);
        }

        // output[tokenIdx] += weight * down
        const outputTokenPtr = output.ptr + BigInt(tokenIdx * hiddenSize * 2);
        this.scaleAdd(down.ptr, outputTokenPtr, weight, hiddenSize);

        // Cleanup
        deviceCtx.free(gateUp);
        deviceCtx.free(swigled);
        deviceCtx.free(down);
      }
    }

    profiler.endSpan(expertSpan);

    // Cleanup
    deviceCtx.free(expertIndices);
    deviceCtx.free(expertWeights);
    deviceCtx.free(expertGateUpDequant);
    deviceCtx.free(expertDownDequant);
    // Free dequanted expert buffers
    for (const ptr of dequantedGateUp.values()) {
      this.cuda.free(ptr);
    }
    for (const ptr of dequantedDown.values()) {
      this.cuda.free(ptr);
    }

    profiler.endSpan(moeSpan);

    return output;
  }

  // Flag for MXFP4 table initialization
  private _mxfp4Initialized = false;

  /**
   * Compute logits from hidden states.
   * Uses device 0 for final lm_head computation.
   */
  private computeLogits(hiddenStates: Map<number, Tensor>, seqLen: number): Float32Array {
    const deviceCtx = this.ctx.devices[0];
    deviceCtx.setActive();
    const hidden = hiddenStates.get(0)!;

    // lm_head is replicated
    const lmHeadSpan = this.profiler.startSpan("lm_head", 0);
    const lmHeadWeight = this.ctx.config.tieWordEmbeddings
      ? deviceCtx.getWeight("model.embed_tokens.weight")
      : deviceCtx.getWeight("lm_head.weight");

    // For prefill, only get logits for last position
    // For decode, seqLen=1 so we get all logits
    const logits = deviceCtx.alloc([1, seqLen, this.ctx.config.vocabSize]);

    this.gemmTransB(
      hidden.ptr,
      lmHeadWeight.ptr,
      logits.ptr,
      seqLen,
      this.ctx.config.vocabSize,
      this.ctx.config.hiddenSize
    );
    this.profiler.endSpan(lmHeadSpan);

    // Extract last position logits
    const logitsData = logits.toArray();
    deviceCtx.free(logits);

    if (seqLen === 1) {
      return logitsData;
    }

    // Extract last position for prefill
    const lastLogits = new Float32Array(this.ctx.config.vocabSize);
    const srcOffset = (seqLen - 1) * this.ctx.config.vocabSize;
    lastLogits.set(logitsData.subarray(srcOffset, srcOffset + this.ctx.config.vocabSize));
    return lastLogits;
  }

  /**
   * Generate tokens from a prompt.
   */
  async generate(
    inputIds: number[],
    maxNewTokens: number,
    onToken?: (token: number) => void | boolean,
    samplingParams?: Partial<SamplingParams>
  ): Promise<number[]> {
    // Initialize or resize KV caches if needed
    const maxSeqLen = inputIds.length + maxNewTokens;
    const existingCache = this.ctx.devices[0].kvCache;
    if (!existingCache || existingCache.maxSeqLen < maxSeqLen) {
      // Dispose old caches if they exist
      if (existingCache) {
        for (const deviceCtx of this.ctx.devices) {
          if (deviceCtx.kvCache) {
            deviceCtx.kvCache.dispose();
            deviceCtx.kvCache = null;
          }
        }
      }
      this.ctx.initKVCaches(maxSeqLen);
    }

    // Initialize RoPE cache if not done or too small
    const existingRope = this.ctx.devices[0].ropeCache;
    if (!existingRope || existingRope.cos.shape[0] < maxSeqLen) {
      this.ctx.initRopeCache(maxSeqLen);
    }

    const generatedTokens: number[] = [];
    const sampler = new Sampler(samplingParams ?? { temperature: 0 });

    // Prefill phase
    let hiddenStates = this.prefill(inputIds);

    // Compute logits and sample first token
    let logits = this.computeLogits(hiddenStates, inputIds.length);
    let nextToken = sampler.sample(logits, generatedTokens);
    generatedTokens.push(nextToken);

    // Check if callback requests stop
    if (onToken && onToken(nextToken) === false) {
      // Cleanup hidden states
      for (const deviceCtx of this.ctx.devices) {
        deviceCtx.setActive();
        deviceCtx.free(hiddenStates.get(deviceCtx.device)!);
      }
      this.cuda.setDevice(0);
      return generatedTokens;
    }

    // Cleanup hidden states
    for (const deviceCtx of this.ctx.devices) {
      deviceCtx.setActive();
      deviceCtx.free(hiddenStates.get(deviceCtx.device)!);
    }

    // Decode loop
    for (let i = 1; i < maxNewTokens; i++) {
      // Yield to event loop periodically to allow signal processing
      if (i % 4 === 0) {
        await new Promise(resolve => setImmediate(resolve));
      }

      const position = inputIds.length + i - 1;

      hiddenStates = this.decode(nextToken, position);
      logits = this.computeLogits(hiddenStates, 1);
      nextToken = sampler.sample(logits, generatedTokens);
      generatedTokens.push(nextToken);

      // Check if callback requests stop
      if (onToken && onToken(nextToken) === false) {
        // Cleanup hidden states
        for (const deviceCtx of this.ctx.devices) {
          deviceCtx.setActive();
          deviceCtx.free(hiddenStates.get(deviceCtx.device)!);
        }
        break;
      }

      // Cleanup hidden states after each step
      for (const deviceCtx of this.ctx.devices) {
        deviceCtx.setActive();
        deviceCtx.free(hiddenStates.get(deviceCtx.device)!);
      }

      // Check for EOS
      if (nextToken === this.ctx.config.eosTokenId) {
        break;
      }
    }

    // Reset to device 0
    this.cuda.setDevice(0);

    return generatedTokens;
  }

  /**
   * Convenience wrapper for single-GPU generation with simpler API.
   */
  async generateSimple(
    inputIds: number[],
    config: Partial<GenerationConfig> = {}
  ): Promise<number[]> {
    const genConfig = { ...DEFAULT_GENERATION_CONFIG, ...config };
    return this.generate(
      inputIds,
      genConfig.maxNewTokens,
      genConfig.streamCallback ? (t) => genConfig.streamCallback!(t) : undefined
    );
  }
}

/** @deprecated Use BatchedInferenceEngine instead */
export const InferenceEngine = LegacyInferenceEngine;
