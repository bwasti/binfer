// LLaMA model architecture implementation

import { Tensor, DType } from "../../tensor/tensor";
import { Backend } from "../../backend/interface";
import { LlamaConfig } from "../config";
import { LoadedModel, LLAMA_WEIGHT_NAMES, getLayerWeightName } from "../loader";

/**
 * LLaMA decoder-only transformer model.
 * Supports both dense MLPs and Mixture of Experts (MoE).
 */
export class LlamaModel {
  private config: LlamaConfig;
  private backend: Backend;
  private weights: Map<string, { ptr: bigint; info: any }>;

  // Precomputed RoPE cache
  private ropeCosSin: { cos: Tensor; sin: Tensor } | null = null;

  // MoE: Dequantized expert weights cache (to avoid re-dequantizing every layer)
  private dequantizedExperts: Map<number, {
    gateUp: Tensor;  // [num_experts, intermediate*2, hidden]
    down: Tensor;    // [num_experts, hidden, intermediate]
  }> = new Map();

  // Track if MXFP4 tables have been initialized
  private mxfp4Initialized = false;

  constructor(
    config: LlamaConfig,
    weights: Map<string, { ptr: bigint; info: any }>,
    backend: Backend
  ) {
    this.config = config;
    this.weights = weights;
    this.backend = backend;
  }

  /**
   * Check if a weight exists.
   */
  private hasWeight(name: string): boolean {
    return this.weights.has(name);
  }

  /**
   * Get a weight tensor by name.
   */
  private getWeight(name: string): Tensor {
    const w = this.weights.get(name);
    if (!w) {
      throw new Error(`Weight not found: ${name}`);
    }
    return Tensor.fromPtr(
      w.ptr,
      w.info.shape,
      this.config.dtype === "bfloat16" ? DType.BFloat16 : DType.Float16
    );
  }

  /**
   * Get a weight tensor by name, returning null if not found.
   */
  private getWeightOrNull(name: string): Tensor | null {
    const w = this.weights.get(name);
    if (!w) {
      return null;
    }
    return Tensor.fromPtr(
      w.ptr,
      w.info.shape,
      this.config.dtype === "bfloat16" ? DType.BFloat16 : DType.Float16
    );
  }

  /**
   * Get a layer-specific weight.
   */
  private getLayerWeight(template: string, layerIdx: number): Tensor {
    return this.getWeight(getLayerWeightName(template, layerIdx));
  }

  /**
   * Get a layer-specific weight, returning null if not found.
   */
  private getLayerWeightOrNull(template: string, layerIdx: number): Tensor | null {
    return this.getWeightOrNull(getLayerWeightName(template, layerIdx));
  }

  /**
   * Check if this model uses MoE (Mixture of Experts).
   */
  private isMoE(): boolean {
    return this.config.numLocalExperts > 0;
  }

  /**
   * Initialize RoPE cosine/sine cache.
   */
  initRopeCache(maxSeqLen: number = 4096): void {
    const { headDim, ropeTheta } = this.config;
    const halfDim = headDim / 2;

    // Compute frequencies on CPU
    const freqs = new Float32Array(halfDim);
    for (let i = 0; i < halfDim; i++) {
      freqs[i] = 1.0 / Math.pow(ropeTheta, (2 * i) / headDim);
    }

    // Compute cos/sin for each position
    const cosData = new Float32Array(maxSeqLen * halfDim);
    const sinData = new Float32Array(maxSeqLen * halfDim);

    for (let pos = 0; pos < maxSeqLen; pos++) {
      for (let i = 0; i < halfDim; i++) {
        const angle = pos * freqs[i];
        cosData[pos * halfDim + i] = Math.cos(angle);
        sinData[pos * halfDim + i] = Math.sin(angle);
      }
    }

    // Upload to GPU
    this.ropeCosSin = {
      cos: Tensor.fromArray(cosData, [maxSeqLen, halfDim], DType.Float16),
      sin: Tensor.fromArray(sinData, [maxSeqLen, halfDim], DType.Float16),
    };
  }

  /**
   * Forward pass for a single transformer layer.
   */
  private forwardLayer(
    hidden: Tensor,
    layerIdx: number,
    kvCache: { k: Tensor; v: Tensor } | null,
    positionOffset: number
  ): { hidden: Tensor; newKv: { k: Tensor; v: Tensor } } {
    const {
      hiddenSize,
      numAttentionHeads,
      numKeyValueHeads,
      headDim,
      intermediateSize,
      rmsNormEps,
    } = this.config;

    const batchSize = hidden.shape[0];
    const seqLen = hidden.shape[1];

    // 1. Input LayerNorm
    const normedInput = Tensor.empty(hidden.shape, { dtype: DType.Float16 });
    this.backend.rmsnorm(
      hidden,
      this.getLayerWeight(LLAMA_WEIGHT_NAMES.inputLayernorm, layerIdx),
      normedInput,
      rmsNormEps
    );

    // 2. Self Attention
    // Q, K, V projections
    const qWeight = this.getLayerWeight(LLAMA_WEIGHT_NAMES.qProj, layerIdx);
    const kWeight = this.getLayerWeight(LLAMA_WEIGHT_NAMES.kProj, layerIdx);
    const vWeight = this.getLayerWeight(LLAMA_WEIGHT_NAMES.vProj, layerIdx);

    const q = Tensor.empty(
      [batchSize, seqLen, numAttentionHeads * headDim],
      { dtype: DType.Float16 }
    );
    const k = Tensor.empty(
      [batchSize, seqLen, numKeyValueHeads * headDim],
      { dtype: DType.Float16 }
    );
    const v = Tensor.empty(
      [batchSize, seqLen, numKeyValueHeads * headDim],
      { dtype: DType.Float16 }
    );

    // Reshape for GEMM: [batch * seq, hidden] @ [hidden, head_dim * heads]
    const flatInput = normedInput.view([batchSize * seqLen, hiddenSize]);
    const flatQ = q.view([batchSize * seqLen, numAttentionHeads * headDim]);
    const flatK = k.view([batchSize * seqLen, numKeyValueHeads * headDim]);
    const flatV = v.view([batchSize * seqLen, numKeyValueHeads * headDim]);

    this.backend.gemm(flatInput, qWeight, flatQ);
    this.backend.gemm(flatInput, kWeight, flatK);
    this.backend.gemm(flatInput, vWeight, flatV);

    // Apply RoPE
    if (this.ropeCosSin) {
      this.backend.rotaryEmbedding(
        q.view([batchSize, seqLen, numAttentionHeads, headDim]),
        k.view([batchSize, seqLen, numKeyValueHeads, headDim]),
        this.ropeCosSin.cos,
        this.ropeCosSin.sin,
        positionOffset
      );
    }

    // TODO: Implement attention with KV cache
    // For now, just do a simple attention without caching
    const attnOutput = this.simpleAttention(q, k, v, batchSize, seqLen);

    // Output projection
    const oWeight = this.getLayerWeight(LLAMA_WEIGHT_NAMES.oProj, layerIdx);
    const attnProjected = Tensor.empty(
      [batchSize, seqLen, hiddenSize],
      { dtype: DType.Float16 }
    );
    this.backend.gemm(
      attnOutput.view([batchSize * seqLen, numAttentionHeads * headDim]),
      oWeight,
      attnProjected.view([batchSize * seqLen, hiddenSize])
    );

    // Residual connection
    const afterAttn = Tensor.empty(hidden.shape, { dtype: DType.Float16 });
    this.backend.add(hidden, attnProjected, afterAttn);

    // 3. Post-attention LayerNorm
    const normedAfterAttn = Tensor.empty(hidden.shape, { dtype: DType.Float16 });
    this.backend.rmsnorm(
      afterAttn,
      this.getLayerWeight(LLAMA_WEIGHT_NAMES.postAttentionLayernorm, layerIdx),
      normedAfterAttn,
      rmsNormEps
    );

    // 4. MLP (Dense SwiGLU or MoE)
    let mlpProjected: Tensor;

    if (this.isMoE()) {
      // MoE forward pass
      mlpProjected = this.forwardMoE(normedAfterAttn, layerIdx, batchSize, seqLen);
    } else {
      // Dense MLP forward pass
      mlpProjected = this.forwardDenseMLP(normedAfterAttn, layerIdx, batchSize, seqLen);
    }

    // Final residual
    const layerOutput = Tensor.empty(hidden.shape, { dtype: DType.Float16 });
    this.backend.add(afterAttn, mlpProjected, layerOutput);

    // Clean up intermediate tensors
    normedInput.dispose();
    q.dispose();
    k.dispose();
    v.dispose();
    attnOutput.dispose();
    attnProjected.dispose();
    afterAttn.dispose();
    normedAfterAttn.dispose();
    mlpProjected.dispose();

    return {
      hidden: layerOutput,
      newKv: { k, v }, // TODO: Return actual cached KV
    };
  }

  /**
   * Dense MLP forward pass (for non-MoE models).
   */
  private forwardDenseMLP(
    input: Tensor,
    layerIdx: number,
    batchSize: number,
    seqLen: number
  ): Tensor {
    const { hiddenSize, intermediateSize } = this.config;
    const dtype = this.config.dtype === "bfloat16" ? DType.BFloat16 : DType.Float16;

    const gateWeight = this.getLayerWeight(LLAMA_WEIGHT_NAMES.gateProj, layerIdx);
    const upWeight = this.getLayerWeight(LLAMA_WEIGHT_NAMES.upProj, layerIdx);
    const downWeight = this.getLayerWeight(LLAMA_WEIGHT_NAMES.downProj, layerIdx);

    const gate = Tensor.empty([batchSize, seqLen, intermediateSize], { dtype });
    const up = Tensor.empty([batchSize, seqLen, intermediateSize], { dtype });

    this.backend.gemm(
      input.view([batchSize * seqLen, hiddenSize]),
      gateWeight,
      gate.view([batchSize * seqLen, intermediateSize])
    );
    this.backend.gemm(
      input.view([batchSize * seqLen, hiddenSize]),
      upWeight,
      up.view([batchSize * seqLen, intermediateSize])
    );

    // SwiGLU: silu(gate) * up
    const mlpOut = Tensor.empty([batchSize, seqLen, intermediateSize], { dtype });
    this.backend.swiglu(gate, up, mlpOut);

    // Down projection
    const mlpProjected = Tensor.empty([batchSize, seqLen, hiddenSize], { dtype });
    this.backend.gemm(
      mlpOut.view([batchSize * seqLen, intermediateSize]),
      downWeight,
      mlpProjected.view([batchSize * seqLen, hiddenSize])
    );

    // Clean up
    gate.dispose();
    up.dispose();
    mlpOut.dispose();

    return mlpProjected;
  }

  /**
   * MoE (Mixture of Experts) MLP forward pass.
   * Uses MXFP4 quantized weights with top-k expert selection.
   */
  private forwardMoE(
    input: Tensor,
    layerIdx: number,
    batchSize: number,
    seqLen: number
  ): Tensor {
    const {
      hiddenSize,
      intermediateSize,
      numLocalExperts,
      numExpertsPerToken,
    } = this.config;
    const topK = numExpertsPerToken;
    const dtype = this.config.dtype === "bfloat16" ? DType.BFloat16 : DType.Float16;

    // Initialize MXFP4 tables if needed
    if (!this.mxfp4Initialized && this.backend.initMxfp4Tables) {
      this.backend.initMxfp4Tables();
      this.mxfp4Initialized = true;
    }

    // Get router weights
    const routerWeight = this.getLayerWeight(LLAMA_WEIGHT_NAMES.moeRouterWeight, layerIdx);
    const routerBias = this.getLayerWeightOrNull(LLAMA_WEIGHT_NAMES.moeRouterBias, layerIdx);

    // Router: compute expert selection
    const numTokens = batchSize * seqLen;
    const expertIndices = Tensor.empty([numTokens, topK], { dtype: DType.Int32 });
    const expertWeights = Tensor.empty([numTokens, topK], { dtype });

    if (this.backend.moeRouterTopK) {
      this.backend.moeRouterTopK(
        input.view([numTokens, hiddenSize]),
        routerWeight,
        routerBias,
        expertIndices,
        expertWeights,
        batchSize,
        seqLen,
        hiddenSize,
        numLocalExperts,
        topK
      );
    } else {
      throw new Error("MoE router not available in backend");
    }

    // Get MXFP4 quantized expert weights
    const gateUpBlocks = this.getLayerWeight(LLAMA_WEIGHT_NAMES.moeGateUpBlocks, layerIdx);
    const gateUpScales = this.getLayerWeight(LLAMA_WEIGHT_NAMES.moeGateUpScales, layerIdx);
    const gateUpBias = this.getLayerWeightOrNull(LLAMA_WEIGHT_NAMES.moeGateUpBias, layerIdx);
    const downBlocks = this.getLayerWeight(LLAMA_WEIGHT_NAMES.moeDownBlocks, layerIdx);
    const downScales = this.getLayerWeight(LLAMA_WEIGHT_NAMES.moeDownScales, layerIdx);
    const downBias = this.getLayerWeightOrNull(LLAMA_WEIGHT_NAMES.moeDownBias, layerIdx);

    // For now, use a simplified approach: dequantize all experts and do sparse GEMM
    // A more efficient implementation would use a fused kernel

    // Dequantize gate_up: [num_experts, intermediate*2, hidden]
    const gateUpDequant = Tensor.empty(
      [numLocalExperts, intermediateSize * 2, hiddenSize],
      { dtype }
    );

    // gateUpBlocks shape: [num_experts, out_features, num_blocks, 16]
    // where out_features = intermediate*2, num_blocks = hidden/32 (since MXFP4 uses 32 elements per block)
    const numBlocks = gateUpBlocks.shape[2]; // 90 for GPT-OSS

    if (this.backend.mxfp4Dequant) {
      this.backend.mxfp4Dequant(
        gateUpBlocks,
        gateUpScales,
        gateUpBias,
        gateUpDequant,
        numLocalExperts,
        intermediateSize * 2,
        numBlocks,
        hiddenSize
      );
    }

    // Dequantize down: [num_experts, hidden, intermediate]
    const downDequant = Tensor.empty(
      [numLocalExperts, hiddenSize, intermediateSize],
      { dtype }
    );

    const downNumBlocks = downBlocks.shape[2];

    if (this.backend.mxfp4Dequant) {
      this.backend.mxfp4Dequant(
        downBlocks,
        downScales,
        downBias,
        downDequant,
        numLocalExperts,
        hiddenSize,
        downNumBlocks,
        intermediateSize
      );
    }

    // Output tensor initialized to zeros
    const output = Tensor.zeros([batchSize, seqLen, hiddenSize], { dtype });

    // TODO: Implement the actual MoE computation
    // For a complete implementation, we would:
    // 1. For each token, get top-k expert indices and weights from expertIndices/expertWeights
    // 2. For each selected expert, compute: gate_up = input @ expert_gate_up_weight^T
    // 3. Apply SwiGLU to gate_up
    // 4. Compute: down = swigled @ expert_down_weight^T
    // 5. Accumulate: output += expert_weight * down
    //
    // This requires either:
    // - A fused CUDA kernel that handles the sparse expert selection
    // - Reading indices back to CPU and doing per-expert batched GEMMs
    //
    // For now, we return zeros as placeholder to allow testing other parts
    console.warn(`MoE layer ${layerIdx}: using placeholder (returning zeros)`);

    // Clean up
    expertIndices.dispose();
    expertWeights.dispose();
    gateUpDequant.dispose();
    downDequant.dispose();

    return output;
  }

  /**
   * Simple attention implementation (without FlashAttention or KV cache).
   */
  private simpleAttention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    batchSize: number,
    seqLen: number
  ): Tensor {
    // This is a placeholder - real implementation would use FlashAttention
    // For now, just return q as a dummy output
    console.warn("Using placeholder attention - FlashAttention not yet integrated");
    return q;
  }

  /**
   * Full forward pass.
   */
  forward(
    inputIds: Tensor,
    kvCaches: Array<{ k: Tensor; v: Tensor }> | null = null,
    positionOffset: number = 0
  ): Tensor {
    const batchSize = inputIds.shape[0];
    const seqLen = inputIds.shape[1];
    const { hiddenSize, numHiddenLayers, vocabSize, rmsNormEps } = this.config;

    // 1. Embedding lookup
    const embedWeight = this.getWeight(LLAMA_WEIGHT_NAMES.embedTokens);
    let hidden = Tensor.empty([batchSize, seqLen, hiddenSize], { dtype: DType.Float16 });
    this.backend.embedding(embedWeight, inputIds, hidden);

    // 2. Transformer layers
    const newKvCaches: Array<{ k: Tensor; v: Tensor }> = [];
    for (let i = 0; i < numHiddenLayers; i++) {
      const kvCache = kvCaches ? kvCaches[i] : null;
      const result = this.forwardLayer(hidden, i, kvCache, positionOffset);

      // Dispose old hidden, use new one
      hidden.dispose();
      hidden = result.hidden;
      newKvCaches.push(result.newKv);
    }

    // 3. Final LayerNorm
    const normWeight = this.getWeight(LLAMA_WEIGHT_NAMES.norm);
    const normed = Tensor.empty([batchSize, seqLen, hiddenSize], { dtype: DType.Float16 });
    this.backend.rmsnorm(hidden, normWeight, normed, rmsNormEps);

    // 4. LM Head (output projection)
    const lmHeadWeight = this.config.tieWordEmbeddings
      ? embedWeight
      : this.getWeight(LLAMA_WEIGHT_NAMES.lmHead);

    const logits = Tensor.empty([batchSize, seqLen, vocabSize], { dtype: DType.Float16 });
    this.backend.gemm(
      normed.view([batchSize * seqLen, hiddenSize]),
      lmHeadWeight,
      logits.view([batchSize * seqLen, vocabSize])
    );

    // Clean up
    hidden.dispose();
    normed.dispose();

    return logits;
  }

  /**
   * Generate tokens autoregressively.
   */
  async generate(
    inputIds: Tensor,
    maxNewTokens: number = 100,
    temperature: number = 1.0,
    topK: number = 50
  ): Promise<number[]> {
    // Initialize RoPE cache if needed
    if (!this.ropeCosSin) {
      this.initRopeCache(this.config.maxPositionEmbeddings);
    }

    const generatedTokens: number[] = [];
    let currentIds = inputIds;
    let positionOffset = 0;

    for (let i = 0; i < maxNewTokens; i++) {
      // Forward pass
      const logits = this.forward(currentIds, null, positionOffset);

      // Get logits for the last token
      const batchSize = logits.shape[0];
      const seqLen = logits.shape[1];
      const vocabSize = logits.shape[2];

      // Apply temperature and sample
      const lastLogits = Tensor.empty([batchSize, vocabSize], { dtype: DType.Float16 });
      // TODO: Implement slicing to get last token logits

      // Apply softmax
      const probs = Tensor.empty([batchSize, vocabSize], { dtype: DType.Float16 });
      this.backend.softmax(lastLogits, probs, temperature);

      // Sample from top-k (simplified: just take argmax for now)
      const probsArray = probs.toArray();
      let maxIdx = 0;
      let maxVal = probsArray[0];
      for (let j = 1; j < vocabSize; j++) {
        if (probsArray[j] > maxVal) {
          maxVal = probsArray[j];
          maxIdx = j;
        }
      }

      generatedTokens.push(maxIdx);

      // Check for EOS
      if (maxIdx === this.config.eosTokenId) {
        break;
      }

      // Prepare next input (just the new token)
      const nextIds = Tensor.fromArray(
        new Int32Array([maxIdx]),
        [1, 1],
        DType.Int32
      );
      currentIds = nextIds;
      positionOffset += seqLen;

      // Clean up
      logits.dispose();
      lastLogits.dispose();
      probs.dispose();
    }

    return generatedTokens;
  }
}

/**
 * Load and create a LLaMA model from a loaded model.
 */
export function createLlamaModel(
  loadedModel: LoadedModel,
  backend: Backend
): LlamaModel {
  return new LlamaModel(loadedModel.config, loadedModel.weights, backend);
}
