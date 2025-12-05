// LLaMA model configuration parser
// Parses HuggingFace config.json for LLaMA-style models
// Supports: Llama, Qwen, GPT-OSS, DeepSeek, etc.

export interface LlamaConfig {
  // Model architecture
  architectures: string[];
  modelType: string;

  // Dimensions
  hiddenSize: number;
  intermediateSize: number;
  numHiddenLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;

  // Vocabulary
  vocabSize: number;
  maxPositionEmbeddings: number;

  // Normalization
  rmsNormEps: number;

  // RoPE configuration
  ropeTheta: number;
  ropeScaling: RopeScaling | null;

  // Activation
  hiddenAct: string;

  // Misc
  tieWordEmbeddings: boolean;
  bosTokenId: number;
  eosTokenId: number;
  padTokenId: number | null;

  // Attention
  attentionBias: boolean;
  attentionDropout: number;

  // QK-norm (Qwen3 style)
  useQkNorm: boolean;

  // Mixture of Experts (MoE)
  numLocalExperts: number;      // Total number of experts (0 = dense model)
  numExpertsPerToken: number;   // Experts activated per token (top-k)

  // Quantization
  quantizationMethod: string | null;  // e.g., "mxfp4", null for unquantized

  // Sliding window attention
  slidingWindow: number | null;

  // Tensor Parallelism
  tensorParallelSize?: number;

  // Data type (detected from weights or config)
  dtype: "float16" | "bfloat16";
}

export interface RopeScaling {
  type: string;
  factor: number;
  lowFreqFactor?: number;
  highFreqFactor?: number;
  originalMaxPositionEmbeddings?: number;
  // YaRN specific
  betaFast?: number;
  betaSlow?: number;
}

export function parseLlamaConfig(configJson: Record<string, unknown>): LlamaConfig {
  // Handle both camelCase and snake_case keys
  const get = <T>(key: string, snakeKey?: string, defaultValue?: T): T => {
    const value = configJson[key] ?? configJson[snakeKey ?? toSnakeCase(key)];
    if (value === undefined && defaultValue !== undefined) {
      return defaultValue;
    }
    if (value === undefined) {
      throw new Error(`Missing required config key: ${key}`);
    }
    return value as T;
  };

  const modelType = get<string>("modelType", "model_type", "llama");
  const architectures = get<string[]>("architectures", "architectures", []);

  // Parse MoE config
  const numLocalExperts = get<number>("numLocalExperts", "num_local_experts", 0);
  const numExpertsPerToken = get<number>(
    "numExpertsPerTok",
    "num_experts_per_tok",
    get<number>("expertsPerToken", "experts_per_token", 0)
  );

  // Parse quantization config
  const quantizationConfig = configJson.quantization_config as Record<string, unknown> | undefined;
  const quantizationMethod = quantizationConfig?.quant_method as string | null ?? null;

  // Parse sliding window
  const slidingWindow = get<number | null>("slidingWindow", "sliding_window", null);

  const numAttentionHeads = get<number>("numAttentionHeads", "num_attention_heads");
  const hiddenSize = get<number>("hiddenSize", "hidden_size");

  // num_key_value_heads defaults to num_attention_heads for MHA
  const numKeyValueHeads = get<number>(
    "numKeyValueHeads",
    "num_key_value_heads",
    numAttentionHeads
  );

  // head_dim is usually hidden_size / num_attention_heads
  const headDim = get<number>(
    "headDim",
    "head_dim",
    Math.floor(hiddenSize / numAttentionHeads)
  );

  // Parse rope_scaling if present
  let ropeScaling: RopeScaling | null = null;
  const rawRopeScaling = configJson.rope_scaling as Record<string, unknown> | null;
  if (rawRopeScaling) {
    ropeScaling = {
      type: rawRopeScaling.type as string || rawRopeScaling.rope_type as string,
      factor: rawRopeScaling.factor as number,
      lowFreqFactor: rawRopeScaling.low_freq_factor as number | undefined,
      highFreqFactor: rawRopeScaling.high_freq_factor as number | undefined,
      originalMaxPositionEmbeddings:
        rawRopeScaling.original_max_position_embeddings as number | undefined,
      betaFast: rawRopeScaling.beta_fast as number | undefined,
      betaSlow: rawRopeScaling.beta_slow as number | undefined,
    };
  }

  // Detect QK-norm based on model type or explicit config
  const useQkNorm =
    get<boolean>("qk_norm", "qk_norm", false) ||
    modelType === "qwen3" ||
    architectures.some(a => a.toLowerCase().includes("qwen3"));

  return {
    architectures,
    modelType,

    hiddenSize,
    intermediateSize: get<number>("intermediateSize", "intermediate_size"),
    numHiddenLayers: get<number>("numHiddenLayers", "num_hidden_layers"),
    numAttentionHeads,
    numKeyValueHeads,
    headDim,

    vocabSize: get<number>("vocabSize", "vocab_size"),
    maxPositionEmbeddings: get<number>(
      "maxPositionEmbeddings",
      "max_position_embeddings",
      4096
    ),

    rmsNormEps: get<number>("rmsNormEps", "rms_norm_eps", 1e-5),

    ropeTheta: get<number>("ropeTheta", "rope_theta", 10000.0),
    ropeScaling,

    hiddenAct: get<string>("hiddenAct", "hidden_act", "silu"),

    tieWordEmbeddings: get<boolean>("tieWordEmbeddings", "tie_word_embeddings", false),
    bosTokenId: get<number>("bosTokenId", "bos_token_id", 1),
    eosTokenId: get<number>("eosTokenId", "eos_token_id", 2),
    padTokenId: get<number | null>("padTokenId", "pad_token_id", null),

    attentionBias: get<boolean>("attentionBias", "attention_bias", false),
    attentionDropout: get<number>("attentionDropout", "attention_dropout", 0.0),

    useQkNorm,

    // MoE config
    numLocalExperts,
    numExpertsPerToken,

    // Quantization
    quantizationMethod,

    // Sliding window attention
    slidingWindow,

    // Default to bfloat16 (most common for modern models)
    // This will be updated by the loader when it detects the actual dtype from weights
    dtype: get<"float16" | "bfloat16">("torch_dtype", "torch_dtype", "bfloat16"),
  };
}

function toSnakeCase(str: string): string {
  return str.replace(/([A-Z])/g, "_$1").toLowerCase();
}

// Calculate model memory requirements
export function estimateMemory(config: LlamaConfig): {
  parametersB: number;
  memoryGB: number;
  kvCachePerTokenMB: number;
} {
  const {
    vocabSize,
    hiddenSize,
    intermediateSize,
    numHiddenLayers,
    numAttentionHeads,
    numKeyValueHeads,
  } = config;

  // Embedding: vocab_size * hidden_size
  const embeddingParams = vocabSize * hiddenSize;

  // Per-layer params:
  // - q_proj: hidden_size * (num_heads * head_dim)
  // - k_proj: hidden_size * (num_kv_heads * head_dim)
  // - v_proj: hidden_size * (num_kv_heads * head_dim)
  // - o_proj: (num_heads * head_dim) * hidden_size
  // - gate_proj: hidden_size * intermediate_size
  // - up_proj: hidden_size * intermediate_size
  // - down_proj: intermediate_size * hidden_size
  // - input_layernorm: hidden_size
  // - post_attention_layernorm: hidden_size

  const qkHeadDim = config.headDim;
  const qParams = hiddenSize * (numAttentionHeads * qkHeadDim);
  const kvParams = hiddenSize * (numKeyValueHeads * qkHeadDim) * 2;
  const oParams = (numAttentionHeads * qkHeadDim) * hiddenSize;
  const mlpParams = hiddenSize * intermediateSize * 3;
  const normParams = hiddenSize * 2;

  const perLayerParams = qParams + kvParams + oParams + mlpParams + normParams;
  const layerParams = perLayerParams * numHiddenLayers;

  // Final norm + LM head (if not tied)
  const finalNorm = hiddenSize;
  const lmHead = config.tieWordEmbeddings ? 0 : vocabSize * hiddenSize;

  const totalParams = embeddingParams + layerParams + finalNorm + lmHead;

  // Memory in GB (fp16 = 2 bytes per param)
  const memoryGB = (totalParams * 2) / 1024 / 1024 / 1024;

  // KV cache per token per layer (k + v, each num_kv_heads * head_dim * 2 bytes)
  const kvPerTokenPerLayer = numKeyValueHeads * qkHeadDim * 2 * 2; // *2 for fp16
  const kvPerTokenMB = (kvPerTokenPerLayer * numHiddenLayers) / 1024 / 1024;

  return {
    parametersB: totalParams / 1e9,
    memoryGB,
    kvCachePerTokenMB: kvPerTokenMB,
  };
}

/**
 * Automatically determine the optimal tensor parallel degree based on model
 * memory requirements and available GPU memory.
 *
 * @param config Model configuration
 * @param gpuMemoryGB Available memory per GPU in GB
 * @param numGpus Number of available GPUs
 * @returns Optimal TP degree (1, 2, 4, or 8)
 */
export function autoDetectTP(
  config: LlamaConfig,
  gpuMemoryGB: number,
  numGpus: number
): number {
  const { memoryGB, kvCachePerTokenMB } = estimateMemory(config);

  // Reserve memory for:
  // - Activations (rough estimate: ~1-2GB for typical batch sizes)
  // - KV cache (assume max 4096 tokens for now)
  // - CUDA overhead (~0.5-1GB)
  const activationsGB = 2.0;
  const kvCacheGB = (kvCachePerTokenMB * 4096) / 1024;
  const cudaOverheadGB = 1.0;
  const reservedGB = activationsGB + kvCacheGB + cudaOverheadGB;

  // Available memory for model weights per GPU
  const availablePerGpu = gpuMemoryGB - reservedGB;

  // Calculate minimum TP needed
  const minTpNeeded = Math.ceil(memoryGB / availablePerGpu);

  // Round up to valid TP values (1, 2, 4, 8)
  const validTps = [1, 2, 4, 8];
  let selectedTp = 1;

  for (const tp of validTps) {
    if (tp >= minTpNeeded && tp <= numGpus) {
      // Verify the model's attention heads are divisible by TP
      if (config.numAttentionHeads % tp === 0 && config.numKeyValueHeads % tp === 0) {
        selectedTp = tp;
        break;
      }
    }
  }

  // If we couldn't find a valid TP, use the largest available
  if (selectedTp < minTpNeeded) {
    for (let i = validTps.length - 1; i >= 0; i--) {
      const tp = validTps[i];
      if (tp <= numGpus &&
          config.numAttentionHeads % tp === 0 &&
          config.numKeyValueHeads % tp === 0) {
        selectedTp = tp;
        break;
      }
    }
  }

  return selectedTp;
}
