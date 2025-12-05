// Test: Config parser (no CUDA or network required)
// Run with: bun test tests/unit/config.test.ts

import { expect, test, describe } from "bun:test";
import { parseLlamaConfig, estimateMemory, LlamaConfig } from "../../src/model/config";

// Sample Llama-3.2-1B config (copied from HuggingFace)
const LLAMA_3_2_1B_CONFIG = {
  architectures: ["LlamaForCausalLM"],
  attention_bias: false,
  attention_dropout: 0.0,
  bos_token_id: 128000,
  eos_token_id: 128001,
  hidden_act: "silu",
  hidden_size: 2048,
  initializer_range: 0.02,
  intermediate_size: 8192,
  max_position_embeddings: 131072,
  model_type: "llama",
  num_attention_heads: 32,
  num_hidden_layers: 16,
  num_key_value_heads: 8,
  rms_norm_eps: 1e-5,
  rope_scaling: {
    factor: 32.0,
    high_freq_factor: 4.0,
    low_freq_factor: 1.0,
    original_max_position_embeddings: 8192,
    rope_type: "llama3"
  },
  rope_theta: 500000.0,
  tie_word_embeddings: true,
  torch_dtype: "bfloat16",
  vocab_size: 128256
};

// Sample Llama-3.2-3B config
const LLAMA_3_2_3B_CONFIG = {
  architectures: ["LlamaForCausalLM"],
  hidden_size: 3072,
  intermediate_size: 8192,
  num_attention_heads: 24,
  num_hidden_layers: 28,
  num_key_value_heads: 8,
  vocab_size: 128256,
  max_position_embeddings: 131072,
  rms_norm_eps: 1e-5,
  rope_theta: 500000.0,
  hidden_act: "silu",
  tie_word_embeddings: true,
  bos_token_id: 128000,
  eos_token_id: 128001,
};

describe("LlamaConfig Parser", () => {
  test("parses Llama-3.2-1B config correctly", () => {
    const config = parseLlamaConfig(LLAMA_3_2_1B_CONFIG);

    expect(config.hiddenSize).toBe(2048);
    expect(config.intermediateSize).toBe(8192);
    expect(config.numHiddenLayers).toBe(16);
    expect(config.numAttentionHeads).toBe(32);
    expect(config.numKeyValueHeads).toBe(8);
    expect(config.vocabSize).toBe(128256);
    expect(config.rmsNormEps).toBe(1e-5);
    expect(config.ropeTheta).toBe(500000.0);
    expect(config.hiddenAct).toBe("silu");
    expect(config.tieWordEmbeddings).toBe(true);
  });

  test("parses Llama-3.2-3B config correctly", () => {
    const config = parseLlamaConfig(LLAMA_3_2_3B_CONFIG);

    expect(config.hiddenSize).toBe(3072);
    expect(config.numHiddenLayers).toBe(28);
    expect(config.numAttentionHeads).toBe(24);
    expect(config.numKeyValueHeads).toBe(8);
  });

  test("calculates head_dim correctly", () => {
    const config = parseLlamaConfig(LLAMA_3_2_1B_CONFIG);
    // head_dim = hidden_size / num_attention_heads = 2048 / 32 = 64
    expect(config.headDim).toBe(64);
  });

  test("parses rope_scaling", () => {
    const config = parseLlamaConfig(LLAMA_3_2_1B_CONFIG);

    expect(config.ropeScaling).not.toBeNull();
    expect(config.ropeScaling?.type).toBe("llama3");
    expect(config.ropeScaling?.factor).toBe(32.0);
    expect(config.ropeScaling?.highFreqFactor).toBe(4.0);
    expect(config.ropeScaling?.lowFreqFactor).toBe(1.0);
  });

  test("handles missing rope_scaling gracefully", () => {
    const configWithoutScaling = { ...LLAMA_3_2_3B_CONFIG };
    const config = parseLlamaConfig(configWithoutScaling);
    expect(config.ropeScaling).toBeNull();
  });
});

describe("Memory Estimation", () => {
  test("estimates Llama-3.2-1B memory correctly", () => {
    const config = parseLlamaConfig(LLAMA_3_2_1B_CONFIG);
    const mem = estimateMemory(config);

    // 1B model should be ~1.2B params, ~2.4GB in fp16
    expect(mem.parametersB).toBeGreaterThan(1.0);
    expect(mem.parametersB).toBeLessThan(2.0);
    expect(mem.memoryGB).toBeGreaterThan(2.0);
    expect(mem.memoryGB).toBeLessThan(4.0);

    console.log(`Llama-3.2-1B: ${mem.parametersB.toFixed(2)}B params, ${mem.memoryGB.toFixed(2)}GB`);
  });

  test("estimates Llama-3.2-3B memory correctly", () => {
    const config = parseLlamaConfig(LLAMA_3_2_3B_CONFIG);
    const mem = estimateMemory(config);

    // 3B model should be ~3B params, ~6GB in fp16
    expect(mem.parametersB).toBeGreaterThan(2.5);
    expect(mem.parametersB).toBeLessThan(4.0);
    expect(mem.memoryGB).toBeGreaterThan(5.0);
    expect(mem.memoryGB).toBeLessThan(8.0);

    console.log(`Llama-3.2-3B: ${mem.parametersB.toFixed(2)}B params, ${mem.memoryGB.toFixed(2)}GB`);
  });

  test("estimates KV cache per token", () => {
    const config = parseLlamaConfig(LLAMA_3_2_1B_CONFIG);
    const mem = estimateMemory(config);

    // KV cache should be reasonable (< 1MB per token for 1B model)
    expect(mem.kvCachePerTokenMB).toBeGreaterThan(0);
    expect(mem.kvCachePerTokenMB).toBeLessThan(1.0);

    console.log(`KV cache per token: ${(mem.kvCachePerTokenMB * 1024).toFixed(2)}KB`);
  });
});
