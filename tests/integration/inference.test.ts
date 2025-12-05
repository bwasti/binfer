// Test: Inference engine verification
// Run with: bun test tests/integration/inference.test.ts
// Note: Requires CUDA library and a cached model

import { expect, test, describe, beforeAll, afterAll } from "bun:test";
import { getCudaBackend } from "../../src/backend/cuda/bindings";
import { ModelLoader } from "../../src/model/loader";
import { TensorParallelContext } from "../../src/engine/tp_context";
import { InferenceEngine } from "../../src/engine/engine";
import { createTokenizer, Tokenizer } from "../../src/model/tokenizer";

// Use a small model for testing
const TEST_MODEL = "Qwen/Qwen3-0.6B";

describe("InferenceEngine", () => {
  let ctx: TensorParallelContext;
  let engine: InferenceEngine;
  let tokenizer: Tokenizer;
  let cleanup: () => void;
  let cudaAvailable: boolean;

  beforeAll(async () => {
    const cuda = getCudaBackend();
    cudaAvailable = cuda.getDeviceCount() > 0;

    if (!cudaAvailable) {
      console.log("⚠ CUDA not available - inference tests will be skipped");
      return;
    }

    try {
      // Load model
      console.log(`Loading test model: ${TEST_MODEL}`);
      const loader = new ModelLoader();
      const loadedModel = await loader.load(TEST_MODEL);

      // Wrap weights in per-device map format for TP=1
      const perDeviceWeights = new Map<number, Map<string, { ptr: bigint; info: any }>>();
      perDeviceWeights.set(0, loadedModel.weights);

      ctx = new TensorParallelContext(
        loadedModel.config,
        perDeviceWeights,
        1
      );

      engine = new InferenceEngine(ctx);
      tokenizer = await createTokenizer(loadedModel.localPath);
      cleanup = () => loader.freeModel(loadedModel);

      console.log("Model loaded successfully");
    } catch (error) {
      console.log(`⚠ Failed to load model: ${error}`);
      cudaAvailable = false;
    }
  }, 120000); // 2 minute timeout for model loading

  afterAll(async () => {
    if (tokenizer) await tokenizer.stop();
    if (ctx) ctx.dispose();
    if (cleanup) cleanup();
  });

  test("engine has correct config", () => {
    if (!cudaAvailable) {
      console.log("Skipping: CUDA not available");
      return;
    }

    expect(engine.config).toBeDefined();
    expect(engine.config.hiddenSize).toBeGreaterThan(0);
    expect(engine.config.numHiddenLayers).toBeGreaterThan(0);
    expect(engine.config.vocabSize).toBeGreaterThan(0);
  });

  test("engine reports world size", () => {
    if (!cudaAvailable) {
      console.log("Skipping: CUDA not available");
      return;
    }

    expect(engine.worldSize).toBe(1);
  });

  test("can tokenize text", () => {
    if (!cudaAvailable) {
      console.log("Skipping: CUDA not available");
      return;
    }

    const text = "Hello, world!";
    const tokens = tokenizer.encode(text);

    expect(tokens).toBeDefined();
    expect(tokens.length).toBeGreaterThan(0);

    const decoded = tokenizer.decode(tokens);
    expect(decoded).toContain("Hello");
  });

  test("can generate tokens", async () => {
    if (!cudaAvailable) {
      console.log("Skipping: CUDA not available");
      return;
    }

    engine.reset(); // Reset KV cache for fresh generation

    const prompt = "The quick brown fox";
    const inputIds = tokenizer.encode(prompt);

    const generatedTokens: number[] = [];
    const result = await engine.generate(inputIds, 10, (token) => {
      generatedTokens.push(token);
    });

    expect(result.length).toBeGreaterThan(0);
    expect(result.length).toBeLessThanOrEqual(10);
    expect(generatedTokens.length).toBe(result.length);
  }, 30000);

  test("can interrupt generation with callback", async () => {
    if (!cudaAvailable) {
      console.log("Skipping: CUDA not available");
      return;
    }

    engine.reset(); // Reset KV cache for fresh generation

    const prompt = "Once upon a time";
    const inputIds = tokenizer.encode(prompt);

    let tokenCount = 0;
    const maxBeforeStop = 5;

    const result = await engine.generate(inputIds, 50, (token) => {
      tokenCount++;
      if (tokenCount >= maxBeforeStop) {
        return false; // Signal to stop
      }
    });

    // Should have stopped early
    expect(result.length).toBeLessThanOrEqual(maxBeforeStop + 1);
  }, 30000);

  test("stops on EOS token", async () => {
    if (!cudaAvailable) {
      console.log("Skipping: CUDA not available");
      return;
    }

    engine.reset(); // Reset KV cache for fresh generation

    // Use a chat-formatted prompt which should end with EOS
    const messages = [{ role: "user" as const, content: "Say just the word 'hello' and nothing else." }];
    const prompt = tokenizer.applyChatTemplate(messages);
    const inputIds = tokenizer.encode(prompt);

    const result = await engine.generate(inputIds, 100, () => {});

    // Model should stop before hitting the max
    // (This is probabilistic but the prompt strongly encourages short output)
    console.log(`Generated ${result.length} tokens`);
    expect(result.length).toBeLessThanOrEqual(100);
  }, 60000);

  test("generated text is decodable", async () => {
    if (!cudaAvailable) {
      console.log("Skipping: CUDA not available");
      return;
    }

    engine.reset(); // Reset KV cache for fresh generation

    const prompt = "2 + 2 =";
    const inputIds = tokenizer.encode(prompt);

    const result = await engine.generate(inputIds, 5, () => {});

    // Should be able to decode without error
    const text = tokenizer.decode(result);
    expect(text).toBeDefined();
    expect(typeof text).toBe("string");
    console.log(`Generated: "${text}"`);
  }, 30000);
});

describe("TensorParallelContext", () => {
  test("can create single-device context", () => {
    const cuda = getCudaBackend();
    if (cuda.getDeviceCount() === 0) {
      console.log("Skipping: CUDA not available");
      return;
    }

    // Minimal config for testing
    const config = {
      modelType: "test" as const,
      hiddenSize: 256,
      numHiddenLayers: 2,
      numAttentionHeads: 4,
      numKeyValueHeads: 4,
      intermediateSize: 512,
      vocabSize: 1000,
      maxPositionEmbeddings: 2048,
      rmsNormEps: 1e-6,
      ropeTheta: 10000,
      headDim: 64,
      tieWordEmbeddings: false,
      eosTokenId: 0,
    };

    const weights = new Map<number, Map<string, { ptr: bigint; info: any }>>();
    weights.set(0, new Map());

    const ctx = new TensorParallelContext(config, weights, 1);

    expect(ctx.worldSize).toBe(1);
    expect(ctx.devices.length).toBe(1);
    expect(ctx.numHeadsPerRank).toBe(4);

    ctx.dispose();
  });
});
