// Test: Sampler (no CUDA or network required)
// Run with: bun test tests/unit/sampler.test.ts

import { expect, test, describe } from "bun:test";
import { Sampler, DEFAULT_SAMPLING_PARAMS, shouldStop } from "../../src/engine/sampler";

describe("Sampler", () => {
  test("greedy sampling (temperature=0) returns argmax", () => {
    const sampler = new Sampler({ temperature: 0 });

    const logits = new Float32Array([1.0, 5.0, 2.0, 3.0]);
    const token = sampler.sample(logits);

    expect(token).toBe(1); // Index of max value (5.0)
  });

  test("greedy sampling is deterministic", () => {
    const sampler = new Sampler({ temperature: 0 });

    const logits = new Float32Array([1.0, 2.0, 10.0, 3.0]);
    const tokens = Array.from({ length: 10 }, () => sampler.sample(logits));

    // All should be the same
    expect(new Set(tokens).size).toBe(1);
    expect(tokens[0]).toBe(2); // Index of max value (10.0)
  });

  test("seeded sampling is reproducible", () => {
    const seed = 42;

    const sampler1 = new Sampler({ temperature: 1.0, seed });
    const sampler2 = new Sampler({ temperature: 1.0, seed });

    const logits = new Float32Array([1.0, 1.0, 1.0, 1.0]); // Equal probabilities

    const tokens1 = Array.from({ length: 10 }, () => sampler1.sample(logits));
    const tokens2 = Array.from({ length: 10 }, () => sampler2.sample(logits));

    expect(tokens1).toEqual(tokens2);
  });

  test("top-k limits candidates", () => {
    const sampler = new Sampler({ temperature: 1.0, topK: 2, topP: 1.0, seed: 123 });

    // Create logits where top 2 are clearly the best
    const logits = new Float32Array([0.1, 10.0, 0.2, 9.0, 0.3]);

    // Sample many times
    const counts = new Map<number, number>();
    for (let i = 0; i < 100; i++) {
      const token = sampler.sample(logits);
      counts.set(token, (counts.get(token) || 0) + 1);
    }

    // Should only sample from indices 1 and 3 (top 2 values)
    for (const [token] of counts) {
      expect([1, 3]).toContain(token);
    }
  });

  test("repetition penalty reduces repeated token probability", () => {
    const sampler = new Sampler({ temperature: 0, repetitionPenalty: 2.0 });

    const logits = new Float32Array([5.0, 4.0, 3.0, 2.0]);

    // Without repetition, should pick index 0
    expect(sampler.sample(logits, [])).toBe(0);

    // With token 0 already generated, should pick next best
    expect(sampler.sample(logits, [0])).toBe(1);
  });

  test("default params are sensible", () => {
    expect(DEFAULT_SAMPLING_PARAMS.temperature).toBe(0.7);
    expect(DEFAULT_SAMPLING_PARAMS.topK).toBe(50);
    expect(DEFAULT_SAMPLING_PARAMS.topP).toBe(0.9);
    expect(DEFAULT_SAMPLING_PARAMS.repetitionPenalty).toBe(1.0);
  });
});

describe("Stopping Criteria", () => {
  test("stops at max tokens", () => {
    const criteria = { maxNewTokens: 5, eosTokenId: 2 };

    expect(shouldStop([1, 1, 1, 1], criteria)).toBe(false);
    expect(shouldStop([1, 1, 1, 1, 1], criteria)).toBe(true);
    expect(shouldStop([1, 1, 1, 1, 1, 1], criteria)).toBe(true);
  });

  test("stops at EOS token", () => {
    const criteria = { maxNewTokens: 100, eosTokenId: 2 };

    expect(shouldStop([1, 1, 1], criteria)).toBe(false);
    expect(shouldStop([1, 1, 2], criteria)).toBe(true);
    expect(shouldStop([1, 2, 1], criteria)).toBe(false); // EOS not at end
  });

  test("handles multiple EOS tokens", () => {
    const criteria = { maxNewTokens: 100, eosTokenId: [2, 128001] };

    expect(shouldStop([1, 1, 2], criteria)).toBe(true);
    expect(shouldStop([1, 1, 128001], criteria)).toBe(true);
    expect(shouldStop([1, 1, 3], criteria)).toBe(false);
  });

  test("empty generated tokens does not crash", () => {
    const criteria = { maxNewTokens: 10, eosTokenId: 2 };
    expect(shouldStop([], criteria)).toBe(false);
  });
});

describe("Probability Distribution", () => {
  test("temperature affects distribution spread", () => {
    // Low temperature = more peaked distribution
    const lowTempSampler = new Sampler({ temperature: 0.1, topK: 0, topP: 1.0, seed: 42 });
    // High temperature = more uniform distribution
    const highTempSampler = new Sampler({ temperature: 2.0, topK: 0, topP: 1.0, seed: 42 });

    const logits = new Float32Array([1.0, 2.0, 3.0, 4.0]);

    // Sample many times and check distribution
    const lowTempCounts = new Map<number, number>();
    const highTempCounts = new Map<number, number>();

    for (let i = 0; i < 1000; i++) {
      const lowToken = lowTempSampler.sample(logits);
      const highToken = highTempSampler.sample(logits);

      lowTempCounts.set(lowToken, (lowTempCounts.get(lowToken) || 0) + 1);
      highTempCounts.set(highToken, (highTempCounts.get(highToken) || 0) + 1);
    }

    // Low temp should heavily favor token 3 (highest logit)
    const lowTempMaxCount = Math.max(...lowTempCounts.values());
    expect(lowTempMaxCount).toBeGreaterThan(900); // Most samples should be the max

    // High temp should be more spread out (but still may favor the max somewhat)
    const highTempMaxCount = Math.max(...highTempCounts.values());
    expect(highTempMaxCount).toBeLessThan(950); // More distributed than low temp
  });
});
