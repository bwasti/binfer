// Token sampling strategies for text generation

export interface SamplingParams {
  temperature: number;      // 0 = greedy, higher = more random
  topK: number;             // Keep only top K tokens (0 = disabled)
  topP: number;             // Nucleus sampling threshold (1.0 = disabled)
  repetitionPenalty: number; // Penalize repeated tokens (1.0 = disabled)
  seed?: number;            // Random seed for reproducibility
}

export const DEFAULT_SAMPLING_PARAMS: SamplingParams = {
  temperature: 0.7,
  topK: 50,
  topP: 0.9,
  repetitionPenalty: 1.0,
};

/**
 * Sampler for token generation.
 * Implements temperature, top-k, top-p (nucleus), and repetition penalty.
 */
export class Sampler {
  private params: SamplingParams;
  private rng: () => number;

  constructor(params: Partial<SamplingParams> = {}) {
    this.params = { ...DEFAULT_SAMPLING_PARAMS, ...params };

    // Simple seeded RNG (xorshift128+)
    if (this.params.seed !== undefined) {
      this.rng = this.createSeededRng(this.params.seed);
    } else {
      this.rng = Math.random;
    }
  }

  private createSeededRng(seed: number): () => number {
    let s0 = seed >>> 0;
    let s1 = (seed * 1103515245 + 12345) >>> 0;

    return () => {
      let t = s0;
      const s = s1;
      s0 = s;
      t ^= t << 23;
      t ^= t >>> 17;
      t ^= s ^ (s >>> 26);
      s1 = t;
      return (s0 + s1) / 4294967296;
    };
  }

  /**
   * Sample a token from logits.
   * Operates on CPU for simplicity (logits should be copied from GPU first).
   */
  sample(logits: Float32Array, generatedTokens: number[] = []): number {
    const { temperature, topK, topP, repetitionPenalty } = this.params;

    // Make a copy to avoid modifying the original
    const scores = new Float32Array(logits);

    // Apply repetition penalty
    if (repetitionPenalty !== 1.0) {
      for (const token of generatedTokens) {
        if (token >= 0 && token < scores.length) {
          if (scores[token] > 0) {
            scores[token] /= repetitionPenalty;
          } else {
            scores[token] *= repetitionPenalty;
          }
        }
      }
    }

    // Greedy decoding
    if (temperature === 0) {
      return this.argmax(scores);
    }

    // Apply temperature
    for (let i = 0; i < scores.length; i++) {
      scores[i] /= temperature;
    }

    // Get sorted indices
    const indices = this.argsort(scores);

    // Apply top-k
    let cutoff = scores.length;
    if (topK > 0 && topK < cutoff) {
      cutoff = topK;
    }

    // Apply top-p (nucleus sampling)
    if (topP < 1.0) {
      // Compute softmax for top-k tokens
      const topScores = new Float32Array(cutoff);
      let maxScore = -Infinity;
      for (let i = 0; i < cutoff; i++) {
        topScores[i] = scores[indices[i]];
        maxScore = Math.max(maxScore, topScores[i]);
      }

      // Stable softmax
      let sumExp = 0;
      for (let i = 0; i < cutoff; i++) {
        topScores[i] = Math.exp(topScores[i] - maxScore);
        sumExp += topScores[i];
      }

      // Find nucleus cutoff
      let cumProb = 0;
      for (let i = 0; i < cutoff; i++) {
        cumProb += topScores[i] / sumExp;
        if (cumProb >= topP) {
          cutoff = i + 1;
          break;
        }
      }
    }

    // Sample from the filtered distribution
    const filteredIndices = indices.slice(0, cutoff);
    const filteredScores = new Float32Array(cutoff);
    let maxScore = -Infinity;

    for (let i = 0; i < cutoff; i++) {
      filteredScores[i] = scores[filteredIndices[i]];
      maxScore = Math.max(maxScore, filteredScores[i]);
    }

    // Softmax
    let sumExp = 0;
    for (let i = 0; i < cutoff; i++) {
      filteredScores[i] = Math.exp(filteredScores[i] - maxScore);
      sumExp += filteredScores[i];
    }

    // Normalize to probabilities
    for (let i = 0; i < cutoff; i++) {
      filteredScores[i] /= sumExp;
    }

    // Sample
    const r = this.rng();
    let cumProb = 0;
    for (let i = 0; i < cutoff; i++) {
      cumProb += filteredScores[i];
      if (r < cumProb) {
        return filteredIndices[i];
      }
    }

    // Fallback to last token
    return filteredIndices[cutoff - 1];
  }

  /**
   * Sample multiple tokens (for batch generation).
   */
  sampleBatch(
    logitsBatch: Float32Array[],
    generatedTokensBatch: number[][] = []
  ): number[] {
    return logitsBatch.map((logits, i) =>
      this.sample(logits, generatedTokensBatch[i] || [])
    );
  }

  private argmax(arr: Float32Array): number {
    let maxIdx = 0;
    let maxVal = arr[0];
    for (let i = 1; i < arr.length; i++) {
      if (arr[i] > maxVal) {
        maxVal = arr[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  private argsort(arr: Float32Array): number[] {
    const indices = Array.from({ length: arr.length }, (_, i) => i);
    indices.sort((a, b) => arr[b] - arr[a]); // Descending
    return indices;
  }
}

/**
 * Stopping criteria for generation.
 */
export interface StoppingCriteria {
  maxNewTokens: number;
  eosTokenId: number | number[];
  stopStrings?: string[];
}

export function shouldStop(
  generatedTokens: number[],
  criteria: StoppingCriteria
): boolean {
  // Check max tokens
  if (generatedTokens.length >= criteria.maxNewTokens) {
    return true;
  }

  // Check EOS token
  if (generatedTokens.length > 0) {
    const lastToken = generatedTokens[generatedTokens.length - 1];
    const eosTokens = Array.isArray(criteria.eosTokenId)
      ? criteria.eosTokenId
      : [criteria.eosTokenId];

    if (eosTokens.includes(lastToken)) {
      return true;
    }
  }

  // Note: stopStrings would require decoding tokens first
  // We'd need tokenizer access for that

  return false;
}
