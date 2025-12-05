// Tensor Parallel Context - Manages multi-device orchestration
// TP=1 is handled as a special case with no-op communication

import { Tensor, DType } from "../tensor/tensor";
import { IKVCache, createKVCache } from "../kv/manager";
import { getCudaBackend, CudaBackend } from "../backend/cuda/bindings";
import { LlamaConfig, RopeScaling } from "../model/config";
import {
  DeviceContext,
  SingleDeviceContext,
  MultiDeviceContext,
  createDeviceLocalKVCache,
} from "./device_context";
import { TensorParallelGroup, NcclDataType } from "../parallel/nccl";

export interface TensorParallelContextConfig {
  worldSize: number;
  config: LlamaConfig;
}

/**
 * Tensor Parallel Context - orchestrates inference across multiple GPUs.
 *
 * For TP=1 (single GPU):
 * - Uses SingleDeviceContext with memory pool
 * - All-reduce is a no-op
 *
 * For TP>1 (multi-GPU):
 * - Uses MultiDeviceContext with direct cuda.malloc per device
 * - Uses NCCL for all-reduce after row-parallel layers
 */
export class TensorParallelContext {
  readonly worldSize: number;
  readonly config: LlamaConfig;
  readonly devices: DeviceContext[];

  // NCCL communicator (null for TP=1)
  private tpGroup: TensorParallelGroup | null = null;
  private cuda: CudaBackend;

  // Per-rank dimensions (adjusted for tensor parallelism)
  readonly numHeadsPerRank: number;
  readonly numKvHeadsPerRank: number;
  readonly intermediateSizePerRank: number;

  constructor(
    config: LlamaConfig,
    perDeviceWeights: Map<number, Map<string, { ptr: bigint; info: any }>>,
    worldSize: number = 1
  ) {
    this.worldSize = worldSize;
    this.config = config;
    this.cuda = getCudaBackend();
    this.devices = [];

    // Validate divisibility
    if (config.numAttentionHeads % worldSize !== 0) {
      throw new Error(
        `numAttentionHeads (${config.numAttentionHeads}) must be divisible by worldSize (${worldSize})`
      );
    }
    if (config.numKeyValueHeads % worldSize !== 0) {
      throw new Error(
        `numKeyValueHeads (${config.numKeyValueHeads}) must be divisible by worldSize (${worldSize})`
      );
    }
    if (config.intermediateSize % worldSize !== 0) {
      throw new Error(
        `intermediateSize (${config.intermediateSize}) must be divisible by worldSize (${worldSize})`
      );
    }

    // Compute per-rank dimensions
    this.numHeadsPerRank = config.numAttentionHeads / worldSize;
    this.numKvHeadsPerRank = config.numKeyValueHeads / worldSize;
    this.intermediateSizePerRank = config.intermediateSize / worldSize;

    // Determine dtype from config
    const dtype = config.dtype === "bfloat16" ? DType.BFloat16 : DType.Float16;

    // Create device contexts
    if (worldSize === 1) {
      // Single GPU - use simplified context
      const weights = perDeviceWeights.get(0) || new Map();
      this.devices.push(new SingleDeviceContext(weights, dtype));
    } else {
      // Multi-GPU - create context for each device
      for (let device = 0; device < worldSize; device++) {
        const weights = perDeviceWeights.get(device);
        if (!weights) {
          throw new Error(`No weights provided for device ${device}`);
        }
        this.devices.push(new MultiDeviceContext(device, weights, this.cuda, dtype));
      }
    }
  }

  /**
   * Initialize NCCL communicators for multi-GPU.
   * Must be called before inference for TP>1.
   */
  async initCommunicators(logger?: (msg: string) => void): Promise<void> {
    if (this.worldSize === 1) {
      return; // No communication needed for single GPU
    }

    this.tpGroup = new TensorParallelGroup(this.worldSize, logger);
    await this.tpGroup.init();
  }

  /**
   * Set a pre-initialized NCCL group (for parallel initialization).
   */
  setCommunicators(tpGroup: TensorParallelGroup): void {
    this.tpGroup = tpGroup;
  }

  /**
   * Compute YaRN-scaled frequencies for RoPE.
   * Matches HuggingFace's _compute_yarn_parameters implementation.
   */
  private computeYarnFreqs(
    headDim: number,
    ropeTheta: number,
    scaling: RopeScaling
  ): { freqs: Float32Array; attentionScaling: number } {
    const halfDim = headDim / 2;

    const factor = scaling.factor;
    const originalMaxLen = scaling.originalMaxPositionEmbeddings ?? 4096;
    const betaFast = scaling.betaFast ?? 32.0;
    const betaSlow = scaling.betaSlow ?? 1.0;

    // Compute attention_factor (mscale)
    // Default: get_mscale(factor) = 0.1 * log(factor) + 1.0
    let attentionFactor: number;
    if (factor <= 1) {
      attentionFactor = 1.0;
    } else {
      attentionFactor = 0.1 * Math.log(factor) + 1.0;
    }

    // Helper: find dimension from number of rotations
    const findCorrectionDim = (numRotations: number): number => {
      return (headDim * Math.log(originalMaxLen / (numRotations * 2 * Math.PI))) / (2 * Math.log(ropeTheta));
    };

    // Find correction range
    const low = Math.max(findCorrectionDim(betaFast), 0);
    const high = Math.min(findCorrectionDim(betaSlow), headDim - 1);

    // Linear ramp function
    const linearRamp = (min: number, max: number, i: number): number => {
      if (min === max) {
        max += 0.001; // Prevent singularity
      }
      const t = (i - min) / (max - min);
      return Math.max(0, Math.min(1, t));
    };

    // Compute inv_freq with YaRN scaling
    const freqs = new Float32Array(halfDim);
    for (let i = 0; i < halfDim; i++) {
      const posFreq = Math.pow(ropeTheta, (2 * i) / headDim);
      const invFreqExtrapolation = 1.0 / posFreq;  // Original (no scaling)
      const invFreqInterpolation = 1.0 / (factor * posFreq);  // Scaled

      // inv_freq_extrapolation_factor = 1 - linear_ramp
      const extrapolationFactor = 1 - linearRamp(low, high, i);

      // inv_freq = interpolation * (1 - extrapolation_factor) + extrapolation * extrapolation_factor
      freqs[i] = invFreqInterpolation * (1 - extrapolationFactor) + invFreqExtrapolation * extrapolationFactor;
    }

    return { freqs, attentionScaling: attentionFactor };
  }

  /**
   * Initialize RoPE cache on all devices.
   */
  initRopeCache(maxSeqLen: number): void {
    const { headDim, ropeTheta, ropeScaling } = this.config;
    const halfDim = headDim / 2;
    const dtype = this.config.dtype === "bfloat16" ? DType.BFloat16 : DType.Float16;

    // Compute RoPE frequencies (with YaRN scaling if configured)
    let freqs: Float32Array;
    let attentionScaling = 1.0;
    if (ropeScaling && ropeScaling.type === "yarn") {
      const result = this.computeYarnFreqs(headDim, ropeTheta, ropeScaling);
      freqs = result.freqs;
      attentionScaling = result.attentionScaling;
    } else {
      // Standard RoPE
      freqs = new Float32Array(halfDim);
      for (let i = 0; i < halfDim; i++) {
        freqs[i] = 1.0 / Math.pow(ropeTheta, (2 * i) / headDim);
      }
    }

    const cosData = new Float32Array(maxSeqLen * halfDim);
    const sinData = new Float32Array(maxSeqLen * halfDim);

    for (let pos = 0; pos < maxSeqLen; pos++) {
      for (let i = 0; i < halfDim; i++) {
        const angle = pos * freqs[i];
        // Apply attention scaling (YaRN mscale)
        cosData[pos * halfDim + i] = Math.cos(angle) * attentionScaling;
        sinData[pos * halfDim + i] = Math.sin(angle) * attentionScaling;
      }
    }

    // For TP=1, use Tensor.fromArray which handles conversion
    if (this.worldSize === 1) {
      const ctx = this.devices[0];
      ctx.ropeCache = {
        cos: Tensor.fromArray(cosData, [maxSeqLen, halfDim], dtype),
        sin: Tensor.fromArray(sinData, [maxSeqLen, halfDim], dtype),
      };
      return;
    }

    // For TP>1, convert to appropriate half type and copy to each device
    const halfCosData = new Uint16Array(maxSeqLen * halfDim);
    const halfSinData = new Uint16Array(maxSeqLen * halfDim);
    for (let i = 0; i < cosData.length; i++) {
      halfCosData[i] = dtype === DType.BFloat16 ? floatToBFloat16(cosData[i]) : floatToHalf(cosData[i]);
      halfSinData[i] = dtype === DType.BFloat16 ? floatToBFloat16(sinData[i]) : floatToHalf(sinData[i]);
    }

    const size = maxSeqLen * halfDim * 2; // fp16/bf16 = 2 bytes

    for (const ctx of this.devices) {
      ctx.setActive();
      const cosPtr = this.cuda.malloc(size);
      const sinPtr = this.cuda.malloc(size);
      this.cuda.memcpyH2D(cosPtr, halfCosData.buffer, size);
      this.cuda.memcpyH2D(sinPtr, halfSinData.buffer, size);

      ctx.ropeCache = {
        cos: Tensor.fromPtr(cosPtr, [maxSeqLen, halfDim], dtype, ctx.device),
        sin: Tensor.fromPtr(sinPtr, [maxSeqLen, halfDim], dtype, ctx.device),
      };
    }

    // Reset to device 0
    this.cuda.setDevice(0);
  }

  /**
   * Initialize KV caches on all devices.
   */
  initKVCaches(maxSeqLen: number, batchSize: number = 1): void {
    const { numHiddenLayers, headDim } = this.config;

    // Determine dtype from config
    const dtype = this.config.dtype === "bfloat16" ? DType.BFloat16 : DType.Float16;

    if (this.worldSize === 1) {
      // Single GPU - use standard KV cache
      const ctx = this.devices[0];
      ctx.kvCache = createKVCache(
        numHiddenLayers,
        this.numKvHeadsPerRank,
        headDim,
        maxSeqLen,
        batchSize,
        dtype
      );
      return;
    }

    // Multi-GPU - create device-local KV caches
    for (const ctx of this.devices) {
      ctx.setActive();
      ctx.kvCache = createDeviceLocalKVCache(
        numHiddenLayers,
        this.numKvHeadsPerRank,
        headDim,
        maxSeqLen,
        batchSize,
        ctx.device,
        this.cuda,
        dtype
      );
    }

    this.cuda.setDevice(0);
  }

  /**
   * All-reduce sum across all devices.
   * No-op for TP=1.
   */
  allReduceSum(tensors: Map<number, Tensor>): void {
    if (this.worldSize === 1 || !this.tpGroup) {
      return; // No-op for single GPU
    }

    const buffers: { send: bigint; recv: bigint; device: number }[] = [];
    const count = tensors.get(0)!.numel;

    tensors.forEach((tensor, device) => {
      buffers.push({
        send: tensor.ptr,
        recv: tensor.ptr, // In-place
        device,
      });
    });

    // Use correct NCCL dtype based on model dtype
    const ncclDtype = this.config.dtype === "bfloat16" ? NcclDataType.BFloat16 : NcclDataType.Float16;
    this.tpGroup.allReduceSum(buffers, count, ncclDtype);
  }

  /**
   * Synchronize all devices.
   */
  synchronize(): void {
    if (this.worldSize === 1) {
      this.cuda.synchronize();
      return;
    }

    for (const ctx of this.devices) {
      ctx.setActive();
      this.cuda.synchronize();
    }
  }

  /**
   * Reset all KV caches.
   */
  resetKVCaches(): void {
    for (const ctx of this.devices) {
      ctx.kvCache?.reset();
    }
  }

  /**
   * Dispose all resources.
   */
  dispose(): void {
    for (const ctx of this.devices) {
      ctx.setActive();
      ctx.kvCache?.dispose();
      if (ctx.ropeCache) {
        // RoPE cache tensors were allocated with direct cuda.malloc (not through pool)
        // so we must free them directly, not through ctx.free() which uses the pool
        if (this.worldSize === 1) {
          ctx.free(ctx.ropeCache.cos);
          ctx.free(ctx.ropeCache.sin);
        } else {
          this.cuda.free(ctx.ropeCache.cos.ptr);
          this.cuda.free(ctx.ropeCache.sin.ptr);
        }
      }
    }

    this.tpGroup?.dispose();
    this.cuda.setDevice(0);
  }
}

/**
 * Convert float32 to float16.
 */
function floatToHalf(value: number): number {
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);

  floatView[0] = value;
  const x = int32View[0];

  const sign = (x >> 31) & 1;
  const exp = (x >> 23) & 0xff;
  const frac = x & 0x7fffff;

  let hSign = sign << 15;
  let hExp: number;
  let hFrac: number;

  if (exp === 0) {
    hExp = 0;
    hFrac = 0;
  } else if (exp === 0xff) {
    hExp = 0x1f;
    hFrac = frac ? 0x200 : 0;
  } else {
    const newExp = exp - 127 + 15;
    if (newExp >= 0x1f) {
      hExp = 0x1f;
      hFrac = 0;
    } else if (newExp <= 0) {
      hExp = 0;
      hFrac = 0;
    } else {
      hExp = newExp;
      hFrac = frac >> 13;
    }
  }

  return hSign | (hExp << 10) | hFrac;
}

/**
 * Convert float32 to bfloat16.
 * BF16 uses 1 sign bit, 8 exponent bits, and 7 mantissa bits.
 * It's essentially truncating the lower 16 bits of float32.
 */
function floatToBFloat16(value: number): number {
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);

  floatView[0] = value;
  const x = int32View[0];

  // BF16 is just the upper 16 bits of float32
  return (x >> 16) & 0xffff;
}
