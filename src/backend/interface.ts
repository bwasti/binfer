// Backend interface - abstraction for CUDA and WebGPU backends

import { Tensor, DType } from "../tensor/tensor";

/**
 * Backend interface that both CUDA and WebGPU implement.
 * Provides core tensor operations for inference.
 */
export interface Backend {
  // Device management
  isAvailable(): Promise<boolean>;
  getDeviceCount(): number;
  setDevice(device: number): void;

  // Memory operations
  allocate(size: number): bigint;
  free(ptr: bigint): void;
  memcpyH2D(dst: bigint, src: ArrayBuffer, size: number): void;
  memcpyD2H(dst: ArrayBuffer, src: bigint, size: number): void;
  synchronize(): void;

  // Matrix operations
  gemm(
    A: Tensor,
    B: Tensor,
    C: Tensor,
    alpha?: number,
    beta?: number
  ): void;

  gemmBatched(
    A: Tensor[],
    B: Tensor[],
    C: Tensor[],
    alpha?: number,
    beta?: number
  ): void;

  // Normalization
  rmsnorm(
    input: Tensor,
    weight: Tensor,
    output: Tensor,
    eps?: number
  ): void;

  // Position embeddings
  rotaryEmbedding(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    positionOffset?: number
  ): void;

  // Activations
  silu(input: Tensor, output: Tensor): void;
  gelu(input: Tensor, output: Tensor): void;
  swiglu(gate: Tensor, up: Tensor, output: Tensor): void;

  // Element-wise
  add(a: Tensor, b: Tensor, output: Tensor): void;
  mul(a: Tensor, b: Tensor, output: Tensor): void;

  // Softmax and sampling
  softmax(input: Tensor, output: Tensor, temperature?: number): void;
  topk(logits: Tensor, k: number): { values: Tensor; indices: Tensor };

  // Embedding
  embedding(weight: Tensor, inputIds: Tensor, output: Tensor): void;

  // MoE (Mixture of Experts) operations
  initMxfp4Tables?(): void;

  mxfp4Dequant?(
    blocks: Tensor,
    scales: Tensor,
    bias: Tensor | null,
    output: Tensor,
    numExperts: number,
    outFeatures: number,
    numBlocks: number,
    inFeatures: number
  ): void;

  moeRouterTopK?(
    hidden: Tensor,
    routerWeight: Tensor,
    routerBias: Tensor | null,
    expertIndices: Tensor,
    expertWeights: Tensor,
    batchSize: number,
    seqLen: number,
    hiddenSize: number,
    numExperts: number,
    topK: number
  ): void;

  moeSwiglu?(
    gateUp: Tensor,
    output: Tensor,
    batch: number,
    intermediateSize: number
  ): void;
}

/**
 * Supported backend types.
 */
export enum BackendType {
  CUDA = "cuda",
  WebGPU = "webgpu",
}

/**
 * Get the best available backend.
 */
export async function getDefaultBackend(): Promise<Backend> {
  // Try CUDA first
  const { CudaBackendWrapper } = await import("./cuda/wrapper");
  const cuda = new CudaBackendWrapper();
  if (await cuda.isAvailable()) {
    return cuda;
  }

  // Fall back to WebGPU (when implemented)
  throw new Error("No available backend found. CUDA is required.");
}
