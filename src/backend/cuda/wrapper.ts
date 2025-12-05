// CUDA Backend wrapper implementing the Backend interface

import { Backend } from "../interface";
import { Tensor, DType } from "../../tensor/tensor";
import { CudaBackend, getCudaBackend } from "./bindings";

/**
 * CUDA backend wrapper that implements the Backend interface.
 */
export class CudaBackendWrapper implements Backend {
  private cuda: CudaBackend;

  constructor() {
    this.cuda = getCudaBackend();
  }

  async isAvailable(): Promise<boolean> {
    return this.cuda.isAvailable();
  }

  getDeviceCount(): number {
    return this.cuda.getDeviceCount();
  }

  setDevice(device: number): void {
    this.cuda.setDevice(device);
  }

  allocate(size: number): bigint {
    return this.cuda.malloc(size);
  }

  free(ptr: bigint): void {
    this.cuda.free(ptr);
  }

  memcpyH2D(dst: bigint, src: ArrayBuffer, size: number): void {
    this.cuda.memcpyH2D(dst, src, size);
  }

  memcpyD2H(dst: ArrayBuffer, src: bigint, size: number): void {
    this.cuda.memcpyD2H(dst, src, size);
  }

  synchronize(): void {
    this.cuda.synchronize();
  }

  gemm(
    A: Tensor,
    B: Tensor,
    C: Tensor,
    alpha: number = 1.0,
    beta: number = 0.0
  ): void {
    // A: [M, K], B: [K, N], C: [M, N]
    const M = A.shape[A.ndim - 2];
    const K = A.shape[A.ndim - 1];
    const N = B.shape[B.ndim - 1];

    if (A.dtype === DType.Float16) {
      this.cuda.gemmF16(A.ptr, B.ptr, C.ptr, M, N, K, alpha, beta);
    } else {
      throw new Error(`Unsupported dtype for GEMM: ${A.dtype}`);
    }
  }

  gemmBatched(
    A: Tensor[],
    B: Tensor[],
    C: Tensor[],
    alpha: number = 1.0,
    beta: number = 0.0
  ): void {
    // TODO: Implement batched GEMM
    // For now, fall back to sequential GEMMs
    for (let i = 0; i < A.length; i++) {
      this.gemm(A[i], B[i], C[i], alpha, beta);
    }
  }

  rmsnorm(
    input: Tensor,
    weight: Tensor,
    output: Tensor,
    eps: number = 1e-5
  ): void {
    // input: [batch, seq_len, hidden_size]
    const batchSize = input.shape.length === 3 ? input.shape[0] : 1;
    const seqLen = input.shape.length === 3 ? input.shape[1] : input.shape[0];
    const hiddenSize = input.shape[input.ndim - 1];

    this.cuda.rmsnormF16(
      input.ptr,
      weight.ptr,
      output.ptr,
      batchSize,
      seqLen,
      hiddenSize,
      eps
    );
  }

  rotaryEmbedding(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    positionOffset: number = 0
  ): void {
    // q: [batch, seq_len, num_heads, head_dim]
    // k: [batch, seq_len, num_kv_heads, head_dim]
    const batchSize = q.shape[0];
    const seqLen = q.shape[1];
    const numHeads = q.shape[2];
    const headDim = q.shape[3];
    const numKvHeads = k.shape[2];

    // Note: This calls the CUDA kernel which modifies q and k in-place
    // We would need to add this to the FFI bindings
    console.warn("rotaryEmbedding not yet implemented in FFI");
  }

  silu(input: Tensor, output: Tensor): void {
    this.cuda.siluF16(input.ptr, output.ptr, input.numel);
  }

  gelu(input: Tensor, output: Tensor): void {
    // TODO: Add GELU to FFI bindings
    console.warn("GELU not yet implemented in FFI, using SiLU");
    this.silu(input, output);
  }

  swiglu(gate: Tensor, up: Tensor, output: Tensor): void {
    this.cuda.swigluF16(gate.ptr, up.ptr, output.ptr, gate.numel);
  }

  add(a: Tensor, b: Tensor, output: Tensor): void {
    this.cuda.addF16(a.ptr, b.ptr, output.ptr, a.numel);
  }

  mul(a: Tensor, b: Tensor, output: Tensor): void {
    // TODO: Add mul to FFI bindings
    console.warn("mul not yet implemented in FFI");
  }

  softmax(input: Tensor, output: Tensor, temperature: number = 1.0): void {
    // input: [batch, seq_len, vocab_size] or [batch, vocab_size]
    const shape = input.shape;
    const vocabSize = shape[shape.length - 1];
    const seqLen = shape.length >= 2 ? shape[shape.length - 2] : 1;
    const batchSize = shape.length >= 3 ? shape[0] : 1;

    this.cuda.softmaxF16(
      input.ptr,
      output.ptr,
      batchSize,
      seqLen,
      vocabSize,
      temperature
    );
  }

  topk(logits: Tensor, k: number): { values: Tensor; indices: Tensor } {
    // TODO: Implement topk properly
    // For now, return empty tensors
    const batchSize = logits.shape[0];
    const values = Tensor.empty([batchSize, k], { dtype: DType.Float16 });
    const indices = Tensor.empty([batchSize, k], { dtype: DType.Int32 });
    return { values, indices };
  }

  embedding(weight: Tensor, inputIds: Tensor, output: Tensor): void {
    // weight: [vocab_size, hidden_size]
    // inputIds: [batch, seq_len]
    // output: [batch, seq_len, hidden_size]
    const batchSize = inputIds.shape[0];
    const seqLen = inputIds.shape[1];
    const vocabSize = weight.shape[0];
    const hiddenSize = weight.shape[1];

    this.cuda.embeddingF16(
      weight.ptr,
      inputIds.ptr,
      output.ptr,
      batchSize,
      seqLen,
      vocabSize,
      hiddenSize
    );
  }

  // MoE (Mixture of Experts) operations

  initMxfp4Tables(): void {
    this.cuda.initMxfp4Tables();
  }

  mxfp4Dequant(
    blocks: Tensor,
    scales: Tensor,
    bias: Tensor | null,
    output: Tensor,
    numExperts: number,
    outFeatures: number,
    numBlocks: number,
    inFeatures: number
  ): void {
    this.cuda.mxfp4Dequant(
      blocks.ptr,
      scales.ptr,
      bias?.ptr ?? BigInt(0),
      output.ptr,
      numExperts,
      outFeatures,
      numBlocks,
      inFeatures
    );
  }

  moeRouterTopK(
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
  ): void {
    this.cuda.moeRouterTopK(
      hidden.ptr,
      routerWeight.ptr,
      routerBias?.ptr ?? BigInt(0),
      expertIndices.ptr,
      expertWeights.ptr,
      batchSize,
      seqLen,
      hiddenSize,
      numExperts,
      topK
    );
  }

  moeSwiglu(
    gateUp: Tensor,
    output: Tensor,
    batch: number,
    intermediateSize: number
  ): void {
    this.cuda.moeSwiglu(gateUp.ptr, output.ptr, batch, intermediateSize);
  }
}
