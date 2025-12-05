// Tensor Parallel Linear Layers
// Implements column and row parallel linear layers for multi-GPU model parallelism

import { Tensor, DType } from "../tensor/tensor";
import { getCudaBackend, CudaBackend } from "../backend/cuda/bindings";
import { NcclCommunicator, NcclDataType } from "./nccl";

/**
 * Column Parallel Linear Layer
 *
 * Input: [batch, in_features]  (replicated across all ranks)
 * Weight: [out_features/TP, in_features]  (partitioned along output dimension)
 * Output: [batch, out_features/TP]  (partitioned, no all-reduce)
 *
 * Used for: QKV projections, MLP gate/up projections
 */
export class ColumnParallelLinear {
  private weight: Tensor;  // [out_features_per_rank, in_features]
  private bias: Tensor | null;
  private comm: NcclCommunicator;
  private cuda: CudaBackend;
  private inFeatures: number;
  private outFeaturesPerRank: number;
  private gatherOutput: boolean;

  constructor(
    weight: Tensor,
    bias: Tensor | null,
    comm: NcclCommunicator,
    gatherOutput: boolean = false
  ) {
    this.weight = weight;
    this.bias = bias;
    this.comm = comm;
    this.cuda = getCudaBackend();
    this.gatherOutput = gatherOutput;

    // weight is [out_features_per_rank, in_features]
    this.outFeaturesPerRank = weight.shape[0];
    this.inFeatures = weight.shape[1];
  }

  /**
   * Forward pass: input @ weight^T
   * Input is replicated across all ranks.
   * Output is partitioned (each rank computes a slice).
   */
  forward(input: Tensor): Tensor {
    const batchSize = input.shape[0];
    const seqLen = input.shape.length === 3 ? input.shape[1] : 1;
    const isFlat = input.shape.length === 2;

    const inputFlat = isFlat ? input : Tensor.fromPtr(
      input.ptr,
      [batchSize * seqLen, this.inFeatures],
      input.dtype
    );

    // Compute local output
    const output = Tensor.empty(
      [batchSize * seqLen, this.outFeaturesPerRank],
      { dtype: DType.Float16 }
    );

    this.cuda.gemmF16TransB(
      inputFlat.ptr,
      this.weight.ptr,
      output.ptr,
      batchSize * seqLen,
      this.outFeaturesPerRank,
      this.inFeatures
    );

    // Add bias if present
    if (this.bias) {
      // TODO: Add bias kernel
      // For now, skip bias
    }

    // If gather_output is True, gather outputs from all ranks
    if (this.gatherOutput) {
      const gathered = Tensor.empty(
        [batchSize * seqLen, this.outFeaturesPerRank * this.comm.size],
        { dtype: DType.Float16 }
      );

      this.comm.allGather(
        output.ptr,
        gathered.ptr,
        batchSize * seqLen * this.outFeaturesPerRank,
        NcclDataType.Float16
      );

      output.dispose();
      return isFlat ? gathered : Tensor.fromPtr(
        gathered.ptr,
        [batchSize, seqLen, this.outFeaturesPerRank * this.comm.size],
        DType.Float16
      );
    }

    // Return partitioned output
    return isFlat ? output : Tensor.fromPtr(
      output.ptr,
      [batchSize, seqLen, this.outFeaturesPerRank],
      DType.Float16
    );
  }
}

/**
 * Row Parallel Linear Layer
 *
 * Input: [batch, in_features/TP]  (partitioned across ranks)
 * Weight: [out_features, in_features/TP]  (partitioned along input dimension)
 * Output: [batch, out_features]  (replicated via all-reduce)
 *
 * Used for: Attention output projection, MLP down projection
 */
export class RowParallelLinear {
  private weight: Tensor;  // [out_features, in_features_per_rank]
  private bias: Tensor | null;
  private comm: NcclCommunicator;
  private cuda: CudaBackend;
  private inFeaturesPerRank: number;
  private outFeatures: number;
  private inputIsParallel: boolean;

  constructor(
    weight: Tensor,
    bias: Tensor | null,
    comm: NcclCommunicator,
    inputIsParallel: boolean = true
  ) {
    this.weight = weight;
    this.bias = bias;
    this.comm = comm;
    this.cuda = getCudaBackend();
    this.inputIsParallel = inputIsParallel;

    // weight is [out_features, in_features_per_rank]
    this.outFeatures = weight.shape[0];
    this.inFeaturesPerRank = weight.shape[1];
  }

  /**
   * Forward pass: input @ weight^T
   * Input is partitioned across ranks.
   * Output is reduced across all ranks (replicated).
   */
  forward(input: Tensor): Tensor {
    const batchSize = input.shape[0];
    const seqLen = input.shape.length === 3 ? input.shape[1] : 1;
    const isFlat = input.shape.length === 2;

    const inputFlat = isFlat ? input : Tensor.fromPtr(
      input.ptr,
      [batchSize * seqLen, this.inFeaturesPerRank],
      input.dtype
    );

    // Compute local output (partial result)
    const partialOutput = Tensor.empty(
      [batchSize * seqLen, this.outFeatures],
      { dtype: DType.Float16 }
    );

    this.cuda.gemmF16TransB(
      inputFlat.ptr,
      this.weight.ptr,
      partialOutput.ptr,
      batchSize * seqLen,
      this.outFeatures,
      this.inFeaturesPerRank
    );

    // All-reduce to sum partial results from all ranks
    const output = Tensor.empty(
      [batchSize * seqLen, this.outFeatures],
      { dtype: DType.Float16 }
    );

    this.comm.allReduceSum(
      partialOutput.ptr,
      output.ptr,
      batchSize * seqLen * this.outFeatures,
      NcclDataType.Float16
    );

    partialOutput.dispose();

    // Add bias if present (only on one rank to avoid duplication)
    if (this.bias && this.comm.myRank === 0) {
      // TODO: Add bias kernel
      // For now, skip bias
    }

    return isFlat ? output : Tensor.fromPtr(
      output.ptr,
      [batchSize, seqLen, this.outFeatures],
      DType.Float16
    );
  }
}

/**
 * Utility: Partition a weight tensor along a dimension for tensor parallelism.
 *
 * @param weight Full weight tensor [out_features, in_features]
 * @param dim Dimension to partition (0 = rows, 1 = columns)
 * @param rank Current GPU rank
 * @param worldSize Total number of GPUs
 * @returns Partitioned weight for this rank
 */
export function partitionWeight(
  weight: Tensor,
  dim: number,
  rank: number,
  worldSize: number
): Tensor {
  if (dim !== 0 && dim !== 1) {
    throw new Error(`Can only partition along dim 0 or 1, got ${dim}`);
  }

  const fullShape = weight.shape;
  const dimSize = fullShape[dim];

  if (dimSize % worldSize !== 0) {
    throw new Error(
      `Cannot evenly partition dimension ${dim} (size ${dimSize}) across ${worldSize} ranks`
    );
  }

  const chunkSize = dimSize / worldSize;
  const start = rank * chunkSize;
  const end = start + chunkSize;

  // Extract the slice for this rank
  // This requires copying data - in practice, weights would be loaded per-rank from disk
  const weightData = weight.toArray();

  let slicedData: Float32Array;
  let slicedShape: number[];

  if (dim === 0) {
    // Partition rows: [out_features, in_features] -> [out_features/TP, in_features]
    slicedShape = [chunkSize, fullShape[1]];
    slicedData = new Float32Array(chunkSize * fullShape[1]);

    for (let i = 0; i < chunkSize; i++) {
      const srcRow = start + i;
      const srcOffset = srcRow * fullShape[1];
      const dstOffset = i * fullShape[1];
      slicedData.set(weightData.subarray(srcOffset, srcOffset + fullShape[1]), dstOffset);
    }
  } else {
    // Partition columns: [out_features, in_features] -> [out_features, in_features/TP]
    slicedShape = [fullShape[0], chunkSize];
    slicedData = new Float32Array(fullShape[0] * chunkSize);

    for (let i = 0; i < fullShape[0]; i++) {
      const srcOffset = i * fullShape[1] + start;
      const dstOffset = i * chunkSize;
      for (let j = 0; j < chunkSize; j++) {
        slicedData[dstOffset + j] = weightData[srcOffset + j];
      }
    }
  }

  return Tensor.fromArray(slicedData, slicedShape, DType.Float16);
}
