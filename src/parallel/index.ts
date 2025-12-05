// Tensor Parallelism module
// Multi-GPU support for large models

export { NcclCommunicator, NcclDataType, NcclReduceOp, TensorParallelGroup } from "./nccl";
export { ColumnParallelLinear, RowParallelLinear, partitionWeight } from "./layers";
export { TensorParallelModelLoader, TensorParallelModel, TPLoaderOptions } from "./loader";
export { ShardCache, ShardCacheMetadata, ShardCacheInfo } from "./shard_cache";

// Re-export engine components
export { TensorParallelContext } from "../engine/tp_context";
export { InferenceEngine } from "../engine/engine_legacy";
export { DeviceContext, SingleDeviceContext, MultiDeviceContext, DeviceLocalKVCache } from "../engine/device_context";
