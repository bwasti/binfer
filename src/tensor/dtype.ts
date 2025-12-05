// Data type definitions for tensors

export enum DType {
  Float32 = "float32",
  Float16 = "float16",
  BFloat16 = "bfloat16",
  Int32 = "int32",
  Int64 = "int64",
  Int8 = "int8",
  UInt8 = "uint8",
}

export const DTYPE_SIZES: Record<DType, number> = {
  [DType.Float32]: 4,
  [DType.Float16]: 2,
  [DType.BFloat16]: 2,
  [DType.Int32]: 4,
  [DType.Int64]: 8,
  [DType.Int8]: 1,
  [DType.UInt8]: 1,
};

export function dtypeFromString(s: string): DType {
  const normalized = s.toLowerCase().replace(/[^a-z0-9]/g, "");
  const mapping: Record<string, DType> = {
    float32: DType.Float32,
    f32: DType.Float32,
    float16: DType.Float16,
    f16: DType.Float16,
    half: DType.Float16,
    bfloat16: DType.BFloat16,
    bf16: DType.BFloat16,
    int32: DType.Int32,
    i32: DType.Int32,
    int64: DType.Int64,
    i64: DType.Int64,
    int8: DType.Int8,
    i8: DType.Int8,
    uint8: DType.UInt8,
    u8: DType.UInt8,
  };
  return mapping[normalized] ?? DType.Float32;
}

export function dtypeSize(dtype: DType): number {
  return DTYPE_SIZES[dtype];
}
