// Safetensors format parser in TypeScript
// Format: 8 bytes (header length, u64 LE) + JSON header + tensor data

export interface SafetensorTensorInfo {
  dtype: string;
  shape: number[];
  data_offsets: [number, number]; // [start, end] relative to data section
}

export interface SafetensorHeader {
  __metadata__?: Record<string, string>;
  [name: string]: SafetensorTensorInfo | Record<string, string> | undefined;
}

export interface ParsedSafetensor {
  header: SafetensorHeader;
  headerLength: number;
  dataOffset: number; // Offset where tensor data starts (8 + headerLength)
  tensors: Map<string, {
    name: string;
    dtype: string;
    shape: number[];
    byteOffset: number; // Absolute offset in file
    byteLength: number;
  }>;
}

/**
 * Parse safetensors header from a file or buffer.
 * Returns metadata about all tensors without loading the actual data.
 */
export function parseSafetensorsHeader(data: ArrayBuffer | Uint8Array): ParsedSafetensor {
  const view = data instanceof ArrayBuffer
    ? new DataView(data)
    : new DataView(data.buffer, data.byteOffset, data.byteLength);

  // Read header length (first 8 bytes, u64 little-endian)
  // Note: JavaScript can't handle full u64, but header lengths are always small
  const headerLengthLow = view.getUint32(0, true);
  const headerLengthHigh = view.getUint32(4, true);

  if (headerLengthHigh !== 0) {
    throw new Error("Header length too large (>4GB)");
  }

  const headerLength = headerLengthLow;
  const dataOffset = 8 + headerLength;

  // Ensure we have enough data for the header
  if (view.byteLength < dataOffset) {
    throw new Error(`Buffer too small: need ${dataOffset} bytes, got ${view.byteLength}`);
  }

  // Parse JSON header
  const headerBytes = new Uint8Array(view.buffer, view.byteOffset + 8, headerLength);
  const headerJson = new TextDecoder().decode(headerBytes);
  const header = JSON.parse(headerJson) as SafetensorHeader;

  // Build tensor map
  const tensors = new Map<string, {
    name: string;
    dtype: string;
    shape: number[];
    byteOffset: number;
    byteLength: number;
  }>();

  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__" || !info || typeof info !== "object" || !("dtype" in info)) {
      continue;
    }

    const tensorInfo = info as SafetensorTensorInfo;
    const [start, end] = tensorInfo.data_offsets;

    tensors.set(name, {
      name,
      dtype: tensorInfo.dtype,
      shape: tensorInfo.shape,
      byteOffset: dataOffset + start,
      byteLength: end - start,
    });
  }

  return {
    header,
    headerLength,
    dataOffset,
    tensors,
  };
}

/**
 * Get the byte size of a dtype.
 */
export function dtypeByteSize(dtype: string): number {
  const sizes: Record<string, number> = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "I8": 1,
    "BOOL": 1,
  };
  return sizes[dtype] ?? 4;
}

/**
 * Convert safetensors dtype to our DType enum string.
 */
export function dtypeToString(dtype: string): string {
  const mapping: Record<string, string> = {
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I32": "int32",
    "I64": "int64",
    "U8": "uint8",
    "I8": "int8",
  };
  return mapping[dtype] ?? "float32";
}
