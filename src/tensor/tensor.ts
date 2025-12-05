// Tensor abstraction for GPU-backed tensors

import { DType, dtypeSize } from "./dtype";
import { MemoryPool, getGlobalMemoryPool } from "./memory";
import { getCudaBackend, CudaBackend } from "../backend/cuda/bindings";

export interface TensorOptions {
  dtype?: DType;
  device?: number;
  requiresGrad?: boolean;
}

/**
 * GPU-backed tensor class.
 * Provides a high-level interface for tensor operations.
 */
export class Tensor {
  readonly shape: number[];
  readonly dtype: DType;
  readonly device: number;
  readonly ptr: bigint;

  private pool: MemoryPool;
  private cuda: CudaBackend;
  private _numel: number;
  private _strides: number[];
  private _ownsMemory: boolean;

  private constructor(
    shape: number[],
    dtype: DType,
    device: number,
    ptr: bigint,
    ownsMemory: boolean,
    pool?: MemoryPool
  ) {
    this.shape = shape;
    this.dtype = dtype;
    this.device = device;
    this.ptr = ptr;
    this._ownsMemory = ownsMemory;
    this.pool = pool ?? getGlobalMemoryPool();
    this.cuda = getCudaBackend();

    // Calculate numel and strides
    this._numel = shape.reduce((a, b) => a * b, 1);
    this._strides = this.computeStrides(shape);
  }

  private computeStrides(shape: number[]): number[] {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  /**
   * Number of elements in the tensor.
   */
  get numel(): number {
    return this._numel;
  }

  /**
   * Size in bytes.
   */
  get nbytes(): number {
    return this._numel * dtypeSize(this.dtype);
  }

  /**
   * Strides for each dimension.
   */
  get strides(): number[] {
    return this._strides;
  }

  /**
   * Number of dimensions.
   */
  get ndim(): number {
    return this.shape.length;
  }

  /**
   * Create a new tensor with allocated GPU memory.
   */
  static empty(
    shape: number[],
    options: TensorOptions = {}
  ): Tensor {
    const dtype = options.dtype ?? DType.Float16;
    const device = options.device ?? 0;
    const pool = getGlobalMemoryPool();

    const numel = shape.reduce((a, b) => a * b, 1);
    const size = numel * dtypeSize(dtype);
    const ptr = pool.alloc(size);

    return new Tensor(shape, dtype, device, ptr, true, pool);
  }

  /**
   * Create a tensor from existing GPU pointer (does not take ownership).
   */
  static fromPtr(
    ptr: bigint,
    shape: number[],
    dtype: DType = DType.Float16,
    device: number = 0
  ): Tensor {
    return new Tensor(shape, dtype, device, ptr, false);
  }

  /**
   * Create a tensor and copy data from CPU.
   * Handles conversion from Float32Array to Float16/BFloat16 if needed.
   */
  static fromArray(
    data: Float32Array | Float16Array | Int32Array | Uint8Array,
    shape: number[],
    dtype: DType = DType.Float16
  ): Tensor {
    const tensor = Tensor.empty(shape, { dtype });

    // Convert Float32Array to Float16 if tensor dtype is Float16
    if (dtype === DType.Float16 && data instanceof Float32Array) {
      const f16Data = new Float16Array(data.length);
      for (let i = 0; i < data.length; i++) {
        f16Data[i] = data[i];
      }
      tensor.copyFromHost(f16Data);
    } else if (dtype === DType.BFloat16 && data instanceof Float32Array) {
      // Convert Float32Array to BFloat16
      const bf16Data = new Uint16Array(data.length);
      const floatView = new Float32Array(1);
      const int32View = new Int32Array(floatView.buffer);
      for (let i = 0; i < data.length; i++) {
        floatView[0] = data[i];
        bf16Data[i] = (int32View[0] >> 16) & 0xffff;
      }
      tensor.copyFromHost(bf16Data);
    } else {
      tensor.copyFromHost(data);
    }
    return tensor;
  }

  /**
   * Create a tensor filled with zeros.
   */
  static zeros(shape: number[], options: TensorOptions = {}): Tensor {
    const tensor = Tensor.empty(shape, options);
    // TODO: Add CUDA memset kernel
    // For now, copy zeros from CPU
    const numel = tensor.numel;
    const dtype = options.dtype ?? DType.Float16;
    if (dtype === DType.Float16) {
      tensor.copyFromHost(new Float16Array(numel));
    } else {
      tensor.copyFromHost(new Float32Array(numel));
    }
    return tensor;
  }

  /**
   * Create a tensor filled with ones.
   */
  static ones(shape: number[], options: TensorOptions = {}): Tensor {
    const tensor = Tensor.empty(shape, options);
    const numel = tensor.numel;
    const dtype = options.dtype ?? DType.Float16;
    if (dtype === DType.Float16) {
      const ones = new Float16Array(numel);
      for (let i = 0; i < numel; i++) ones[i] = 1.0;
      tensor.copyFromHost(ones);
    } else {
      const ones = new Float32Array(numel).fill(1.0);
      tensor.copyFromHost(ones);
    }
    return tensor;
  }

  /**
   * Copy data from host to device.
   */
  copyFromHost(data: ArrayBufferView): void {
    if (data.byteLength !== this.nbytes) {
      throw new Error(
        `Size mismatch: expected ${this.nbytes} bytes, got ${data.byteLength}`
      );
    }
    this.cuda.memcpyH2D(this.ptr, data.buffer, this.nbytes);
  }

  /**
   * Copy data from device to host.
   */
  toArray(): Float32Array {
    // Sync to ensure all pending GPU ops complete
    this.cuda.synchronize();

    const buffer = new ArrayBuffer(this.nbytes);
    this.cuda.memcpyD2H(buffer, this.ptr, this.nbytes);

    // Convert to Float32Array based on dtype
    if (this.dtype === DType.Float32) {
      return new Float32Array(buffer);
    } else if (this.dtype === DType.Float16) {
      // Convert fp16 to fp32
      const fp16 = new Uint16Array(buffer);
      const fp32 = new Float32Array(this._numel);
      for (let i = 0; i < this._numel; i++) {
        fp32[i] = fp16ToFp32(fp16[i]);
      }
      return fp32;
    } else if (this.dtype === DType.BFloat16) {
      // Convert bf16 to fp32
      const bf16 = new Uint16Array(buffer);
      const fp32 = new Float32Array(this._numel);
      const floatView = new Float32Array(1);
      const int32View = new Int32Array(floatView.buffer);
      for (let i = 0; i < this._numel; i++) {
        // BF16 is upper 16 bits of float32, so shift left by 16
        int32View[0] = bf16[i] << 16;
        fp32[i] = floatView[0];
      }
      return fp32;
    } else {
      // For other types, just reinterpret as float32
      return new Float32Array(buffer);
    }
  }

  /**
   * Reshape the tensor (must have same numel).
   */
  reshape(newShape: number[]): Tensor {
    const newNumel = newShape.reduce((a, b) => a * b, 1);
    if (newNumel !== this._numel) {
      throw new Error(
        `Cannot reshape tensor of ${this._numel} elements to ${newNumel}`
      );
    }
    // Return a view (shares memory, different shape)
    return new Tensor(newShape, this.dtype, this.device, this.ptr, false);
  }

  /**
   * Create a view with different shape (alias for reshape).
   */
  view(newShape: number[]): Tensor {
    return this.reshape(newShape);
  }

  /**
   * Transpose the last two dimensions.
   */
  T(): Tensor {
    if (this.ndim < 2) {
      throw new Error("Cannot transpose tensor with less than 2 dimensions");
    }
    const newShape = [...this.shape];
    const tmp = newShape[this.ndim - 1];
    newShape[this.ndim - 1] = newShape[this.ndim - 2];
    newShape[this.ndim - 2] = tmp;

    // Note: This creates a non-contiguous view
    // For actual operations, we might need to materialize
    return new Tensor(newShape, this.dtype, this.device, this.ptr, false);
  }

  /**
   * Free the tensor's GPU memory.
   */
  dispose(): void {
    if (this._ownsMemory) {
      this.pool.free(this.ptr);
    }
  }

  /**
   * String representation for debugging.
   */
  toString(): string {
    return `Tensor(shape=[${this.shape.join(", ")}], dtype=${this.dtype}, device=${this.device})`;
  }
}

// Helper function to convert fp16 to fp32
function fp16ToFp32(h: number): number {
  const sign = (h >> 15) & 0x1;
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x3ff;

  if (exponent === 0) {
    if (fraction === 0) {
      return sign === 0 ? 0 : -0;
    }
    // Subnormal
    return (sign === 0 ? 1 : -1) * Math.pow(2, -14) * (fraction / 1024);
  } else if (exponent === 31) {
    if (fraction === 0) {
      return sign === 0 ? Infinity : -Infinity;
    }
    return NaN;
  }

  return (sign === 0 ? 1 : -1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

// Re-export for convenience
export { DType, dtypeFromString, DTYPE_SIZES } from "./dtype";
