// Test: Tensor operations (no CUDA required - tests CPU-side logic)
// Run with: bun test tests/unit/tensor.test.ts

import { expect, test, describe } from "bun:test";
import { DType, dtypeFromString, dtypeSize } from "../../src/tensor/dtype";

describe("DType", () => {
  test("dtypeFromString parses common formats", () => {
    expect(dtypeFromString("float32")).toBe(DType.Float32);
    expect(dtypeFromString("f32")).toBe(DType.Float32);
    expect(dtypeFromString("float16")).toBe(DType.Float16);
    expect(dtypeFromString("f16")).toBe(DType.Float16);
    expect(dtypeFromString("half")).toBe(DType.Float16);
    expect(dtypeFromString("bfloat16")).toBe(DType.BFloat16);
    expect(dtypeFromString("bf16")).toBe(DType.BFloat16);
    expect(dtypeFromString("int32")).toBe(DType.Int32);
    expect(dtypeFromString("int8")).toBe(DType.Int8);
  });

  test("dtypeSize returns correct byte sizes", () => {
    expect(dtypeSize(DType.Float32)).toBe(4);
    expect(dtypeSize(DType.Float16)).toBe(2);
    expect(dtypeSize(DType.BFloat16)).toBe(2);
    expect(dtypeSize(DType.Int32)).toBe(4);
    expect(dtypeSize(DType.Int64)).toBe(8);
    expect(dtypeSize(DType.Int8)).toBe(1);
    expect(dtypeSize(DType.UInt8)).toBe(1);
  });
});

describe("Tensor Shape Utilities", () => {
  test("compute numel from shape", () => {
    const computeNumel = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

    expect(computeNumel([10])).toBe(10);
    expect(computeNumel([2, 3])).toBe(6);
    expect(computeNumel([2, 3, 4])).toBe(24);
    expect(computeNumel([1, 128, 4096])).toBe(524288);
  });

  test("compute strides from shape", () => {
    const computeStrides = (shape: number[]) => {
      const strides = new Array(shape.length);
      let stride = 1;
      for (let i = shape.length - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
      }
      return strides;
    };

    expect(computeStrides([10])).toEqual([1]);
    expect(computeStrides([2, 3])).toEqual([3, 1]);
    expect(computeStrides([2, 3, 4])).toEqual([12, 4, 1]);
    expect(computeStrides([1, 128, 4096])).toEqual([524288, 4096, 1]);
  });

  test("compute nbytes from shape and dtype", () => {
    const computeNbytes = (shape: number[], dtype: DType) => {
      const numel = shape.reduce((a, b) => a * b, 1);
      return numel * dtypeSize(dtype);
    };

    // [1, 128, 4096] in fp16 = 524288 * 2 = 1MB
    expect(computeNbytes([1, 128, 4096], DType.Float16)).toBe(1048576);

    // [32, 128, 4096] in fp16 = 16M elements * 2 bytes = 32MB
    expect(computeNbytes([32, 128, 4096], DType.Float16)).toBe(33554432);
  });
});

describe("FP16 Conversion", () => {
  // Test the fp16 to fp32 conversion logic
  const fp16ToFp32 = (h: number): number => {
    const sign = (h >> 15) & 0x1;
    const exponent = (h >> 10) & 0x1f;
    const fraction = h & 0x3ff;

    if (exponent === 0) {
      if (fraction === 0) {
        return sign === 0 ? 0 : -0;
      }
      return (sign === 0 ? 1 : -1) * Math.pow(2, -14) * (fraction / 1024);
    } else if (exponent === 31) {
      if (fraction === 0) {
        return sign === 0 ? Infinity : -Infinity;
      }
      return NaN;
    }

    return (sign === 0 ? 1 : -1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
  };

  test("converts fp16 zero", () => {
    expect(fp16ToFp32(0x0000)).toBe(0);
    expect(Object.is(fp16ToFp32(0x8000), -0)).toBe(true);
  });

  test("converts fp16 one", () => {
    // 1.0 in fp16: sign=0, exp=15, frac=0 -> 0x3C00
    expect(fp16ToFp32(0x3c00)).toBe(1.0);
  });

  test("converts fp16 negative one", () => {
    // -1.0 in fp16: sign=1, exp=15, frac=0 -> 0xBC00
    expect(fp16ToFp32(0xbc00)).toBe(-1.0);
  });

  test("converts fp16 infinity", () => {
    expect(fp16ToFp32(0x7c00)).toBe(Infinity);
    expect(fp16ToFp32(0xfc00)).toBe(-Infinity);
  });

  test("converts fp16 NaN", () => {
    expect(Number.isNaN(fp16ToFp32(0x7c01))).toBe(true);
  });
});
