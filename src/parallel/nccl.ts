// NCCL bindings for multi-GPU tensor parallelism
// Uses FFI to call NCCL for all-reduce, all-gather operations

// Suppress NCCL info messages unless user has set NCCL_DEBUG
// Must be set before library is loaded
if (!process.env.NCCL_DEBUG) {
  process.env.NCCL_DEBUG = "WARN";  // Only show warnings and errors
  process.env.NCCL_DEBUG_SUBSYS = "INIT";  // Limit to init subsystem
}

import { dlopen, FFIType, suffix, ptr } from "bun:ffi";
import { getCudaBackend } from "../backend/cuda/bindings";

// NCCL data types
export enum NcclDataType {
  Float16 = 6,  // ncclHalf
  Float32 = 7,  // ncclFloat
  BFloat16 = 9, // ncclBfloat16
}

// NCCL reduce operations
export enum NcclReduceOp {
  Sum = 0,
  Prod = 1,
  Max = 2,
  Min = 3,
  Avg = 4,
}

// For single-process multi-GPU, we need ncclCommInitAll which initializes all GPUs at once
interface NcclLib {
  ncclGetUniqueId: (id: bigint) => number;
  ncclCommInitRank: (comm: bigint, nranks: number, id: bigint, rank: number) => number;
  ncclCommInitAll: (comms: bigint, ndev: number, devlist: bigint) => number;
  ncclCommDestroy: (comm: bigint) => number;
  ncclAllReduce: (sendbuff: bigint, recvbuff: bigint, count: number, datatype: number, op: number, comm: bigint, stream: bigint) => number;
  ncclAllGather: (sendbuff: bigint, recvbuff: bigint, sendcount: number, datatype: number, comm: bigint, stream: bigint) => number;
  ncclBroadcast: (sendbuff: bigint, recvbuff: bigint, count: number, datatype: number, root: number, comm: bigint, stream: bigint) => number;
  ncclReduceScatter: (sendbuff: bigint, recvbuff: bigint, recvcount: number, datatype: number, op: number, comm: bigint, stream: bigint) => number;
  ncclGroupStart: () => number;
  ncclGroupEnd: () => number;
}

let ncclLib: NcclLib | null = null;
let ncclLogger: ((msg: string) => void) | null = null;

function loadNccl(logger?: (msg: string) => void): NcclLib {
  if (ncclLib) return ncclLib;
  ncclLogger = logger ?? null;

  // Search common NCCL locations
  const ncclDirLib = process.env.NCCL_DIR
    ? `${process.env.NCCL_DIR}/libnccl.so.2`
    : null;
  const searchPaths = [
    process.env.NCCL_LIB,
    ncclDirLib,
    "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
    "/usr/lib64/libnccl.so.2",
    "/usr/local/cuda/lib64/libnccl.so.2",
  ].filter(Boolean) as string[];

  let lastError: Error | null = null;

  for (const libPath of searchPaths) {
    try {
      const lib = dlopen(libPath, {
        ncclGetUniqueId: {
          args: [FFIType.ptr],
          returns: FFIType.i32,
        },
        ncclCommInitRank: {
          args: [FFIType.ptr, FFIType.i32, FFIType.ptr, FFIType.i32],
          returns: FFIType.i32,
        },
        ncclCommInitAll: {
          args: [FFIType.ptr, FFIType.i32, FFIType.ptr],
          returns: FFIType.i32,
        },
        ncclCommDestroy: {
          args: [FFIType.u64],  // ncclComm_t is passed by value (it's a pointer value)
          returns: FFIType.i32,
        },
        ncclAllReduce: {
          args: [FFIType.u64, FFIType.u64, FFIType.u64, FFIType.i32, FFIType.i32, FFIType.u64, FFIType.u64],
          returns: FFIType.i32,
        },
        ncclAllGather: {
          args: [FFIType.u64, FFIType.u64, FFIType.u64, FFIType.i32, FFIType.u64, FFIType.u64],
          returns: FFIType.i32,
        },
        ncclBroadcast: {
          args: [FFIType.u64, FFIType.u64, FFIType.u64, FFIType.i32, FFIType.i32, FFIType.u64, FFIType.u64],
          returns: FFIType.i32,
        },
        ncclReduceScatter: {
          args: [FFIType.u64, FFIType.u64, FFIType.u64, FFIType.i32, FFIType.i32, FFIType.u64, FFIType.u64],
          returns: FFIType.i32,
        },
        ncclGroupStart: {
          args: [],
          returns: FFIType.i32,
        },
        ncclGroupEnd: {
          args: [],
          returns: FFIType.i32,
        },
      });

      ncclLib = lib.symbols as unknown as NcclLib;
      ncclLogger?.(`  Loaded NCCL from ${libPath}`);
      return ncclLib;
    } catch (e) {
      lastError = e as Error;
      continue;
    }
  }

  throw new Error(`Failed to load NCCL library from any of: ${searchPaths.join(", ")}\nLast error: ${lastError}`);
}

/**
 * Tensor Parallel Group - manages NCCL communicators for all GPUs in a single process.
 * Uses ncclCommInitAll for single-process multi-GPU setup.
 */
export class TensorParallelGroup {
  private comms: bigint[] = [];
  private worldSize: number;
  private nccl: NcclLib;
  private cuda: ReturnType<typeof getCudaBackend>;
  private logger: ((msg: string) => void) | null;

  constructor(worldSize: number, logger?: (msg: string) => void) {
    this.worldSize = worldSize;
    this.logger = logger ?? null;
    this.nccl = loadNccl(logger);
    this.cuda = getCudaBackend();
  }

  /**
   * Initialize all communicators at once using ncclCommInitAll.
   */
  async init(): Promise<void> {
    // Allocate array of communicator pointers (8 bytes per comm)
    const commsBuffer = new ArrayBuffer(8 * this.worldSize);
    const commsPtr = ptr(commsBuffer);

    // Allocate device list
    const devListBuffer = new ArrayBuffer(4 * this.worldSize);
    const devListView = new Int32Array(devListBuffer);
    for (let i = 0; i < this.worldSize; i++) {
      devListView[i] = i;
    }
    const devListPtr = ptr(devListBuffer);

    // Initialize all communicators at once
    const result = this.nccl.ncclCommInitAll(commsPtr, this.worldSize, devListPtr);
    if (result !== 0) {
      throw new Error(`ncclCommInitAll failed with error ${result}`);
    }

    // Read all communicator pointers
    const commsView = new BigUint64Array(commsBuffer);
    for (let i = 0; i < this.worldSize; i++) {
      this.comms.push(commsView[i]);
    }

    this.logger?.(`  Initialized ${this.worldSize} NCCL communicators`);
  }

  /**
   * Get communicator for a specific rank.
   */
  getComm(rank: number): bigint {
    return this.comms[rank];
  }

  /**
   * All-reduce across all GPUs.
   * Launches the operation on all GPUs using ncclGroupStart/End.
   * Uses the current stream (important for CUDA graph capture).
   */
  allReduceSum(
    buffers: { send: bigint; recv: bigint; device: number }[],
    count: number,
    dtype: NcclDataType = NcclDataType.Float16
  ): void {
    this.nccl.ncclGroupStart();

    for (const { send, recv, device } of buffers) {
      this.cuda.setDevice(device);
      // Use current stream for CUDA graph compatibility
      const stream = this.cuda.getCurrentStream();
      const result = this.nccl.ncclAllReduce(
        send, recv, count,
        dtype, NcclReduceOp.Sum,
        this.comms[device], stream
      );
      if (result !== 0) {
        this.nccl.ncclGroupEnd();
        throw new Error(`ncclAllReduce failed on device ${device} with error ${result}`);
      }
    }

    this.nccl.ncclGroupEnd();
  }

  /**
   * Synchronize all devices.
   */
  synchronize(): void {
    for (let device = 0; device < this.worldSize; device++) {
      this.cuda.setDevice(device);
      this.cuda.synchronize();
    }
  }

  /**
   * Dispose all communicators.
   */
  dispose(): void {
    for (const comm of this.comms) {
      this.nccl.ncclCommDestroy(comm);
    }
    this.comms = [];
  }
}

/**
 * NCCL Communicator for tensor parallelism (single rank).
 * For use in multi-process setup or when you need a single communicator.
 */
export class NcclCommunicator {
  private commPtr: bigint = 0n;
  private worldSize: number;
  private rank: number;
  private nccl: NcclLib;
  private streamPtr: bigint = 0n;

  constructor(worldSize: number, rank: number) {
    this.worldSize = worldSize;
    this.rank = rank;
    this.nccl = loadNccl();
  }

  /**
   * Initialize the communicator with a unique ID.
   * Rank 0 generates the ID, others receive it.
   */
  async init(uniqueIdBuffer?: ArrayBuffer): Promise<ArrayBuffer> {
    // Allocate memory for unique ID (128 bytes for NCCL internal structure)
    const idBuffer = uniqueIdBuffer || new ArrayBuffer(128);
    const idPtr = ptr(idBuffer);

    if (!uniqueIdBuffer) {
      // Rank 0 generates the unique ID
      const result = this.nccl.ncclGetUniqueId(idPtr);
      if (result !== 0) {
        throw new Error(`ncclGetUniqueId failed with error ${result}`);
      }
    }

    // Initialize communicator
    const commBuffer = new ArrayBuffer(8);
    const commPtr = ptr(commBuffer);

    const result = this.nccl.ncclCommInitRank(commPtr, this.worldSize, idPtr, this.rank);
    if (result !== 0) {
      throw new Error(`ncclCommInitRank failed with error ${result}`);
    }

    // Read the communicator pointer
    const commView = new BigUint64Array(commBuffer);
    this.commPtr = commView[0];

    // Get CUDA stream from backend (use default stream = 0)
    this.streamPtr = 0n;

    return idBuffer;
  }

  /**
   * All-reduce: Sum values across all ranks.
   * In-place operation: result is stored in recvBuff.
   */
  allReduceSum(sendBuff: bigint, recvBuff: bigint, count: number, dtype: NcclDataType = NcclDataType.Float16): void {
    const result = this.nccl.ncclAllReduce(
      sendBuff, recvBuff, count,
      dtype, NcclReduceOp.Sum,
      this.commPtr, this.streamPtr
    );
    if (result !== 0) {
      throw new Error(`ncclAllReduce failed with error ${result}`);
    }
  }

  /**
   * All-gather: Gather values from all ranks.
   * Each rank contributes sendCount elements, recvBuff has worldSize * sendCount elements.
   */
  allGather(sendBuff: bigint, recvBuff: bigint, sendCount: number, dtype: NcclDataType = NcclDataType.Float16): void {
    const result = this.nccl.ncclAllGather(
      sendBuff, recvBuff, sendCount,
      dtype, this.commPtr, this.streamPtr
    );
    if (result !== 0) {
      throw new Error(`ncclAllGather failed with error ${result}`);
    }
  }

  /**
   * Broadcast: Send data from root to all ranks.
   */
  broadcast(buff: bigint, count: number, root: number, dtype: NcclDataType = NcclDataType.Float16): void {
    const result = this.nccl.ncclBroadcast(
      buff, buff, count,
      dtype, root, this.commPtr, this.streamPtr
    );
    if (result !== 0) {
      throw new Error(`ncclBroadcast failed with error ${result}`);
    }
  }

  /**
   * Reduce-scatter: Reduce and scatter result to all ranks.
   * Each rank receives recvCount elements of the reduced result.
   */
  reduceScatter(sendBuff: bigint, recvBuff: bigint, recvCount: number, dtype: NcclDataType = NcclDataType.Float16): void {
    const result = this.nccl.ncclReduceScatter(
      sendBuff, recvBuff, recvCount,
      dtype, NcclReduceOp.Sum, this.commPtr, this.streamPtr
    );
    if (result !== 0) {
      throw new Error(`ncclReduceScatter failed with error ${result}`);
    }
  }

  get size(): number {
    return this.worldSize;
  }

  get myRank(): number {
    return this.rank;
  }

  dispose(): void {
    if (this.commPtr !== 0n) {
      this.nccl.ncclCommDestroy(this.commPtr);
      this.commPtr = 0n;
    }
  }
}
