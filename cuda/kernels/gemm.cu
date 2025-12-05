#include "binfer.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>

// Per-device cuBLAS handles (max 16 GPUs)
#define MAX_DEVICES 16
static cublasHandle_t g_cublas_handles[MAX_DEVICES] = {nullptr};
static int g_num_devices = 0;

// Per-device current stream (0 = default stream)
// Used for CUDA graph capture - when capturing, all operations use the capture stream
static cudaStream_t g_current_streams[MAX_DEVICES] = {0};

// Get current stream for current device (internal helper)
static cudaStream_t get_current_stream_internal() {
    int device = 0;
    cudaGetDevice(&device);
    return g_current_streams[device];
}

// Get current stream for current device (for use by other CUDA files like moe_kernels.cu)
cudaStream_t get_current_stream() {
    return get_current_stream_internal();
}

// Set current stream for current device (call before capture, reset to 0 after)
extern "C" BinferError binfer_set_current_stream(void* stream) {
    int device = 0;
    cudaGetDevice(&device);
    g_current_streams[device] = (cudaStream_t)stream;
    // Also update cuBLAS handle to use this stream
    if (g_cublas_handles[device] != nullptr) {
        cublasSetStream(g_cublas_handles[device], (cudaStream_t)stream);
    }
    return BINFER_SUCCESS;
}

extern "C" void* binfer_get_current_stream() {
    return (void*)get_current_stream_internal();
}

static cublasHandle_t get_cublas_handle() {
    // Get current device
    int device = 0;
    cudaGetDevice(&device);

    // Lazily initialize handle for this device
    if (g_cublas_handles[device] == nullptr) {
        cublasCreate(&g_cublas_handles[device]);
        cublasSetMathMode(g_cublas_handles[device], CUBLAS_TF32_TENSOR_OP_MATH);
    }
    // Always ensure the handle uses the current stream (for graph capture)
    cublasSetStream(g_cublas_handles[device], g_current_streams[device]);
    return g_cublas_handles[device];
}

// Initialize cuBLAS handles for all devices upfront
// This should be called once at startup to avoid lazy initialization issues with multi-GPU
extern "C" BinferError binfer_init_cublas(int num_devices) {
    int currentDevice = 0;
    cudaGetDevice(&currentDevice);  // Save current device

    for (int device = 0; device < num_devices && device < MAX_DEVICES; device++) {
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess) {
            cudaSetDevice(currentDevice);
            return BINFER_ERROR_CUDA;
        }

        if (g_cublas_handles[device] == nullptr) {
            cublasStatus_t status = cublasCreate(&g_cublas_handles[device]);
            if (status != CUBLAS_STATUS_SUCCESS) {
                cudaSetDevice(currentDevice);
                return BINFER_ERROR_CUDA;
            }
            cublasSetMathMode(g_cublas_handles[device], CUBLAS_TF32_TENSOR_OP_MATH);
        }
    }

    // Restore original device
    cudaSetDevice(currentDevice);
    g_num_devices = num_devices;
    return BINFER_SUCCESS;
}

// Device management
extern "C" int binfer_get_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

extern "C" BinferError binfer_get_device_properties(int device, char* name, size_t name_len, size_t* total_memory) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return BINFER_ERROR_CUDA;

    snprintf(name, name_len, "%s", prop.name);
    *total_memory = prop.totalGlobalMem;
    return BINFER_SUCCESS;
}

extern "C" BinferError binfer_set_device(int device) {
    cudaError_t err = cudaSetDevice(device);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_mem_info(size_t* free_bytes, size_t* total_bytes) {
    cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Memory management
extern "C" BinferError binfer_malloc(void** ptr, size_t size) {
    cudaError_t err = cudaMalloc(ptr, size);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_OUT_OF_MEMORY;
}

extern "C" BinferError binfer_free(void* ptr) {
    cudaError_t err = cudaFree(ptr);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_memcpy_h2d(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_memcpy_d2h(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_memcpy_d2d(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Pinned host memory management
extern "C" BinferError binfer_malloc_host(void** ptr, size_t size) {
    cudaError_t err = cudaHostAlloc(ptr, size, cudaHostAllocDefault);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_OUT_OF_MEMORY;
}

extern "C" BinferError binfer_free_host(void* ptr) {
    cudaError_t err = cudaFreeHost(ptr);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_memcpy_h2d_async(void* dst, const void* src, size_t size) {
    // Use current stream (default 0, or capture stream during graph capture)
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, get_current_stream_internal());
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_memcpy_host_to_pinned(void* pinned_dst, const void* src, size_t size) {
    // Fast CPU memcpy to pinned memory (for use before async H2D)
    memcpy(pinned_dst, src, size);
    return BINFER_SUCCESS;
}

// GEMM operations using cuBLAS
extern "C" BinferError binfer_gemm_f16(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta
) {
    cublasHandle_t handle = get_cublas_handle();

    // cuBLAS uses column-major, so we compute B^T @ A^T = (A @ B)^T
    // But since our tensors are row-major, we swap A and B and transpose semantics
    __half alpha_h = __float2half(alpha);
    __half beta_h = __float2half(beta);

    cublasStatus_t status = cublasHgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha_h,
        (const __half*)B, N,
        (const __half*)A, K,
        &beta_h,
        (__half*)C, N
    );

    return status == CUBLAS_STATUS_SUCCESS ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// GEMM with B transposed: C = A @ B^T where A is [M,K], B is [N,K], C is [M,N]
extern "C" BinferError binfer_gemm_f16_transb(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta
) {
    cublasHandle_t handle = get_cublas_handle();

    // For row-major with B transposed:
    // C[M,N] = A[M,K] @ B^T[K,N] where B is stored as [N,K]
    // In column-major: C^T = B @ A^T
    __half alpha_h = __float2half(alpha);
    __half beta_h = __float2half(beta);

    cublasStatus_t status = cublasHgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,  // B transposed
        N, M, K,
        &alpha_h,
        (const __half*)B, K,  // B is [N,K], leading dim is K
        (const __half*)A, K,  // A is [M,K], leading dim is K
        &beta_h,
        (__half*)C, N  // C is [M,N], leading dim is N
    );

    return status == CUBLAS_STATUS_SUCCESS ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_gemm_bf16(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta
) {
    cublasHandle_t handle = get_cublas_handle();

    // Use cublasGemmEx for bf16
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, N,
        A, CUDA_R_16BF, K,
        &beta,
        C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    return status == CUBLAS_STATUS_SUCCESS ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// GEMM with B transposed for BF16: C = A @ B^T where A is [M,K], B is [N,K], C is [M,N]
extern "C" BinferError binfer_gemm_bf16_transb(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta
) {
    cublasHandle_t handle = get_cublas_handle();

    // For row-major with B transposed:
    // C[M,N] = A[M,K] @ B^T[K,N] where B is stored as [N,K]
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,  // B transposed
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, K,  // B is [N,K], leading dim is K
        A, CUDA_R_16BF, K,  // A is [M,K], leading dim is K
        &beta,
        C, CUDA_R_16BF, N,  // C is [M,N], leading dim is N
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    return status == CUBLAS_STATUS_SUCCESS ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_gemm_batched_f16(
    const void** A, const void** B, void** C,
    int M, int N, int K,
    float alpha, float beta,
    int batch_size
) {
    cublasHandle_t handle = get_cublas_handle();

    __half alpha_h = __float2half(alpha);
    __half beta_h = __float2half(beta);

    cublasStatus_t status = cublasHgemmBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha_h,
        (const __half**)B, N,
        (const __half**)A, K,
        &beta_h,
        (__half**)C, N,
        batch_size
    );

    return status == CUBLAS_STATUS_SUCCESS ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Synchronization
extern "C" BinferError binfer_synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// CUDA Streams for async overlap
extern "C" BinferError binfer_stream_create(cudaStream_t* stream) {
    cudaError_t err = cudaStreamCreate(stream);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_stream_destroy(cudaStream_t stream) {
    cudaError_t err = cudaStreamDestroy(stream);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_stream_synchronize(cudaStream_t stream) {
    cudaError_t err = cudaStreamSynchronize(stream);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_memcpy_h2d_async_stream(void* dst, const void* src, size_t size, cudaStream_t stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Profiling - CUDA events
extern "C" BinferError binfer_event_create(void** event) {
    cudaEvent_t* e = new cudaEvent_t;
    cudaError_t err = cudaEventCreate(e);
    if (err != cudaSuccess) {
        delete e;
        return BINFER_ERROR_CUDA;
    }
    *event = e;
    return BINFER_SUCCESS;
}

extern "C" BinferError binfer_event_destroy(void* event) {
    cudaEvent_t* e = static_cast<cudaEvent_t*>(event);
    cudaError_t err = cudaEventDestroy(*e);
    delete e;
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_event_record(void* event) {
    cudaEvent_t* e = static_cast<cudaEvent_t*>(event);
    cudaError_t err = cudaEventRecord(*e, 0);  // Default stream
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_event_synchronize(void* event) {
    cudaEvent_t* e = static_cast<cudaEvent_t*>(event);
    cudaError_t err = cudaEventSynchronize(*e);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_event_elapsed_time(float* ms, void* start, void* end) {
    cudaEvent_t* s = static_cast<cudaEvent_t*>(start);
    cudaEvent_t* e = static_cast<cudaEvent_t*>(end);
    cudaError_t err = cudaEventElapsedTime(ms, *s, *e);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Record event on a specific stream (for multi-device graph capture)
extern "C" BinferError binfer_event_record_stream(void* event, void* stream) {
    cudaEvent_t* e = static_cast<cudaEvent_t*>(event);
    cudaError_t err = cudaEventRecord(*e, (cudaStream_t)stream);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Make a stream wait on an event (for multi-device graph capture)
extern "C" BinferError binfer_stream_wait_event(void* stream, void* event) {
    cudaEvent_t* e = static_cast<cudaEvent_t*>(event);
    cudaError_t err = cudaStreamWaitEvent((cudaStream_t)stream, *e, 0);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// =============================================================================
// CUDA Graph capture and replay
// =============================================================================

extern "C" BinferError binfer_stream_begin_capture(void* stream, int mode) {
    cudaStreamCaptureMode captureMode;
    switch (mode) {
        case 0: captureMode = cudaStreamCaptureModeGlobal; break;
        case 1: captureMode = cudaStreamCaptureModeThreadLocal; break;
        case 2: captureMode = cudaStreamCaptureModeRelaxed; break;
        default: return BINFER_ERROR_INVALID_ARGUMENT;
    }
    cudaError_t err = cudaStreamBeginCapture((cudaStream_t)stream, captureMode);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_stream_end_capture(void* stream, void** graph) {
    cudaError_t err = cudaStreamEndCapture((cudaStream_t)stream, (cudaGraph_t*)graph);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_graph_instantiate(void** graph_exec, void* graph) {
    cudaError_t err = cudaGraphInstantiate((cudaGraphExec_t*)graph_exec, (cudaGraph_t)graph, nullptr, nullptr, 0);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_graph_destroy(void* graph) {
    cudaError_t err = cudaGraphDestroy((cudaGraph_t)graph);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_graph_exec_destroy(void* graph_exec) {
    cudaError_t err = cudaGraphExecDestroy((cudaGraphExec_t)graph_exec);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_graph_launch(void* graph_exec, void* stream) {
    cudaError_t err = cudaGraphLaunch((cudaGraphExec_t)graph_exec, (cudaStream_t)stream);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

extern "C" BinferError binfer_graph_exec_update(void* graph_exec, void* graph) {
    cudaGraphExecUpdateResult updateResult;
    cudaError_t err = cudaGraphExecUpdate((cudaGraphExec_t)graph_exec, (cudaGraph_t)graph, nullptr, &updateResult);
    if (err != cudaSuccess) return BINFER_ERROR_CUDA;
    // updateResult tells us if the update was successful or if graph structure changed
    return updateResult == cudaGraphExecUpdateSuccess ? BINFER_SUCCESS : BINFER_ERROR_INVALID_ARGUMENT;
}
