// MoE with CUTLASS Grouped GEMM - optimal for H100 tensor cores
// Uses CUTLASS grouped GEMM to run all expert GEMMs in parallel with BF16 support

#include "../include/binfer.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/device_memory.h"

// Max experts we support
#define MAX_EXPERTS 128

// E8M0 lookup table
__device__ __constant__ float cutlass_e8m0_table[256];

// FP4 E2M1 lookup table
__device__ __constant__ float cutlass_fp4_table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// Initialize E8M0 table
extern "C" BinferError binfer_init_moe_cutlass_tables() {
    float h_table[256];
    for (int i = 0; i < 256; i++) {
        if (i == 255) {
            h_table[i] = NAN;
        } else {
            union { float f; uint32_t u; } converter;
            converter.u = (uint32_t)i << 23;
            h_table[i] = converter.f;
        }
    }
    cudaError_t err = cudaMemcpyToSymbol(cutlass_e8m0_table, h_table, sizeof(h_table));
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Kernel to build gather indices
__global__ void cutlass_build_gather_kernel(
    const int32_t* __restrict__ expert_indices,
    int* __restrict__ gather_indices,
    int* __restrict__ expert_counts,
    int numTokens,
    int topK,
    int numExperts,
    int maxPerExpert
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTokens * topK) return;

    int expert = expert_indices[tid];
    if (expert >= 0 && expert < numExperts) {
        int slot = atomicAdd(&expert_counts[expert], 1);
        if (slot < maxPerExpert) {
            gather_indices[expert * maxPerExpert + slot] = tid;
        }
    }
}

// Gather hidden states for one expert
__global__ void cutlass_gather_kernel(
    const cutlass::bfloat16_t* __restrict__ hidden,
    const int* __restrict__ gather_indices,
    cutlass::bfloat16_t* __restrict__ gathered,
    int count,
    int hiddenSize,
    int topK
) {
    const int token_out_idx = blockIdx.x;
    const int feat_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (token_out_idx >= count || feat_idx >= hiddenSize) return;

    int source_idx = gather_indices[token_out_idx];
    int original_token = source_idx / topK;

    gathered[token_out_idx * hiddenSize + feat_idx] = hidden[original_token * hiddenSize + feat_idx];
}

// Dequantize MXFP4 weights for one expert
__global__ void cutlass_dequant_kernel(
    const uint8_t* __restrict__ blocks,
    const uint8_t* __restrict__ scales,
    cutlass::bfloat16_t* __restrict__ output,
    int expertIdx,
    int numExperts,
    int outFeatures,
    int numBlocks,
    int inFeatures
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = outFeatures * inFeatures;

    if (tid >= total_elements) return;

    int out_idx = tid / inFeatures;
    int in_idx = tid % inFeatures;

    int block_idx = in_idx / 32;
    int pos_in_block = in_idx % 32;
    int byte_idx = pos_in_block / 2;
    int is_high = pos_in_block % 2;

    int block_offset = (expertIdx * outFeatures + out_idx) * numBlocks + block_idx;

    float scale = cutlass_e8m0_table[scales[block_offset]];
    uint8_t packed = blocks[block_offset * 16 + byte_idx];
    uint8_t nibble = is_high ? ((packed >> 4) & 0xF) : (packed & 0xF);
    float val = cutlass_fp4_table[nibble] * scale;

    output[tid] = cutlass::bfloat16_t(val);
}

// Add bias to GEMM output
__global__ void cutlass_add_bias_kernel(
    cutlass::bfloat16_t* __restrict__ output,
    const cutlass::bfloat16_t* __restrict__ bias,
    int expertIdx,
    int count,
    int outFeatures
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = count * outFeatures;
    if (idx >= total) return;

    int out_idx = idx % outFeatures;
    float val = float(output[idx]);
    float b = float(bias[expertIdx * outFeatures + out_idx]);
    output[idx] = cutlass::bfloat16_t(val + b);
}

// GPT-OSS activation
__global__ void cutlass_activation_kernel(
    const cutlass::bfloat16_t* __restrict__ gate_up,
    cutlass::bfloat16_t* __restrict__ activated,
    int N,
    int intermediateSize,
    float alpha,
    float limit
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * intermediateSize;
    if (idx >= total) return;

    int n = idx / intermediateSize;
    int i = idx % intermediateSize;

    int gate_idx = n * intermediateSize * 2 + i * 2;
    int up_idx = gate_idx + 1;

    float gate = float(gate_up[gate_idx]);
    float up = float(gate_up[up_idx]);

    gate = fminf(gate, limit);
    up = fmaxf(fminf(up, limit), -limit);
    float glu = gate / (1.0f + expf(-gate * alpha));
    float result = (up + 1.0f) * glu;

    activated[idx] = cutlass::bfloat16_t(result);
}

// Scatter accumulate to float32 output
__global__ void cutlass_scatter_kernel(
    const cutlass::bfloat16_t* __restrict__ expert_output,
    const int* __restrict__ gather_indices,
    const cutlass::bfloat16_t* __restrict__ routing_weights,
    float* __restrict__ output_f32,
    int count,
    int hiddenSize,
    int topK
) {
    const int token_out_idx = blockIdx.x;
    const int feat_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (token_out_idx >= count || feat_idx >= hiddenSize) return;

    int source_idx = gather_indices[token_out_idx];
    int original_token = source_idx / topK;

    float weight = float(routing_weights[source_idx]);
    float val = float(expert_output[token_out_idx * hiddenSize + feat_idx]);
    float weighted = val * weight;

    atomicAdd(&output_f32[original_token * hiddenSize + feat_idx], weighted);
}

// Convert float32 to BF16
__global__ void cutlass_f32_to_bf16_kernel(
    const float* __restrict__ input,
    cutlass::bfloat16_t* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = cutlass::bfloat16_t(input[idx]);
    }
}

// CUTLASS Grouped GEMM type definitions for H100 (sm_90)
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;  // Transposed weights
using LayoutC = cutlass::layout::RowMajor;

// For sm_90 (H100), use appropriate tile sizes
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementAccumulator
>;

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, 8,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, 8,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,  // Use Sm80 for broader compatibility
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4,  // Stages
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly
>::GemmKernel;

using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

// Main MoE forward with CUTLASS
extern "C" BinferError binfer_moe_cutlass_forward(
    const void* hidden,
    const void* gate_up_blocks,
    const void* gate_up_scales,
    const void* gate_up_bias,
    const void* down_blocks,
    const void* down_scales,
    const void* down_bias,
    const void* expert_indices,
    const void* expert_weights,
    void* output,
    void* scratch,
    size_t scratch_size,
    int numTokens,
    int hiddenSize,
    int intermediateSize,
    int numExperts,
    int topK
) {
    if (numExperts > MAX_EXPERTS) {
        fprintf(stderr, "MoE CUTLASS: too many experts %d > %d\n", numExperts, MAX_EXPERTS);
        return BINFER_ERROR_INVALID_ARGUMENT;
    }

    const int numBlocksIn = hiddenSize / 32;
    const int numBlocksInter = intermediateSize / 32;
    const int gateUpSize = intermediateSize * 2;
    const int maxPerExpert = numTokens * topK;

    // Partition scratch buffer
    char* scratch_ptr = (char*)scratch;

    // Per-expert buffers
    std::vector<cutlass::bfloat16_t*> gathered_hidden(numExperts);
    std::vector<cutlass::bfloat16_t*> gate_up_out(numExperts);
    std::vector<cutlass::bfloat16_t*> activated_buf(numExperts);
    std::vector<cutlass::bfloat16_t*> down_out(numExperts);
    std::vector<cutlass::bfloat16_t*> dequant_gate_up(numExperts);
    std::vector<cutlass::bfloat16_t*> dequant_down(numExperts);

    size_t per_expert_gathered = (size_t)maxPerExpert * hiddenSize * sizeof(cutlass::bfloat16_t);
    size_t per_expert_gate_up_out = (size_t)maxPerExpert * gateUpSize * sizeof(cutlass::bfloat16_t);
    size_t per_expert_activated = (size_t)maxPerExpert * intermediateSize * sizeof(cutlass::bfloat16_t);
    size_t per_expert_down_out = (size_t)maxPerExpert * hiddenSize * sizeof(cutlass::bfloat16_t);
    size_t per_expert_dequant_gu = (size_t)gateUpSize * hiddenSize * sizeof(cutlass::bfloat16_t);
    size_t per_expert_dequant_down = (size_t)hiddenSize * intermediateSize * sizeof(cutlass::bfloat16_t);

    for (int e = 0; e < numExperts; e++) {
        gathered_hidden[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_gathered;
        gate_up_out[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_gate_up_out;
        activated_buf[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_activated;
        down_out[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_down_out;
        dequant_gate_up[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_dequant_gu;
        dequant_down[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_dequant_down;
    }

    int* gather_indices_buf = (int*)scratch_ptr;
    scratch_ptr += numExperts * maxPerExpert * sizeof(int);
    int* expert_counts = (int*)scratch_ptr;
    scratch_ptr += numExperts * sizeof(int);
    float* output_f32 = (float*)scratch_ptr;
    scratch_ptr += numTokens * hiddenSize * sizeof(float);

    // Problem descriptors for grouped GEMM (on device)
    cutlass::gemm::GemmCoord* problem_sizes_gu = (cutlass::gemm::GemmCoord*)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(cutlass::gemm::GemmCoord);
    cutlass::bfloat16_t** ptr_A_gu = (cutlass::bfloat16_t**)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(void*);
    cutlass::bfloat16_t** ptr_B_gu = (cutlass::bfloat16_t**)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(void*);
    cutlass::bfloat16_t** ptr_C_gu = (cutlass::bfloat16_t**)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(void*);
    int64_t* lda_gu = (int64_t*)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(int64_t);
    int64_t* ldb_gu = (int64_t*)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(int64_t);
    int64_t* ldc_gu = (int64_t*)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(int64_t);

    // Zero output and counts
    cudaMemsetAsync(output_f32, 0, numTokens * hiddenSize * sizeof(float));
    cudaMemsetAsync(expert_counts, 0, numExperts * sizeof(int));

    // Build gather indices
    {
        int total = numTokens * topK;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        cutlass_build_gather_kernel<<<blocks, threads>>>(
            (const int32_t*)expert_indices,
            gather_indices_buf,
            expert_counts,
            numTokens,
            topK,
            numExperts,
            maxPerExpert
        );
    }

    // Copy counts to host
    int h_counts[MAX_EXPERTS];
    cudaMemcpy(h_counts, expert_counts, numExperts * sizeof(int), cudaMemcpyDeviceToHost);

    // Find active experts
    int active_experts = 0;
    int active_indices[MAX_EXPERTS];
    for (int e = 0; e < numExperts; e++) {
        if (h_counts[e] > 0) {
            active_indices[active_experts++] = e;
        }
    }

    if (active_experts == 0) {
        cudaMemset(output, 0, numTokens * hiddenSize * sizeof(cutlass::bfloat16_t));
        return BINFER_SUCCESS;
    }

    // Prepare host arrays for grouped GEMM
    std::vector<cutlass::gemm::GemmCoord> h_problem_sizes_gu(active_experts);
    std::vector<cutlass::bfloat16_t*> h_ptr_A_gu(active_experts);
    std::vector<cutlass::bfloat16_t*> h_ptr_B_gu(active_experts);
    std::vector<cutlass::bfloat16_t*> h_ptr_C_gu(active_experts);
    std::vector<int64_t> h_lda_gu(active_experts);
    std::vector<int64_t> h_ldb_gu(active_experts);
    std::vector<int64_t> h_ldc_gu(active_experts);

    // Launch gather and dequant for each active expert
    for (int i = 0; i < active_experts; i++) {
        int e = active_indices[i];
        int count = h_counts[e];
        int* expert_gather = gather_indices_buf + e * maxPerExpert;

        // Gather hidden states
        {
            dim3 grid(count, (hiddenSize + 255) / 256);
            dim3 block(256);
            cutlass_gather_kernel<<<grid, block>>>(
                (const cutlass::bfloat16_t*)hidden,
                expert_gather,
                gathered_hidden[e],
                count,
                hiddenSize,
                topK
            );
        }

        // Dequant gate_up weights
        {
            int total = gateUpSize * hiddenSize;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            cutlass_dequant_kernel<<<blocks, threads>>>(
                (const uint8_t*)gate_up_blocks,
                (const uint8_t*)gate_up_scales,
                dequant_gate_up[e],
                e, numExperts, gateUpSize, numBlocksIn, hiddenSize
            );
        }

        // Setup GEMM problem: C = A @ B^T
        // A = gathered_hidden[e]: [count, hiddenSize] row-major
        // B = dequant_gate_up[e]: [gateUpSize, hiddenSize] row-major -> treat as [hiddenSize, gateUpSize] col-major
        // C = gate_up_out[e]: [count, gateUpSize] row-major
        h_problem_sizes_gu[i] = cutlass::gemm::GemmCoord(count, gateUpSize, hiddenSize);
        h_ptr_A_gu[i] = gathered_hidden[e];
        h_ptr_B_gu[i] = dequant_gate_up[e];
        h_ptr_C_gu[i] = gate_up_out[e];
        h_lda_gu[i] = hiddenSize;  // A is [count, hiddenSize] row-major, lda = hiddenSize
        h_ldb_gu[i] = hiddenSize;  // B is [gateUpSize, hiddenSize] -> col-major view has ldb = hiddenSize
        h_ldc_gu[i] = gateUpSize;  // C is [count, gateUpSize] row-major, ldc = gateUpSize
    }

    cudaDeviceSynchronize();

    // Copy problem descriptors to device
    cudaMemcpy(problem_sizes_gu, h_problem_sizes_gu.data(),
               active_experts * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_A_gu, h_ptr_A_gu.data(), active_experts * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_B_gu, h_ptr_B_gu.data(), active_experts * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_C_gu, h_ptr_C_gu.data(), active_experts * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(lda_gu, h_lda_gu.data(), active_experts * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ldb_gu, h_ldb_gu.data(), active_experts * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ldc_gu, h_ldc_gu.data(), active_experts * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Run grouped GEMM for gate_up projection
    {
        typename GemmGrouped::Arguments args(
            problem_sizes_gu,
            active_experts,
            512,  // threadblock_count (heuristic)
            {float(1.0f), float(0.0f)},  // alpha, beta
            reinterpret_cast<cutlass::bfloat16_t**>(ptr_A_gu),
            reinterpret_cast<cutlass::bfloat16_t**>(ptr_B_gu),
            ptr_C_gu,
            ptr_C_gu,
            lda_gu,
            ldb_gu,
            ldc_gu,
            ldc_gu
        );

        GemmGrouped gemm_op;
        cutlass::Status status = gemm_op.initialize(args);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "MoE CUTLASS: gate_up GEMM init failed\n");
            return BINFER_ERROR_CUDA;
        }

        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "MoE CUTLASS: gate_up GEMM run failed\n");
            return BINFER_ERROR_CUDA;
        }
    }

    // Add gate_up bias and apply activation for each active expert
    for (int i = 0; i < active_experts; i++) {
        int e = active_indices[i];
        int count = h_counts[e];

        // Add bias if present
        if (gate_up_bias != nullptr) {
            int total = count * gateUpSize;
            int threads = 256;
            int blocks_bias = (total + threads - 1) / threads;
            cutlass_add_bias_kernel<<<blocks_bias, threads>>>(
                gate_up_out[e],
                (const cutlass::bfloat16_t*)gate_up_bias,
                e, count, gateUpSize
            );
        }

        // Activation
        int total = count * intermediateSize;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        cutlass_activation_kernel<<<blocks, threads>>>(
            gate_up_out[e],
            activated_buf[e],
            count,
            intermediateSize,
            1.702f,
            7.0f
        );
    }

    // Prepare down projection grouped GEMM
    std::vector<cutlass::gemm::GemmCoord> h_problem_sizes_down(active_experts);
    std::vector<cutlass::bfloat16_t*> h_ptr_A_down(active_experts);
    std::vector<cutlass::bfloat16_t*> h_ptr_B_down(active_experts);
    std::vector<cutlass::bfloat16_t*> h_ptr_C_down(active_experts);
    std::vector<int64_t> h_lda_down(active_experts);
    std::vector<int64_t> h_ldb_down(active_experts);
    std::vector<int64_t> h_ldc_down(active_experts);

    for (int i = 0; i < active_experts; i++) {
        int e = active_indices[i];
        int count = h_counts[e];

        // Dequant down weights
        {
            int total = hiddenSize * intermediateSize;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            cutlass_dequant_kernel<<<blocks, threads>>>(
                (const uint8_t*)down_blocks,
                (const uint8_t*)down_scales,
                dequant_down[e],
                e, numExperts, hiddenSize, numBlocksInter, intermediateSize
            );
        }

        // C = A @ B^T
        // A = activated_buf[e]: [count, intermediateSize]
        // B = dequant_down[e]: [hiddenSize, intermediateSize]
        // C = down_out[e]: [count, hiddenSize]
        h_problem_sizes_down[i] = cutlass::gemm::GemmCoord(count, hiddenSize, intermediateSize);
        h_ptr_A_down[i] = activated_buf[e];
        h_ptr_B_down[i] = dequant_down[e];
        h_ptr_C_down[i] = down_out[e];
        h_lda_down[i] = intermediateSize;
        h_ldb_down[i] = intermediateSize;
        h_ldc_down[i] = hiddenSize;
    }

    cudaDeviceSynchronize();

    // Copy down projection descriptors
    cudaMemcpy(problem_sizes_gu, h_problem_sizes_down.data(),
               active_experts * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_A_gu, h_ptr_A_down.data(), active_experts * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_B_gu, h_ptr_B_down.data(), active_experts * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_C_gu, h_ptr_C_down.data(), active_experts * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(lda_gu, h_lda_down.data(), active_experts * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ldb_gu, h_ldb_down.data(), active_experts * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ldc_gu, h_ldc_down.data(), active_experts * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Run grouped GEMM for down projection
    {
        typename GemmGrouped::Arguments args(
            problem_sizes_gu,
            active_experts,
            512,
            {float(1.0f), float(0.0f)},
            reinterpret_cast<cutlass::bfloat16_t**>(ptr_A_gu),
            reinterpret_cast<cutlass::bfloat16_t**>(ptr_B_gu),
            ptr_C_gu,
            ptr_C_gu,
            lda_gu,
            ldb_gu,
            ldc_gu,
            ldc_gu
        );

        GemmGrouped gemm_op;
        cutlass::Status status = gemm_op.initialize(args);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "MoE CUTLASS: down GEMM init failed\n");
            return BINFER_ERROR_CUDA;
        }

        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "MoE CUTLASS: down GEMM run failed\n");
            return BINFER_ERROR_CUDA;
        }
    }

    // Add down bias and scatter accumulate
    for (int i = 0; i < active_experts; i++) {
        int e = active_indices[i];
        int count = h_counts[e];
        int* expert_gather = gather_indices_buf + e * maxPerExpert;

        // Add bias if present
        if (down_bias != nullptr) {
            int total = count * hiddenSize;
            int threads = 256;
            int blocks_bias = (total + threads - 1) / threads;
            cutlass_add_bias_kernel<<<blocks_bias, threads>>>(
                down_out[e],
                (const cutlass::bfloat16_t*)down_bias,
                e, count, hiddenSize
            );
        }

        dim3 grid(count, (hiddenSize + 255) / 256);
        dim3 block(256);
        cutlass_scatter_kernel<<<grid, block>>>(
            down_out[e],
            expert_gather,
            (const cutlass::bfloat16_t*)expert_weights,
            output_f32,
            count,
            hiddenSize,
            topK
        );
    }

    // Convert to BF16
    {
        int total = numTokens * hiddenSize;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        cutlass_f32_to_bf16_kernel<<<blocks, threads>>>(
            output_f32,
            (cutlass::bfloat16_t*)output,
            total
        );
    }

    return cudaGetLastError() == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// External function from gemm.cu to get current stream for CUDA graph capture
extern cudaStream_t get_current_stream();

// ============================================================================
// Async CUTLASS MoE (no host-device sync)
// ============================================================================

// Scatter kernel that reads count from device memory (for async version)
__global__ void cutlass_scatter_async_kernel(
    const cutlass::bfloat16_t* __restrict__ expert_output,
    const int* __restrict__ gather_indices,
    const int* __restrict__ expert_counts,
    const cutlass::bfloat16_t* __restrict__ routing_weights,
    float* __restrict__ output_f32,
    int expert_idx,
    int maxPerExpert,
    int hiddenSize,
    int topK
) {
    const int token_out_idx = blockIdx.x;
    const int feat_idx = threadIdx.x + blockIdx.y * blockDim.x;

    // Read count from device memory
    int count = expert_counts[expert_idx];
    if (token_out_idx >= count || feat_idx >= hiddenSize) return;

    int source_idx = gather_indices[token_out_idx];
    int original_token = source_idx / topK;

    float weight = float(routing_weights[source_idx]);
    float val = float(expert_output[token_out_idx * hiddenSize + feat_idx]);
    float weighted = val * weight;

    atomicAdd(&output_f32[original_token * hiddenSize + feat_idx], weighted);
}

// Device kernel to setup GEMM problem descriptors
__global__ void cutlass_setup_gemm_problems_kernel(
    const int* __restrict__ expert_counts,
    cutlass::gemm::GemmCoord* problem_sizes,
    cutlass::bfloat16_t** ptr_A,
    cutlass::bfloat16_t** ptr_B,
    cutlass::bfloat16_t** ptr_C,
    int64_t* lda,
    int64_t* ldb,
    int64_t* ldc,
    cutlass::bfloat16_t** gathered_ptrs,
    cutlass::bfloat16_t** weight_ptrs,
    cutlass::bfloat16_t** output_ptrs,
    int N,  // output features (gateUpSize or hiddenSize)
    int K,  // input features (hiddenSize or intermediateSize)
    int numExperts
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= numExperts) return;

    int count = expert_counts[e];

    problem_sizes[e] = cutlass::gemm::GemmCoord(count, N, K);
    ptr_A[e] = gathered_ptrs[e];
    ptr_B[e] = weight_ptrs[e];
    ptr_C[e] = output_ptrs[e];
    lda[e] = K;
    ldb[e] = K;
    ldc[e] = N;
}

// Async MoE forward - no host-device synchronization
extern "C" BinferError binfer_moe_cutlass_forward_async(
    const void* hidden,
    const void* gate_up_blocks,
    const void* gate_up_scales,
    const void* gate_up_bias,
    const void* down_blocks,
    const void* down_scales,
    const void* down_bias,
    const void* expert_indices,
    const void* expert_weights,
    void* output,
    void* scratch,
    size_t scratch_size,
    int numTokens,
    int hiddenSize,
    int intermediateSize,
    int numExperts,
    int topK
) {
    if (numExperts > MAX_EXPERTS) {
        return BINFER_ERROR_INVALID_ARGUMENT;
    }

    cudaStream_t stream = get_current_stream();
    const int numBlocksIn = hiddenSize / 32;
    const int numBlocksInter = intermediateSize / 32;
    const int gateUpSize = intermediateSize * 2;
    const int maxPerExpert = numTokens * topK;

    // Partition scratch buffer - need extra space for pointer arrays
    char* scratch_ptr = (char*)scratch;

    // Device pointer arrays
    cutlass::bfloat16_t** gathered_hidden = (cutlass::bfloat16_t**)scratch_ptr;
    scratch_ptr += numExperts * sizeof(void*);
    cutlass::bfloat16_t** gate_up_out = (cutlass::bfloat16_t**)scratch_ptr;
    scratch_ptr += numExperts * sizeof(void*);
    cutlass::bfloat16_t** activated_buf = (cutlass::bfloat16_t**)scratch_ptr;
    scratch_ptr += numExperts * sizeof(void*);
    cutlass::bfloat16_t** down_out = (cutlass::bfloat16_t**)scratch_ptr;
    scratch_ptr += numExperts * sizeof(void*);
    cutlass::bfloat16_t** dequant_gate_up = (cutlass::bfloat16_t**)scratch_ptr;
    scratch_ptr += numExperts * sizeof(void*);
    cutlass::bfloat16_t** dequant_down = (cutlass::bfloat16_t**)scratch_ptr;
    scratch_ptr += numExperts * sizeof(void*);

    // Host arrays for pointer setup (small, on stack)
    cutlass::bfloat16_t* h_gathered[MAX_EXPERTS];
    cutlass::bfloat16_t* h_gate_up_out[MAX_EXPERTS];
    cutlass::bfloat16_t* h_activated[MAX_EXPERTS];
    cutlass::bfloat16_t* h_down_out[MAX_EXPERTS];
    cutlass::bfloat16_t* h_dequant_gu[MAX_EXPERTS];
    cutlass::bfloat16_t* h_dequant_down[MAX_EXPERTS];

    size_t per_expert_gathered = (size_t)maxPerExpert * hiddenSize * sizeof(cutlass::bfloat16_t);
    size_t per_expert_gate_up_out = (size_t)maxPerExpert * gateUpSize * sizeof(cutlass::bfloat16_t);
    size_t per_expert_activated = (size_t)maxPerExpert * intermediateSize * sizeof(cutlass::bfloat16_t);
    size_t per_expert_down_out = (size_t)maxPerExpert * hiddenSize * sizeof(cutlass::bfloat16_t);
    size_t per_expert_dequant_gu = (size_t)gateUpSize * hiddenSize * sizeof(cutlass::bfloat16_t);
    size_t per_expert_dequant_down = (size_t)hiddenSize * intermediateSize * sizeof(cutlass::bfloat16_t);

    for (int e = 0; e < numExperts; e++) {
        h_gathered[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_gathered;
        h_gate_up_out[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_gate_up_out;
        h_activated[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_activated;
        h_down_out[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_down_out;
        h_dequant_gu[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_dequant_gu;
        h_dequant_down[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_dequant_down;
    }

    int* gather_indices_buf = (int*)scratch_ptr;
    scratch_ptr += numExperts * maxPerExpert * sizeof(int);
    int* expert_counts = (int*)scratch_ptr;
    scratch_ptr += numExperts * sizeof(int);
    float* output_f32 = (float*)scratch_ptr;
    scratch_ptr += numTokens * hiddenSize * sizeof(float);

    // GEMM descriptors (on device)
    cutlass::gemm::GemmCoord* problem_sizes = (cutlass::gemm::GemmCoord*)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(cutlass::gemm::GemmCoord);
    cutlass::bfloat16_t** ptr_A = (cutlass::bfloat16_t**)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(void*);
    cutlass::bfloat16_t** ptr_B = (cutlass::bfloat16_t**)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(void*);
    cutlass::bfloat16_t** ptr_C = (cutlass::bfloat16_t**)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(void*);
    int64_t* lda_arr = (int64_t*)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(int64_t);
    int64_t* ldb_arr = (int64_t*)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(int64_t);
    int64_t* ldc_arr = (int64_t*)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(int64_t);

    // Copy pointer arrays to device (async)
    cudaMemcpyAsync(gathered_hidden, h_gathered, numExperts * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gate_up_out, h_gate_up_out, numExperts * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(activated_buf, h_activated, numExperts * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(down_out, h_down_out, numExperts * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dequant_gate_up, h_dequant_gu, numExperts * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dequant_down, h_dequant_down, numExperts * sizeof(void*), cudaMemcpyHostToDevice, stream);

    // Zero output and counts
    cudaMemsetAsync(output_f32, 0, numTokens * hiddenSize * sizeof(float), stream);
    cudaMemsetAsync(expert_counts, 0, numExperts * sizeof(int), stream);

    // Build gather indices
    {
        int total = numTokens * topK;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        cutlass_build_gather_kernel<<<blocks, threads, 0, stream>>>(
            (const int32_t*)expert_indices,
            gather_indices_buf,
            expert_counts,
            numTokens,
            topK,
            numExperts,
            maxPerExpert
        );
    }

    // Launch gather and dequant for ALL experts (kernels will process based on actual count)
    for (int e = 0; e < numExperts; e++) {
        // Gather hidden states
        {
            dim3 grid(maxPerExpert, (hiddenSize + 255) / 256);
            dim3 block(256);
            cutlass_gather_kernel<<<grid, block, 0, stream>>>(
                (const cutlass::bfloat16_t*)hidden,
                gather_indices_buf + e * maxPerExpert,
                h_gathered[e],
                maxPerExpert,
                hiddenSize,
                topK
            );
        }

        // Dequant gate_up weights
        {
            int total = gateUpSize * hiddenSize;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            cutlass_dequant_kernel<<<blocks, threads, 0, stream>>>(
                (const uint8_t*)gate_up_blocks,
                (const uint8_t*)gate_up_scales,
                h_dequant_gu[e],
                e, numExperts, gateUpSize, numBlocksIn, hiddenSize
            );
        }
    }

    // Setup GEMM problems on device (reads expert_counts from device memory)
    {
        int threads = 256;
        int blocks = (numExperts + threads - 1) / threads;
        cutlass_setup_gemm_problems_kernel<<<blocks, threads, 0, stream>>>(
            expert_counts,
            problem_sizes,
            ptr_A, ptr_B, ptr_C,
            lda_arr, ldb_arr, ldc_arr,
            gathered_hidden,
            dequant_gate_up,
            gate_up_out,
            gateUpSize,   // N
            hiddenSize,   // K
            numExperts
        );
    }

    // Run grouped GEMM for gate_up projection
    {
        typename GemmGrouped::Arguments args(
            problem_sizes,
            numExperts,
            512,
            {float(1.0f), float(0.0f)},
            reinterpret_cast<cutlass::bfloat16_t**>(ptr_A),
            reinterpret_cast<cutlass::bfloat16_t**>(ptr_B),
            ptr_C,
            ptr_C,
            lda_arr, ldb_arr, ldc_arr, ldc_arr
        );

        GemmGrouped gemm_op;
        cutlass::Status status = gemm_op.initialize(args, nullptr, stream);
        if (status != cutlass::Status::kSuccess) {
            return BINFER_ERROR_CUDA;
        }
        status = gemm_op(stream);
        if (status != cutlass::Status::kSuccess) {
            return BINFER_ERROR_CUDA;
        }
    }

    // Activation for all experts
    for (int e = 0; e < numExperts; e++) {
        if (gate_up_bias != nullptr) {
            int total = maxPerExpert * gateUpSize;
            int threads = 256;
            int blocks_bias = (total + threads - 1) / threads;
            cutlass_add_bias_kernel<<<blocks_bias, threads, 0, stream>>>(
                h_gate_up_out[e],
                (const cutlass::bfloat16_t*)gate_up_bias,
                e, maxPerExpert, gateUpSize
            );
        }

        int total = maxPerExpert * intermediateSize;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        cutlass_activation_kernel<<<blocks, threads, 0, stream>>>(
            h_gate_up_out[e],
            h_activated[e],
            maxPerExpert,
            intermediateSize,
            1.702f,
            7.0f
        );
    }

    // Dequant down weights
    for (int e = 0; e < numExperts; e++) {
        int total = hiddenSize * intermediateSize;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        cutlass_dequant_kernel<<<blocks, threads, 0, stream>>>(
            (const uint8_t*)down_blocks,
            (const uint8_t*)down_scales,
            h_dequant_down[e],
            e, numExperts, hiddenSize, numBlocksInter, intermediateSize
        );
    }

    // Setup down projection GEMM problems
    {
        int threads = 256;
        int blocks = (numExperts + threads - 1) / threads;
        cutlass_setup_gemm_problems_kernel<<<blocks, threads, 0, stream>>>(
            expert_counts,
            problem_sizes,
            ptr_A, ptr_B, ptr_C,
            lda_arr, ldb_arr, ldc_arr,
            activated_buf,
            dequant_down,
            down_out,
            hiddenSize,       // N
            intermediateSize, // K
            numExperts
        );
    }

    // Run grouped GEMM for down projection
    {
        typename GemmGrouped::Arguments args(
            problem_sizes,
            numExperts,
            512,
            {float(1.0f), float(0.0f)},
            reinterpret_cast<cutlass::bfloat16_t**>(ptr_A),
            reinterpret_cast<cutlass::bfloat16_t**>(ptr_B),
            ptr_C,
            ptr_C,
            lda_arr, ldb_arr, ldc_arr, ldc_arr
        );

        GemmGrouped gemm_op;
        cutlass::Status status = gemm_op.initialize(args, nullptr, stream);
        if (status != cutlass::Status::kSuccess) {
            return BINFER_ERROR_CUDA;
        }
        status = gemm_op(stream);
        if (status != cutlass::Status::kSuccess) {
            return BINFER_ERROR_CUDA;
        }
    }

    // Add bias and scatter to output
    for (int e = 0; e < numExperts; e++) {
        if (down_bias != nullptr) {
            int total = maxPerExpert * hiddenSize;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            cutlass_add_bias_kernel<<<blocks, threads, 0, stream>>>(
                h_down_out[e],
                (const cutlass::bfloat16_t*)down_bias,
                e, maxPerExpert, hiddenSize
            );
        }

        // Scatter with weights (reads count from device memory)
        {
            dim3 grid(maxPerExpert, (hiddenSize + 255) / 256);
            dim3 block(256);
            cutlass_scatter_async_kernel<<<grid, block, 0, stream>>>(
                h_down_out[e],
                gather_indices_buf + e * maxPerExpert,
                expert_counts,
                (const cutlass::bfloat16_t*)expert_weights,
                output_f32,
                e,
                maxPerExpert,
                hiddenSize,
                topK
            );
        }
    }

    // Convert to BF16
    {
        int total = numTokens * hiddenSize;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        cutlass_f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
            output_f32,
            (cutlass::bfloat16_t*)output,
            total
        );
    }

    return cudaGetLastError() == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Get scratch size (includes extra space for async version pointer arrays)
extern "C" size_t binfer_moe_cutlass_scratch_size(
    int numTokens,
    int hiddenSize,
    int intermediateSize,
    int numExperts,
    int topK
) {
    const int gateUpSize = intermediateSize * 2;
    const int maxPerExpert = numTokens * topK;

    // Device pointer arrays for async version (6 arrays of numExperts pointers)
    size_t total = numExperts * sizeof(void*) * 6;

    size_t per_expert = 0;
    per_expert += maxPerExpert * hiddenSize * sizeof(cutlass::bfloat16_t);      // gathered
    per_expert += maxPerExpert * gateUpSize * sizeof(cutlass::bfloat16_t);      // gate_up_out
    per_expert += maxPerExpert * intermediateSize * sizeof(cutlass::bfloat16_t); // activated
    per_expert += maxPerExpert * hiddenSize * sizeof(cutlass::bfloat16_t);      // down_out
    per_expert += gateUpSize * hiddenSize * sizeof(cutlass::bfloat16_t);        // dequant_gate_up
    per_expert += hiddenSize * intermediateSize * sizeof(cutlass::bfloat16_t);  // dequant_down

    total += per_expert * numExperts;
    total += numExperts * maxPerExpert * sizeof(int);  // gather indices
    total += numExperts * sizeof(int);                  // counts
    total += numTokens * hiddenSize * sizeof(float);    // output_f32

    // GEMM descriptors
    total += MAX_EXPERTS * sizeof(cutlass::gemm::GemmCoord);
    total += MAX_EXPERTS * sizeof(void*) * 3;  // A, B, C pointers
    total += MAX_EXPERTS * sizeof(int64_t) * 3; // lda, ldb, ldc

    return total;
}

// ============================================================================
// Expert Parallel (EP) version for tensor parallelism
// Each GPU only has expertsPerRank experts and processes only those
// ============================================================================

// EP version of build gather - only counts tokens for experts on this rank
__global__ void cutlass_build_gather_ep_kernel(
    const int32_t* __restrict__ expert_indices,
    int* __restrict__ gather_indices,
    int* __restrict__ expert_counts,
    int numTokens,
    int topK,
    int expertsPerRank,
    int rank,
    int maxPerExpert
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTokens * topK) return;

    int global_expert = expert_indices[tid];
    // Check if this expert is on our rank
    int expert_start = rank * expertsPerRank;
    int expert_end = expert_start + expertsPerRank;

    if (global_expert >= expert_start && global_expert < expert_end) {
        int local_expert = global_expert - expert_start;
        int slot = atomicAdd(&expert_counts[local_expert], 1);
        if (slot < maxPerExpert) {
            gather_indices[local_expert * maxPerExpert + slot] = tid;
        }
    }
}

// EP MoE forward with CUTLASS grouped GEMM
extern "C" BinferError binfer_moe_cutlass_forward_ep(
    const void* hidden,
    const void* gate_up_blocks,
    const void* gate_up_scales,
    const void* gate_up_bias,
    const void* down_blocks,
    const void* down_scales,
    const void* down_bias,
    const void* expert_indices,
    const void* expert_weights,
    void* output,
    void* scratch,
    size_t scratch_size,
    int numTokens,
    int hiddenSize,
    int intermediateSize,
    int expertsPerRank,
    int topK,
    int rank,
    int worldSize
) {
    if (expertsPerRank > MAX_EXPERTS) {
        fprintf(stderr, "MoE CUTLASS EP: too many experts per rank %d > %d\n", expertsPerRank, MAX_EXPERTS);
        return BINFER_ERROR_INVALID_ARGUMENT;
    }

    cudaStream_t stream = get_current_stream();
    const int numBlocksIn = hiddenSize / 32;
    const int numBlocksInter = intermediateSize / 32;
    const int gateUpSize = intermediateSize * 2;
    const int maxPerExpert = numTokens * topK;

    // Partition scratch buffer (same layout as non-EP, but for expertsPerRank)
    char* scratch_ptr = (char*)scratch;

    std::vector<cutlass::bfloat16_t*> gathered_hidden(expertsPerRank);
    std::vector<cutlass::bfloat16_t*> gate_up_out(expertsPerRank);
    std::vector<cutlass::bfloat16_t*> activated_buf(expertsPerRank);
    std::vector<cutlass::bfloat16_t*> down_out(expertsPerRank);
    std::vector<cutlass::bfloat16_t*> dequant_gate_up(expertsPerRank);
    std::vector<cutlass::bfloat16_t*> dequant_down(expertsPerRank);

    size_t per_expert_gathered = (size_t)maxPerExpert * hiddenSize * sizeof(cutlass::bfloat16_t);
    size_t per_expert_gate_up_out = (size_t)maxPerExpert * gateUpSize * sizeof(cutlass::bfloat16_t);
    size_t per_expert_activated = (size_t)maxPerExpert * intermediateSize * sizeof(cutlass::bfloat16_t);
    size_t per_expert_down_out = (size_t)maxPerExpert * hiddenSize * sizeof(cutlass::bfloat16_t);
    size_t per_expert_dequant_gu = (size_t)gateUpSize * hiddenSize * sizeof(cutlass::bfloat16_t);
    size_t per_expert_dequant_down = (size_t)hiddenSize * intermediateSize * sizeof(cutlass::bfloat16_t);

    for (int e = 0; e < expertsPerRank; e++) {
        gathered_hidden[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_gathered;
        gate_up_out[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_gate_up_out;
        activated_buf[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_activated;
        down_out[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_down_out;
        dequant_gate_up[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_dequant_gu;
        dequant_down[e] = (cutlass::bfloat16_t*)scratch_ptr; scratch_ptr += per_expert_dequant_down;
    }

    int* gather_indices_buf = (int*)scratch_ptr;
    scratch_ptr += expertsPerRank * maxPerExpert * sizeof(int);
    int* expert_counts = (int*)scratch_ptr;
    scratch_ptr += expertsPerRank * sizeof(int);
    float* output_f32 = (float*)scratch_ptr;
    scratch_ptr += numTokens * hiddenSize * sizeof(float);

    // Problem descriptors for grouped GEMM
    cutlass::gemm::GemmCoord* problem_sizes = (cutlass::gemm::GemmCoord*)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(cutlass::gemm::GemmCoord);
    cutlass::bfloat16_t** ptr_A = (cutlass::bfloat16_t**)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(void*);
    cutlass::bfloat16_t** ptr_B = (cutlass::bfloat16_t**)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(void*);
    cutlass::bfloat16_t** ptr_C = (cutlass::bfloat16_t**)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(void*);
    int64_t* lda = (int64_t*)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(int64_t);
    int64_t* ldb = (int64_t*)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(int64_t);
    int64_t* ldc = (int64_t*)scratch_ptr;
    scratch_ptr += MAX_EXPERTS * sizeof(int64_t);

    // Zero output and counts - output starts as zero, allReduce will combine
    cudaMemsetAsync(output_f32, 0, numTokens * hiddenSize * sizeof(float), stream);
    cudaMemsetAsync(expert_counts, 0, expertsPerRank * sizeof(int), stream);

    // Build gather indices - EP version filters by rank
    {
        int total = numTokens * topK;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        cutlass_build_gather_ep_kernel<<<blocks, threads, 0, stream>>>(
            (const int32_t*)expert_indices,
            gather_indices_buf,
            expert_counts,
            numTokens,
            topK,
            expertsPerRank,
            rank,
            maxPerExpert
        );
    }

    // Copy counts to host (sync point - unavoidable for dynamic expert counts)
    cudaStreamSynchronize(stream);
    int h_counts[MAX_EXPERTS];
    cudaMemcpy(h_counts, expert_counts, expertsPerRank * sizeof(int), cudaMemcpyDeviceToHost);

    // Find active local experts
    int active_experts = 0;
    int active_indices[MAX_EXPERTS];
    for (int e = 0; e < expertsPerRank; e++) {
        if (h_counts[e] > 0) {
            active_indices[active_experts++] = e;
        }
    }

    if (active_experts == 0) {
        // No experts on this rank selected - output zeros
        cudaMemsetAsync(output, 0, numTokens * hiddenSize * sizeof(cutlass::bfloat16_t), stream);
        return BINFER_SUCCESS;
    }

    // Prepare host arrays for grouped GEMM
    std::vector<cutlass::gemm::GemmCoord> h_problem_sizes(active_experts);
    std::vector<cutlass::bfloat16_t*> h_ptr_A(active_experts);
    std::vector<cutlass::bfloat16_t*> h_ptr_B(active_experts);
    std::vector<cutlass::bfloat16_t*> h_ptr_C(active_experts);
    std::vector<int64_t> h_lda(active_experts);
    std::vector<int64_t> h_ldb(active_experts);
    std::vector<int64_t> h_ldc(active_experts);

    // Launch gather and dequant for each active local expert
    for (int i = 0; i < active_experts; i++) {
        int local_e = active_indices[i];
        int count = h_counts[local_e];
        int* expert_gather = gather_indices_buf + local_e * maxPerExpert;

        // Gather hidden states
        {
            dim3 grid(count, (hiddenSize + 255) / 256);
            dim3 block(256);
            cutlass_gather_kernel<<<grid, block, 0, stream>>>(
                (const cutlass::bfloat16_t*)hidden,
                expert_gather,
                gathered_hidden[local_e],
                count,
                hiddenSize,
                topK
            );
        }

        // Dequant gate_up weights - use LOCAL expert index for weight access
        {
            int total = gateUpSize * hiddenSize;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            cutlass_dequant_kernel<<<blocks, threads, 0, stream>>>(
                (const uint8_t*)gate_up_blocks,
                (const uint8_t*)gate_up_scales,
                dequant_gate_up[local_e],
                local_e, expertsPerRank, gateUpSize, numBlocksIn, hiddenSize
            );
        }

        h_problem_sizes[i] = cutlass::gemm::GemmCoord(count, gateUpSize, hiddenSize);
        h_ptr_A[i] = gathered_hidden[local_e];
        h_ptr_B[i] = dequant_gate_up[local_e];
        h_ptr_C[i] = gate_up_out[local_e];
        h_lda[i] = hiddenSize;
        h_ldb[i] = hiddenSize;
        h_ldc[i] = gateUpSize;
    }

    cudaStreamSynchronize(stream);

    // Copy problem descriptors to device
    cudaMemcpyAsync(problem_sizes, h_problem_sizes.data(),
               active_experts * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ptr_A, h_ptr_A.data(), active_experts * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ptr_B, h_ptr_B.data(), active_experts * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ptr_C, h_ptr_C.data(), active_experts * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(lda, h_lda.data(), active_experts * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ldb, h_ldb.data(), active_experts * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ldc, h_ldc.data(), active_experts * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Run grouped GEMM for gate_up projection
    {
        typename GemmGrouped::Arguments args(
            problem_sizes,
            active_experts,
            512,
            {float(1.0f), float(0.0f)},
            reinterpret_cast<cutlass::bfloat16_t**>(ptr_A),
            reinterpret_cast<cutlass::bfloat16_t**>(ptr_B),
            ptr_C,
            ptr_C,
            lda,
            ldb,
            ldc,
            ldc
        );

        GemmGrouped gemm_op;
        cutlass::Status status = gemm_op.initialize(args, nullptr, stream);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "MoE CUTLASS EP: gate_up GEMM init failed\n");
            return BINFER_ERROR_CUDA;
        }

        status = gemm_op(stream);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "MoE CUTLASS EP: gate_up GEMM run failed\n");
            return BINFER_ERROR_CUDA;
        }
    }

    // Add gate_up bias and apply activation
    for (int i = 0; i < active_experts; i++) {
        int local_e = active_indices[i];
        int count = h_counts[local_e];

        if (gate_up_bias != nullptr) {
            int total = count * gateUpSize;
            int threads = 256;
            int blocks_bias = (total + threads - 1) / threads;
            cutlass_add_bias_kernel<<<blocks_bias, threads, 0, stream>>>(
                gate_up_out[local_e],
                (const cutlass::bfloat16_t*)gate_up_bias,
                local_e, count, gateUpSize
            );
        }

        int total = count * intermediateSize;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        cutlass_activation_kernel<<<blocks, threads, 0, stream>>>(
            gate_up_out[local_e],
            activated_buf[local_e],
            count,
            intermediateSize,
            1.702f,
            7.0f
        );
    }

    // Prepare down projection
    for (int i = 0; i < active_experts; i++) {
        int local_e = active_indices[i];
        int count = h_counts[local_e];

        // Dequant down weights
        {
            int total = hiddenSize * intermediateSize;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            cutlass_dequant_kernel<<<blocks, threads, 0, stream>>>(
                (const uint8_t*)down_blocks,
                (const uint8_t*)down_scales,
                dequant_down[local_e],
                local_e, expertsPerRank, hiddenSize, numBlocksInter, intermediateSize
            );
        }

        h_problem_sizes[i] = cutlass::gemm::GemmCoord(count, hiddenSize, intermediateSize);
        h_ptr_A[i] = activated_buf[local_e];
        h_ptr_B[i] = dequant_down[local_e];
        h_ptr_C[i] = down_out[local_e];
        h_lda[i] = intermediateSize;
        h_ldb[i] = intermediateSize;
        h_ldc[i] = hiddenSize;
    }

    cudaStreamSynchronize(stream);

    // Copy down projection descriptors
    cudaMemcpyAsync(problem_sizes, h_problem_sizes.data(),
               active_experts * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ptr_A, h_ptr_A.data(), active_experts * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ptr_B, h_ptr_B.data(), active_experts * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ptr_C, h_ptr_C.data(), active_experts * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(lda, h_lda.data(), active_experts * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ldb, h_ldb.data(), active_experts * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ldc, h_ldc.data(), active_experts * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Run grouped GEMM for down projection
    {
        typename GemmGrouped::Arguments args(
            problem_sizes,
            active_experts,
            512,
            {float(1.0f), float(0.0f)},
            reinterpret_cast<cutlass::bfloat16_t**>(ptr_A),
            reinterpret_cast<cutlass::bfloat16_t**>(ptr_B),
            ptr_C,
            ptr_C,
            lda,
            ldb,
            ldc,
            ldc
        );

        GemmGrouped gemm_op;
        cutlass::Status status = gemm_op.initialize(args, nullptr, stream);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "MoE CUTLASS EP: down GEMM init failed\n");
            return BINFER_ERROR_CUDA;
        }

        status = gemm_op(stream);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "MoE CUTLASS EP: down GEMM run failed\n");
            return BINFER_ERROR_CUDA;
        }
    }

    // Add down bias and scatter accumulate
    for (int i = 0; i < active_experts; i++) {
        int local_e = active_indices[i];
        int count = h_counts[local_e];
        int* expert_gather = gather_indices_buf + local_e * maxPerExpert;

        if (down_bias != nullptr) {
            int total = count * hiddenSize;
            int threads = 256;
            int blocks_bias = (total + threads - 1) / threads;
            cutlass_add_bias_kernel<<<blocks_bias, threads, 0, stream>>>(
                down_out[local_e],
                (const cutlass::bfloat16_t*)down_bias,
                local_e, count, hiddenSize
            );
        }

        dim3 grid(count, (hiddenSize + 255) / 256);
        dim3 block(256);
        cutlass_scatter_kernel<<<grid, block, 0, stream>>>(
            down_out[local_e],
            expert_gather,
            (const cutlass::bfloat16_t*)expert_weights,
            output_f32,
            count,
            hiddenSize,
            topK
        );
    }

    // Convert to BF16
    {
        int total = numTokens * hiddenSize;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        cutlass_f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
            output_f32,
            (cutlass::bfloat16_t*)output,
            total
        );
    }

    return cudaGetLastError() == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Get scratch size for EP version
extern "C" size_t binfer_moe_cutlass_scratch_size_ep(
    int numTokens,
    int hiddenSize,
    int intermediateSize,
    int expertsPerRank,
    int topK
) {
    const int gateUpSize = intermediateSize * 2;
    const int maxPerExpert = numTokens * topK;

    size_t per_expert = 0;
    per_expert += maxPerExpert * hiddenSize * sizeof(cutlass::bfloat16_t);
    per_expert += maxPerExpert * gateUpSize * sizeof(cutlass::bfloat16_t);
    per_expert += maxPerExpert * intermediateSize * sizeof(cutlass::bfloat16_t);
    per_expert += maxPerExpert * hiddenSize * sizeof(cutlass::bfloat16_t);
    per_expert += gateUpSize * hiddenSize * sizeof(cutlass::bfloat16_t);
    per_expert += hiddenSize * intermediateSize * sizeof(cutlass::bfloat16_t);

    size_t total = per_expert * expertsPerRank;
    total += expertsPerRank * maxPerExpert * sizeof(int);
    total += expertsPerRank * sizeof(int);
    total += numTokens * hiddenSize * sizeof(float);

    total += MAX_EXPERTS * sizeof(cutlass::gemm::GemmCoord);
    total += MAX_EXPERTS * sizeof(void*) * 3;
    total += MAX_EXPERTS * sizeof(int64_t) * 3;

    return total;
}
