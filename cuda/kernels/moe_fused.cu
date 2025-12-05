// Fused MoE expert kernels - optimized for single-token decode
// Key optimizations:
// 1. Expert indices stay on GPU - no CPU sync needed
// 2. Processes all top-k experts in parallel
// 3. Coalesced memory access patterns
// 4. Shared memory for weight block caching

#include "../include/binfer.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// External function from gemm.cu to get current stream for CUDA graph capture
extern cudaStream_t get_current_stream();

// E8M0 lookup table - compute scale inline instead of extern table
// E8M0: 8-bit unsigned exponent, bias=127
// value = 2^(exp - 127) for exp in [0, 254], exp=255 is NaN
__device__ __forceinline__ float e8m0_to_float_inline(uint8_t val) {
    if (val == 255) return NAN;
    // 2^(val - 127) = ldexpf(1.0f, val - 127)
    // Use bit manipulation: for 2^n where n can be negative,
    // IEEE float exponent = n + 127 (with bias), mantissa = 0
    // So for 2^(val - 127), the IEEE exponent field = (val - 127) + 127 = val
    union { float f; uint32_t u; } converter;
    converter.u = (uint32_t)val << 23;  // val is already the biased exponent
    return converter.f;
}

// FP4 E2M1 to float lookup table
__device__ __constant__ float fused_fp4_table[16] = {
    0.0f,    // 0000: subnormal, m=0 -> 0 * 0.5 = 0
    0.5f,    // 0001: subnormal, m=1 -> 1 * 0.5 = 0.5
    1.0f,    // 0010: e=1, m=0 -> 1.0 * 2^0 = 1
    1.5f,    // 0011: e=1, m=1 -> 1.5 * 2^0 = 1.5
    2.0f,    // 0100: e=2, m=0 -> 1.0 * 2^1 = 2
    3.0f,    // 0101: e=2, m=1 -> 1.5 * 2^1 = 3
    4.0f,    // 0110: e=3, m=0 -> 1.0 * 2^2 = 4
    6.0f,    // 0111: e=3, m=1 -> 1.5 * 2^2 = 6
    -0.0f,   // 1000: -0
    -0.5f,   // 1001: -0.5
    -1.0f,   // 1010: -1
    -1.5f,   // 1011: -1.5
    -2.0f,   // 1100: -2
    -3.0f,   // 1101: -3
    -4.0f,   // 1110: -4
    -6.0f,   // 1111: -6
};

// Tile sizes for shared memory
#define TILE_K 32  // Process 32 input features at a time (one MXFP4 block)
#define TILE_N 32  // 32 output features per thread block

// Helper kernel to convert BF16 expert weights to F32 on GPU
__global__ void bf16_to_f32_kernel(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __bfloat162float(input[idx]);
    }
}

// First kernel: Compute gate_up projection (hidden @ gate_up_weight^T)
// Each block computes a tile of the output for one expert
__global__ void moe_gate_up_kernel(
    const __nv_bfloat16* __restrict__ hidden,       // [hidden_size]
    const uint8_t* __restrict__ blocks,             // [num_experts, intermediate*2, num_blocks, 16]
    const uint8_t* __restrict__ scales,             // [num_experts, intermediate*2, num_blocks]
    const __nv_bfloat16* __restrict__ bias,         // [num_experts, intermediate*2] or nullptr
    const int32_t* __restrict__ expert_indices,     // [top_k]
    __nv_bfloat16* __restrict__ gate_up_out,        // [top_k, intermediate*2]
    int hidden_size,
    int intermediate_size,
    int num_blocks,
    int num_experts,
    int top_k
) {
    // expert_k = which of the top-k experts we're processing
    const int expert_k = blockIdx.z;
    if (expert_k >= top_k) return;

    const int expert_idx = expert_indices[expert_k];
    if (expert_idx < 0) return;

    // Output index within gate_up (0 to intermediate*2 - 1)
    const int out_base = blockIdx.x * TILE_N;
    const int out_local = threadIdx.x;
    const int out_idx = out_base + out_local;

    if (out_idx >= intermediate_size * 2) return;

    // Shared memory for hidden tile and weight tile
    __shared__ float shared_hidden[TILE_K];
    __shared__ float shared_weights[TILE_N][TILE_K];

    float accum = 0.0f;

    // Process hidden in tiles of TILE_K
    const int out_features = intermediate_size * 2;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        // Load hidden tile to shared memory
        if (threadIdx.x < TILE_K) {
            int h_idx = block_idx * 32 + threadIdx.x;
            if (h_idx < hidden_size) {
                shared_hidden[threadIdx.x] = __bfloat162float(hidden[h_idx]);
            } else {
                shared_hidden[threadIdx.x] = 0.0f;
            }
        }

        // Load weight block for this output row
        // blocks layout: [expert_idx, out_idx, block_idx, 16]
        // scales layout: [expert_idx, out_idx, block_idx]
        const int block_offset = (expert_idx * out_features + out_idx) * num_blocks + block_idx;
        float scale = e8m0_to_float_inline(scales[block_offset]);

        // Dequantize the 16 bytes (32 FP4 values) into shared memory
        if (out_local < TILE_N && out_idx < out_features) {
            for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
                uint8_t packed = blocks[block_offset * 16 + byte_idx];
                shared_weights[out_local][byte_idx * 2] = fused_fp4_table[packed & 0xF] * scale;
                shared_weights[out_local][byte_idx * 2 + 1] = fused_fp4_table[(packed >> 4) & 0xF] * scale;
            }
        }
        __syncthreads();

        // Compute dot product for this tile
        if (out_idx < out_features) {
            for (int k = 0; k < TILE_K; k++) {
                accum += shared_hidden[k] * shared_weights[out_local][k];
            }
        }
        __syncthreads();
    }

    // Write result (with optional bias)
    if (out_idx < out_features) {
        if (bias != nullptr) {
            // Bias layout: [num_experts, intermediate*2] - add expert-specific bias
            accum += __bfloat162float(bias[expert_idx * out_features + out_idx]);
        }
        gate_up_out[expert_k * out_features + out_idx] = __float2bfloat16(accum);
    }
}

// Second kernel: Apply GPT-OSS activation
// Input: gate_up [top_k, intermediate*2] (gate and up INTERLEAVED)
// Output: activated [top_k, intermediate]
// NOTE: gate_up format is INTERLEAVED [gate_0, up_0, gate_1, up_1, ...]
__global__ void moe_activation_kernel(
    const __nv_bfloat16* __restrict__ gate_up,      // [top_k, intermediate*2]
    __nv_bfloat16* __restrict__ activated,          // [top_k, intermediate]
    const int32_t* __restrict__ expert_indices,     // [top_k]
    int intermediate_size,
    int top_k,
    float alpha,
    float limit
) {
    const int expert_k = blockIdx.y;
    if (expert_k >= top_k) return;
    if (expert_indices[expert_k] < 0) return;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= intermediate_size) return;

    // Interleaved format: gate = gate_up[::2], up = gate_up[1::2]
    const int gate_idx = expert_k * intermediate_size * 2 + idx * 2;
    const int up_idx = expert_k * intermediate_size * 2 + idx * 2 + 1;

    float gate = __bfloat162float(gate_up[gate_idx]);
    float up = __bfloat162float(gate_up[up_idx]);

    // GPT-OSS activation: (up + 1) * gate * sigmoid(gate * alpha)
    gate = fminf(gate, limit);
    up = fmaxf(fminf(up, limit), -limit);
    float glu = gate / (1.0f + expf(-gate * alpha));
    float result = (up + 1.0f) * glu;

    activated[expert_k * intermediate_size + idx] = __float2bfloat16(result);
}

// Third kernel: Compute down projection and accumulate
// Each block processes one output dimension across all experts
__global__ void moe_down_accum_kernel(
    const __nv_bfloat16* __restrict__ activated,    // [top_k, intermediate]
    const uint8_t* __restrict__ blocks,             // [num_experts, hidden, num_blocks, 16]
    const uint8_t* __restrict__ scales,             // [num_experts, hidden, num_blocks]
    const __nv_bfloat16* __restrict__ bias,         // [num_experts, hidden] or nullptr
    const int32_t* __restrict__ expert_indices,     // [top_k]
    const float* __restrict__ expert_weights,       // [top_k]
    __nv_bfloat16* __restrict__ output,             // [hidden_size]
    int hidden_size,
    int intermediate_size,
    int num_blocks,
    int num_experts,
    int top_k
) {
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= hidden_size) return;

    __shared__ float shared_inter[256];  // Cache for intermediate values

    float out_accum = 0.0f;

    // Process each selected expert
    for (int k = 0; k < top_k; k++) {
        const int expert_idx = expert_indices[k];
        if (expert_idx < 0) continue;

        const float expert_weight = expert_weights[k];
        float down = 0.0f;

        // Process intermediate in tiles
        for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
            // Load intermediate tile to shared memory (cooperative)
            const int tile_start = block_idx * 32;
            if (threadIdx.x < 32 && tile_start + threadIdx.x < intermediate_size) {
                shared_inter[threadIdx.x] = __bfloat162float(
                    activated[k * intermediate_size + tile_start + threadIdx.x]
                );
            }
            __syncthreads();

            // Dequant weight block and accumulate
            const int block_offset = (expert_idx * hidden_size + out_idx) * num_blocks + block_idx;
            float scale = e8m0_to_float_inline(scales[block_offset]);

            for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
                uint8_t packed = blocks[block_offset * 16 + byte_idx];
                int pos0 = byte_idx * 2;
                int pos1 = byte_idx * 2 + 1;

                if (tile_start + pos0 < intermediate_size) {
                    down += shared_inter[pos0] * (fused_fp4_table[packed & 0xF] * scale);
                }
                if (tile_start + pos1 < intermediate_size) {
                    down += shared_inter[pos1] * (fused_fp4_table[(packed >> 4) & 0xF] * scale);
                }
            }
            __syncthreads();
        }

        // Add bias for this expert (after down projection, before scaling)
        if (bias != nullptr) {
            down += __bfloat162float(bias[expert_idx * hidden_size + out_idx]);
        }

        out_accum += expert_weight * down;
    }

    output[out_idx] = __float2bfloat16(out_accum);
}

extern "C" BinferError binfer_moe_fused_forward(
    const void* hidden,
    const void* gate_up_blocks,
    const void* gate_up_scales,
    const void* gate_up_bias,     // [num_experts, intermediate*2] or nullptr
    const void* down_blocks,
    const void* down_scales,
    const void* down_bias,        // [num_experts, hidden] or nullptr
    const void* expert_indices,
    const void* expert_weights,   // BF16 weights from router
    void* output,
    int hidden_size,
    int intermediate_size,
    int num_experts,
    int top_k
) {
    const int num_blocks_in = hidden_size / 32;
    const int num_blocks_inter = intermediate_size / 32;
    const int out_features = intermediate_size * 2;

    // Allocate intermediate buffers
    __nv_bfloat16* gate_up_out;
    __nv_bfloat16* activated;
    float* expert_weights_f32;

    cudaMalloc(&gate_up_out, top_k * out_features * sizeof(__nv_bfloat16));
    cudaMalloc(&activated, top_k * intermediate_size * sizeof(__nv_bfloat16));
    cudaMalloc(&expert_weights_f32, top_k * sizeof(float));

    // 0. Convert BF16 expert weights to F32
    {
        dim3 grid((top_k + 255) / 256, 1, 1);
        dim3 block(256, 1, 1);
        bf16_to_f32_kernel<<<grid, block, 0, get_current_stream()>>>(
            (const __nv_bfloat16*)expert_weights,
            expert_weights_f32,
            top_k
        );
    }

    // 1. Gate-up projection
    {
        dim3 grid((out_features + TILE_N - 1) / TILE_N, 1, top_k);
        dim3 block(TILE_N, 1, 1);
        moe_gate_up_kernel<<<grid, block, 0, get_current_stream()>>>(
            (const __nv_bfloat16*)hidden,
            (const uint8_t*)gate_up_blocks,
            (const uint8_t*)gate_up_scales,
            (const __nv_bfloat16*)gate_up_bias,
            (const int32_t*)expert_indices,
            gate_up_out,
            hidden_size,
            intermediate_size,
            num_blocks_in,
            num_experts,
            top_k
        );
    }

    // 2. Activation
    {
        dim3 grid((intermediate_size + 255) / 256, top_k, 1);
        dim3 block(256, 1, 1);
        moe_activation_kernel<<<grid, block, 0, get_current_stream()>>>(
            gate_up_out,
            activated,
            (const int32_t*)expert_indices,
            intermediate_size,
            top_k,
            1.702f,  // GPT-OSS alpha
            7.0f     // GPT-OSS limit
        );
    }

    // 3. Down projection with accumulation
    {
        dim3 grid((hidden_size + 255) / 256, 1, 1);
        dim3 block(256, 1, 1);
        moe_down_accum_kernel<<<grid, block, 0, get_current_stream()>>>(
            activated,
            (const uint8_t*)down_blocks,
            (const uint8_t*)down_scales,
            (const __nv_bfloat16*)down_bias,
            (const int32_t*)expert_indices,
            expert_weights_f32,
            (__nv_bfloat16*)output,
            hidden_size,
            intermediate_size,
            num_blocks_inter,
            num_experts,
            top_k
        );
    }

    cudaFree(gate_up_out);
    cudaFree(activated);
    cudaFree(expert_weights_f32);

    return cudaGetLastError() == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// Expert Parallel MoE kernels for tensor parallelism
// Each GPU owns a subset of experts: experts [rank*experts_per_rank, (rank+1)*experts_per_rank)
// The kernel filters to only process experts owned by this rank and maps global to local indices
// ============================================================================

// EP version of gate_up kernel - filters experts by rank ownership
__global__ void moe_gate_up_kernel_ep(
    const __nv_bfloat16* __restrict__ hidden,       // [hidden_size]
    const uint8_t* __restrict__ blocks,             // [experts_per_rank, intermediate*2, num_blocks, 16]
    const uint8_t* __restrict__ scales,             // [experts_per_rank, intermediate*2, num_blocks]
    const __nv_bfloat16* __restrict__ bias,         // [experts_per_rank, intermediate*2] or nullptr
    const int32_t* __restrict__ expert_indices,     // [top_k] - global expert indices
    __nv_bfloat16* __restrict__ gate_up_out,        // [top_k, intermediate*2]
    int hidden_size,
    int intermediate_size,
    int num_blocks,
    int experts_per_rank,
    int top_k,
    int rank,
    int world_size
) {
    const int expert_k = blockIdx.z;
    if (expert_k >= top_k) return;

    const int global_expert_idx = expert_indices[expert_k];
    if (global_expert_idx < 0) return;

    // Check if this expert belongs to this rank
    // The total number of experts is experts_per_rank * world_size
    // Each rank owns experts [rank * experts_per_rank, (rank+1) * experts_per_rank)
    const int expert_owner = global_expert_idx / experts_per_rank;

    if (expert_owner != rank) return;

    // Map to local expert index
    const int local_expert_idx = global_expert_idx % experts_per_rank;

    const int out_base = blockIdx.x * TILE_N;
    const int out_local = threadIdx.x;
    const int out_idx = out_base + out_local;

    if (out_idx >= intermediate_size * 2) return;

    __shared__ float shared_hidden[TILE_K];
    __shared__ float shared_weights[TILE_N][TILE_K];

    float accum = 0.0f;
    const int out_features = intermediate_size * 2;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        if (threadIdx.x < TILE_K) {
            int h_idx = block_idx * 32 + threadIdx.x;
            if (h_idx < hidden_size) {
                shared_hidden[threadIdx.x] = __bfloat162float(hidden[h_idx]);
            } else {
                shared_hidden[threadIdx.x] = 0.0f;
            }
        }

        // Use LOCAL expert index to access weights
        const int block_offset = (local_expert_idx * out_features + out_idx) * num_blocks + block_idx;
        float scale = e8m0_to_float_inline(scales[block_offset]);

        if (out_local < TILE_N && out_idx < out_features) {
            for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
                uint8_t packed = blocks[block_offset * 16 + byte_idx];
                shared_weights[out_local][byte_idx * 2] = fused_fp4_table[packed & 0xF] * scale;
                shared_weights[out_local][byte_idx * 2 + 1] = fused_fp4_table[(packed >> 4) & 0xF] * scale;
            }
        }
        __syncthreads();

        if (out_idx < out_features) {
            for (int k = 0; k < TILE_K; k++) {
                accum += shared_hidden[k] * shared_weights[out_local][k];
            }
        }
        __syncthreads();
    }

    if (out_idx < out_features) {
        if (bias != nullptr) {
            accum += __bfloat162float(bias[local_expert_idx * out_features + out_idx]);
        }
        gate_up_out[expert_k * out_features + out_idx] = __float2bfloat16(accum);
    }
}

// EP version of activation kernel - filters by rank ownership
__global__ void moe_activation_kernel_ep(
    const __nv_bfloat16* __restrict__ gate_up,
    __nv_bfloat16* __restrict__ activated,
    const int32_t* __restrict__ expert_indices,
    int intermediate_size,
    int top_k,
    float alpha,
    float limit,
    int experts_per_rank,
    int rank
) {
    const int expert_k = blockIdx.y;
    if (expert_k >= top_k) return;

    const int global_expert_idx = expert_indices[expert_k];
    if (global_expert_idx < 0) return;

    // Check if this expert belongs to this rank
    if (global_expert_idx / experts_per_rank != rank) return;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= intermediate_size) return;

    const int gate_idx = expert_k * intermediate_size * 2 + idx * 2;
    const int up_idx = expert_k * intermediate_size * 2 + idx * 2 + 1;

    float gate = __bfloat162float(gate_up[gate_idx]);
    float up = __bfloat162float(gate_up[up_idx]);

    gate = fminf(gate, limit);
    up = fmaxf(fminf(up, limit), -limit);
    float glu = gate / (1.0f + expf(-gate * alpha));
    float result = (up + 1.0f) * glu;

    activated[expert_k * intermediate_size + idx] = __float2bfloat16(result);
}

// EP version of down_accum kernel - processes only local experts, outputs partial result
__global__ void moe_down_accum_kernel_ep(
    const __nv_bfloat16* __restrict__ activated,
    const uint8_t* __restrict__ blocks,             // [experts_per_rank, hidden, num_blocks, 16]
    const uint8_t* __restrict__ scales,             // [experts_per_rank, hidden, num_blocks]
    const __nv_bfloat16* __restrict__ bias,         // [experts_per_rank, hidden] or nullptr
    const int32_t* __restrict__ expert_indices,
    const float* __restrict__ expert_weights,
    __nv_bfloat16* __restrict__ output,             // [hidden_size] - partial result for this rank
    int hidden_size,
    int intermediate_size,
    int num_blocks,
    int experts_per_rank,
    int top_k,
    int rank,
    int world_size
) {
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= hidden_size) return;

    __shared__ float shared_inter[256];

    float out_accum = 0.0f;

    for (int k = 0; k < top_k; k++) {
        const int global_expert_idx = expert_indices[k];
        if (global_expert_idx < 0) continue;

        // Check if this expert belongs to this rank
        if (global_expert_idx / experts_per_rank != rank) continue;

        const int local_expert_idx = global_expert_idx % experts_per_rank;
        const float expert_weight = expert_weights[k];
        float down = 0.0f;

        for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
            const int tile_start = block_idx * 32;
            if (threadIdx.x < 32 && tile_start + threadIdx.x < intermediate_size) {
                shared_inter[threadIdx.x] = __bfloat162float(
                    activated[k * intermediate_size + tile_start + threadIdx.x]
                );
            }
            __syncthreads();

            // Use LOCAL expert index to access weights
            const int block_offset = (local_expert_idx * hidden_size + out_idx) * num_blocks + block_idx;
            float scale = e8m0_to_float_inline(scales[block_offset]);

            for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
                uint8_t packed = blocks[block_offset * 16 + byte_idx];
                int pos0 = byte_idx * 2;
                int pos1 = byte_idx * 2 + 1;

                if (tile_start + pos0 < intermediate_size) {
                    down += shared_inter[pos0] * (fused_fp4_table[packed & 0xF] * scale);
                }
                if (tile_start + pos1 < intermediate_size) {
                    down += shared_inter[pos1] * (fused_fp4_table[(packed >> 4) & 0xF] * scale);
                }
            }
            __syncthreads();
        }

        if (bias != nullptr) {
            down += __bfloat162float(bias[local_expert_idx * hidden_size + out_idx]);
        }

        out_accum += expert_weight * down;
    }

    output[out_idx] = __float2bfloat16(out_accum);
}

// Expert-parallel MoE forward pass
// Returns partial result that should be allReduced across ranks
extern "C" BinferError binfer_moe_fused_forward_ep(
    const void* hidden,
    const void* gate_up_blocks,      // [experts_per_rank, intermediate*2, num_blocks, 16]
    const void* gate_up_scales,      // [experts_per_rank, intermediate*2, num_blocks]
    const void* gate_up_bias,
    const void* down_blocks,         // [experts_per_rank, hidden, num_blocks, 16]
    const void* down_scales,         // [experts_per_rank, hidden, num_blocks]
    const void* down_bias,
    const void* expert_indices,      // [top_k] - global expert indices from router
    const void* expert_weights,      // [top_k] - BF16 weights from router
    void* output,                    // [hidden_size] - partial result
    int hidden_size,
    int intermediate_size,
    int experts_per_rank,            // number of experts on this rank
    int top_k,
    int rank,
    int world_size
) {
    const int num_blocks_in = hidden_size / 32;
    const int num_blocks_inter = intermediate_size / 32;
    const int out_features = intermediate_size * 2;

    // Allocate intermediate buffers
    __nv_bfloat16* gate_up_out;
    __nv_bfloat16* activated;
    float* expert_weights_f32;

    cudaMalloc(&gate_up_out, top_k * out_features * sizeof(__nv_bfloat16));
    cudaMalloc(&activated, top_k * intermediate_size * sizeof(__nv_bfloat16));
    cudaMalloc(&expert_weights_f32, top_k * sizeof(float));

    // Zero initialize gate_up_out and activated (for experts not on this rank)
    cudaMemsetAsync(gate_up_out, 0, top_k * out_features * sizeof(__nv_bfloat16), get_current_stream());
    cudaMemsetAsync(activated, 0, top_k * intermediate_size * sizeof(__nv_bfloat16), get_current_stream());

    // Convert BF16 expert weights to F32
    {
        dim3 grid((top_k + 255) / 256, 1, 1);
        dim3 block(256, 1, 1);
        bf16_to_f32_kernel<<<grid, block, 0, get_current_stream()>>>(
            (const __nv_bfloat16*)expert_weights,
            expert_weights_f32,
            top_k
        );
    }

    // 1. Gate-up projection (EP version)
    {
        dim3 grid((out_features + TILE_N - 1) / TILE_N, 1, top_k);
        dim3 block(TILE_N, 1, 1);
        moe_gate_up_kernel_ep<<<grid, block, 0, get_current_stream()>>>(
            (const __nv_bfloat16*)hidden,
            (const uint8_t*)gate_up_blocks,
            (const uint8_t*)gate_up_scales,
            (const __nv_bfloat16*)gate_up_bias,
            (const int32_t*)expert_indices,
            gate_up_out,
            hidden_size,
            intermediate_size,
            num_blocks_in,
            experts_per_rank,
            top_k,
            rank,
            world_size
        );
    }

    // 2. Activation (EP version)
    {
        dim3 grid((intermediate_size + 255) / 256, top_k, 1);
        dim3 block(256, 1, 1);
        moe_activation_kernel_ep<<<grid, block, 0, get_current_stream()>>>(
            gate_up_out,
            activated,
            (const int32_t*)expert_indices,
            intermediate_size,
            top_k,
            1.702f,  // GPT-OSS alpha
            7.0f,    // GPT-OSS limit
            experts_per_rank,
            rank
        );
    }

    // 3. Down projection with accumulation (EP version)
    {
        dim3 grid((hidden_size + 255) / 256, 1, 1);
        dim3 block(256, 1, 1);
        moe_down_accum_kernel_ep<<<grid, block, 0, get_current_stream()>>>(
            activated,
            (const uint8_t*)down_blocks,
            (const uint8_t*)down_scales,
            (const __nv_bfloat16*)down_bias,
            (const int32_t*)expert_indices,
            expert_weights_f32,
            (__nv_bfloat16*)output,
            hidden_size,
            intermediate_size,
            num_blocks_inter,
            experts_per_rank,
            top_k,
            rank,
            world_size
        );
    }

    cudaFree(gate_up_out);
    cudaFree(activated);
    cudaFree(expert_weights_f32);

    return cudaGetLastError() == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Debug function: just run gate_up kernel and return output
extern "C" BinferError binfer_moe_gate_up_debug(
    const void* hidden,
    const void* gate_up_blocks,
    const void* gate_up_scales,
    const void* expert_indices,
    void* gate_up_out,  // output buffer [top_k, intermediate*2]
    int hidden_size,
    int intermediate_size,
    int num_blocks,
    int num_experts,
    int top_k
) {
    const int out_features = intermediate_size * 2;

    dim3 grid((out_features + TILE_N - 1) / TILE_N, 1, top_k);
    dim3 block(TILE_N, 1, 1);
    moe_gate_up_kernel<<<grid, block, 0, get_current_stream()>>>(
        (const __nv_bfloat16*)hidden,
        (const uint8_t*)gate_up_blocks,
        (const uint8_t*)gate_up_scales,
        nullptr,  // no bias for debug
        (const int32_t*)expert_indices,
        (__nv_bfloat16*)gate_up_out,
        hidden_size,
        intermediate_size,
        num_blocks,
        num_experts,
        top_k
    );

    return cudaGetLastError() == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}
