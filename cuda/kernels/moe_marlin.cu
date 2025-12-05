// Marlin-style MoE kernel with fused MXFP4 dequantization and work-stealing
//
// Key optimizations from Marlin:
// 1. Single kernel launch for ALL experts per projection (work-stealing across tiles)
// 2. Fused MXFP4 (FP4 E2M1 + E8M0 scales) dequantization
// 3. Reduces kernel launch overhead from O(num_experts) to O(1)
//
// Full MoE layer: gate_up -> activation -> down

#include "../include/binfer.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <vector>

extern cudaStream_t get_current_stream();

namespace marlin_moe {

// ============================================================================
// Configuration
// ============================================================================
static constexpr int THREADS = 256;       // 8 warps per block
static constexpr int WARP_SIZE = 32;

// Tile sizes for GEMM
static constexpr int TILE_M = 16;         // M tile (tokens per tile)
static constexpr int TILE_N = 64;         // N tile (output features)
static constexpr int TILE_K = 64;         // K tile (2 MXFP4 blocks of 32)

static constexpr int MAX_EXPERTS = 128;

// ============================================================================
// FP4 lookup table in constant memory
// ============================================================================
__device__ __constant__ float c_fp4_table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// E8M0 decode
__device__ __forceinline__ float e8m0_to_float(uint8_t val) {
    union { float f; uint32_t u; } c;
    c.u = (uint32_t)val << 23;
    return c.f;
}

// ============================================================================
// Precomputed work distribution data (in constant memory for fast access)
// ============================================================================
struct ExpertWorkInfo {
    int token_start;     // Start index in sorted_token_ids
    int token_count;     // Number of tokens for this expert
    int m_tiles;         // Number of M tiles = ceil(token_count / TILE_M)
    int work_offset;     // Cumulative work items before this expert
};

__device__ __constant__ ExpertWorkInfo c_expert_info[MAX_EXPERTS];
__device__ __constant__ int c_num_experts;
__device__ __constant__ int c_n_tiles;  // Total N tiles per expert
__device__ __constant__ int c_total_work_items;

// ============================================================================
// Work-stealing MoE GEMM kernel for gate_up projection
// Single kernel launch processes all tiles across all experts
// ============================================================================
__global__ void __launch_bounds__(THREADS, 2)
gate_up_kernel(
    // Input
    const nv_bfloat16* __restrict__ hidden,          // [numTokens, hiddenSize]
    const uint8_t* __restrict__ weight_blocks,       // [numExperts, outFeatures, numBlocks, 16]
    const uint8_t* __restrict__ weight_scales,       // [numExperts, outFeatures, numBlocks]
    const nv_bfloat16* __restrict__ bias,            // [numExperts, outFeatures]
    const int32_t* __restrict__ sorted_token_ids,    // [totalSlots]
    // Output
    nv_bfloat16* __restrict__ output,                // [totalSlots, outFeatures]
    // Dimensions
    int hiddenSize,
    int outFeatures,
    int numBlocks,        // hiddenSize / 32
    int topK,
    bool hasBias
) {
    const int work_idx = blockIdx.x;
    if (work_idx >= c_total_work_items) return;

    // Binary search for expert
    int expert_id = 0;
    int lo = 0, hi = c_num_experts - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (c_expert_info[mid].work_offset <= work_idx) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    expert_id = lo;

    const ExpertWorkInfo& info = c_expert_info[expert_id];

    int local_work = work_idx - info.work_offset;
    int m_tile = local_work / c_n_tiles;
    int n_tile = local_work % c_n_tiles;

    if (m_tile >= info.m_tiles) return;

    const int m_start = m_tile * TILE_M;
    const int n_start = n_tile * TILE_N;

    __shared__ float sh_hidden[TILE_M][TILE_K + 4];
    __shared__ float sh_weights[TILE_K][TILE_N + 4];
    __shared__ float sh_accum[TILE_M][TILE_N];

    const int tid = threadIdx.x;

    // Initialize accumulators
    #pragma unroll
    for (int i = tid; i < TILE_M * TILE_N; i += THREADS) {
        sh_accum[i / TILE_N][i % TILE_N] = 0.0f;
    }
    __syncthreads();

    const int k_tiles = (numBlocks * 32 + TILE_K - 1) / TILE_K;

    for (int k_tile = 0; k_tile < k_tiles; k_tile++) {
        const int k_start = k_tile * TILE_K;

        // Load hidden states
        #pragma unroll
        for (int i = tid; i < TILE_M * TILE_K; i += THREADS) {
            int m = i / TILE_K;
            int k = i % TILE_K;
            int token_idx = m_start + m;
            int k_idx = k_start + k;

            float val = 0.0f;
            if (token_idx < info.token_count && k_idx < hiddenSize) {
                int sorted_idx = info.token_start + token_idx;
                int source_idx = sorted_token_ids[sorted_idx];
                int orig_token = source_idx / topK;
                val = __bfloat162float(hidden[orig_token * hiddenSize + k_idx]);
            }
            sh_hidden[m][k] = val;
        }

        // Load and dequantize weights
        #pragma unroll
        for (int i = tid; i < TILE_N * 2; i += THREADS) {
            int n = i % TILE_N;
            int block_offset = i / TILE_N;

            int global_n = n_start + n;
            int block_idx = (k_start / 32) + block_offset;

            if (global_n < outFeatures && block_idx < numBlocks) {
                int weight_offset = (expert_id * outFeatures + global_n) * numBlocks + block_idx;
                float scale = e8m0_to_float(weight_scales[weight_offset]);
                const uint8_t* block_ptr = weight_blocks + weight_offset * 16;

                #pragma unroll
                for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
                    uint8_t packed = block_ptr[byte_idx];
                    int k0 = block_offset * 32 + byte_idx * 2;
                    int k1 = k0 + 1;
                    sh_weights[k0][n] = c_fp4_table[packed & 0xF] * scale;
                    sh_weights[k1][n] = c_fp4_table[(packed >> 4) & 0xF] * scale;
                }
            } else {
                int k_base = block_offset * 32;
                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    sh_weights[k_base + k][n] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Matrix multiply
        #pragma unroll
        for (int i = tid; i < TILE_M * TILE_N; i += THREADS) {
            int m = i / TILE_N;
            int n = i % TILE_N;
            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < TILE_K; k++) {
                sum += sh_hidden[m][k] * sh_weights[k][n];
            }
            sh_accum[m][n] += sum;
        }
        __syncthreads();
    }

    // Write output (to sorted position)
    #pragma unroll
    for (int i = tid; i < TILE_M * TILE_N; i += THREADS) {
        int m = i / TILE_N;
        int n = i % TILE_N;
        int token_idx = m_start + m;
        int global_n = n_start + n;

        if (token_idx < info.token_count && global_n < outFeatures) {
            float result = sh_accum[m][n];
            if (hasBias) {
                result += __bfloat162float(bias[expert_id * outFeatures + global_n]);
            }
            int sorted_idx = info.token_start + token_idx;
            output[sorted_idx * outFeatures + global_n] = __float2bfloat16(result);
        }
    }
}

// ============================================================================
// Activation kernel (SiLU-gated)
// ============================================================================
__global__ void activation_kernel(
    const nv_bfloat16* __restrict__ gate_up,  // [totalSlots, intermediateSize * 2]
    nv_bfloat16* __restrict__ activated,      // [totalSlots, intermediateSize]
    int totalSlots,
    int intermediateSize,
    float alpha,
    float limit
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int slot = idx / intermediateSize;
    int feat = idx % intermediateSize;

    if (slot >= totalSlots || feat >= intermediateSize) return;

    // Interleaved layout: gate at even positions, up at odd
    float gate = __bfloat162float(gate_up[slot * intermediateSize * 2 + feat * 2]);
    float up = __bfloat162float(gate_up[slot * intermediateSize * 2 + feat * 2 + 1]);

    // SiLU(gate) * up
    float x = gate * alpha;
    x = fminf(fmaxf(x, -limit), limit);
    float sigmoid = 1.0f / (1.0f + expf(-x));
    float result = gate * sigmoid * up;

    activated[slot * intermediateSize + feat] = __float2bfloat16(result);
}

// ============================================================================
// Down projection kernel with work-stealing + scatter-accumulate
// ============================================================================
__global__ void __launch_bounds__(THREADS, 2)
down_kernel(
    // Input
    const nv_bfloat16* __restrict__ activated,       // [totalSlots, intermediateSize]
    const uint8_t* __restrict__ weight_blocks,       // [numExperts, hiddenSize, numBlocks, 16]
    const uint8_t* __restrict__ weight_scales,       // [numExperts, hiddenSize, numBlocks]
    const nv_bfloat16* __restrict__ bias,            // [numExperts, hiddenSize]
    const int32_t* __restrict__ sorted_token_ids,    // [totalSlots]
    const nv_bfloat16* __restrict__ topk_weights,    // [numTokens * topK] BF16
    // Output
    float* __restrict__ output_f32,                  // [numTokens, hiddenSize]
    // Dimensions
    int hiddenSize,
    int intermediateSize,
    int numBlocks,        // intermediateSize / 32
    int topK,
    bool hasBias
) {
    const int work_idx = blockIdx.x;
    if (work_idx >= c_total_work_items) return;

    // Binary search for expert
    int expert_id = 0;
    int lo = 0, hi = c_num_experts - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (c_expert_info[mid].work_offset <= work_idx) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    expert_id = lo;

    const ExpertWorkInfo& info = c_expert_info[expert_id];

    int local_work = work_idx - info.work_offset;
    int m_tile = local_work / c_n_tiles;
    int n_tile = local_work % c_n_tiles;

    if (m_tile >= info.m_tiles) return;

    const int m_start = m_tile * TILE_M;
    const int n_start = n_tile * TILE_N;  // n is hidden dimension

    __shared__ float sh_input[TILE_M][TILE_K + 4];
    __shared__ float sh_weights[TILE_K][TILE_N + 4];
    __shared__ float sh_accum[TILE_M][TILE_N];

    const int tid = threadIdx.x;

    // Initialize accumulators
    #pragma unroll
    for (int i = tid; i < TILE_M * TILE_N; i += THREADS) {
        sh_accum[i / TILE_N][i % TILE_N] = 0.0f;
    }
    __syncthreads();

    const int k_tiles = (numBlocks * 32 + TILE_K - 1) / TILE_K;

    for (int k_tile = 0; k_tile < k_tiles; k_tile++) {
        const int k_start = k_tile * TILE_K;

        // Load activated input
        #pragma unroll
        for (int i = tid; i < TILE_M * TILE_K; i += THREADS) {
            int m = i / TILE_K;
            int k = i % TILE_K;
            int token_idx = m_start + m;
            int k_idx = k_start + k;

            float val = 0.0f;
            if (token_idx < info.token_count && k_idx < intermediateSize) {
                int sorted_idx = info.token_start + token_idx;
                val = __bfloat162float(activated[sorted_idx * intermediateSize + k_idx]);
            }
            sh_input[m][k] = val;
        }

        // Load and dequantize weights
        #pragma unroll
        for (int i = tid; i < TILE_N * 2; i += THREADS) {
            int n = i % TILE_N;
            int block_offset = i / TILE_N;

            int global_n = n_start + n;
            int block_idx = (k_start / 32) + block_offset;

            if (global_n < hiddenSize && block_idx < numBlocks) {
                int weight_offset = (expert_id * hiddenSize + global_n) * numBlocks + block_idx;
                float scale = e8m0_to_float(weight_scales[weight_offset]);
                const uint8_t* block_ptr = weight_blocks + weight_offset * 16;

                #pragma unroll
                for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
                    uint8_t packed = block_ptr[byte_idx];
                    int k0 = block_offset * 32 + byte_idx * 2;
                    int k1 = k0 + 1;
                    sh_weights[k0][n] = c_fp4_table[packed & 0xF] * scale;
                    sh_weights[k1][n] = c_fp4_table[(packed >> 4) & 0xF] * scale;
                }
            } else {
                int k_base = block_offset * 32;
                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    sh_weights[k_base + k][n] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Matrix multiply
        #pragma unroll
        for (int i = tid; i < TILE_M * TILE_N; i += THREADS) {
            int m = i / TILE_N;
            int n = i % TILE_N;
            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < TILE_K; k++) {
                sum += sh_input[m][k] * sh_weights[k][n];
            }
            sh_accum[m][n] += sum;
        }
        __syncthreads();
    }

    // Apply bias, routing weight, and scatter-accumulate to output
    #pragma unroll
    for (int i = tid; i < TILE_M * TILE_N; i += THREADS) {
        int m = i / TILE_N;
        int n = i % TILE_N;
        int token_idx = m_start + m;
        int global_n = n_start + n;

        if (token_idx < info.token_count && global_n < hiddenSize) {
            float result = sh_accum[m][n];
            if (hasBias) {
                result += __bfloat162float(bias[expert_id * hiddenSize + global_n]);
            }

            int sorted_idx = info.token_start + token_idx;
            int source_idx = sorted_token_ids[sorted_idx];
            int orig_token = source_idx / topK;
            float weight = __bfloat162float(topk_weights[source_idx]);

            atomicAdd(&output_f32[orig_token * hiddenSize + global_n], result * weight);
        }
    }
}

// ============================================================================
// Convert F32 output to BF16
// ============================================================================
__global__ void f32_to_bf16_kernel(
    const float* __restrict__ input,
    nv_bfloat16* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2bfloat16(input[idx]);
    }
}

} // namespace marlin_moe

// ============================================================================
// Sorting kernels
// ============================================================================

__global__ void count_expert_tokens_kernel(
    const int32_t* __restrict__ expert_indices,
    int32_t* __restrict__ expert_counts,
    int numTokensTopK,
    int numExperts,
    int expertsPerRank,
    int rank
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTokensTopK) return;

    int global_expert = expert_indices[tid];
    int expert_start = rank * expertsPerRank;
    int expert_end = expert_start + expertsPerRank;

    if (global_expert >= expert_start && global_expert < expert_end) {
        int local_expert = global_expert - expert_start;
        atomicAdd(&expert_counts[local_expert], 1);
    }
}

__global__ void compute_expert_offsets_kernel(
    const int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ expert_offsets,
    int numExperts
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int offset = 0;
        for (int i = 0; i < numExperts; i++) {
            expert_offsets[i] = offset;
            offset += expert_counts[i];
        }
        expert_offsets[numExperts] = offset;
    }
}

__global__ void sort_tokens_kernel(
    const int32_t* __restrict__ expert_indices,
    const int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ expert_slot_counts,
    int32_t* __restrict__ sorted_token_ids,
    int numTokensTopK,
    int numExperts,
    int expertsPerRank,
    int rank
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTokensTopK) return;

    int global_expert = expert_indices[tid];
    int expert_start = rank * expertsPerRank;
    int expert_end = expert_start + expertsPerRank;

    if (global_expert >= expert_start && global_expert < expert_end) {
        int local_expert = global_expert - expert_start;
        int slot = atomicAdd(&expert_slot_counts[local_expert], 1);
        int offset = expert_offsets[local_expert];
        sorted_token_ids[offset + slot] = tid;
    }
}

// ============================================================================
// Host-side preparation and kernel launch
// ============================================================================

extern "C" size_t binfer_moe_marlin_scratch_size_ep(
    int numTokens,
    int numExperts,
    int hiddenSize,
    int intermediateSize,
    int topK
) {
    size_t size = 0;
    int numTokensTopK = numTokens * topK;
    int gateUpSize = intermediateSize * 2;

    size += numTokensTopK * sizeof(int32_t);      // sorted_token_ids
    size += (numExperts + 1) * sizeof(int32_t);   // expert_offsets
    size += numExperts * sizeof(int32_t);         // expert_counts
    size += numExperts * sizeof(int32_t);         // expert_slot_counts

    // Intermediate buffers
    size += (size_t)numTokensTopK * gateUpSize * sizeof(nv_bfloat16);    // gate_up output
    size += (size_t)numTokensTopK * intermediateSize * sizeof(nv_bfloat16); // activated
    size += (size_t)numTokens * hiddenSize * sizeof(float);  // output_f32

    size = (size + 255) & ~255;
    return size;
}

extern "C" void binfer_moe_marlin_forward_ep(
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
    cudaStream_t stream = get_current_stream();

    const int numTokensTopK = numTokens * topK;
    const int numBlocksIn = hiddenSize / 32;
    const int numBlocksInter = intermediateSize / 32;
    const int gateUpSize = intermediateSize * 2;
    const int n_tiles_gate_up = (gateUpSize + marlin_moe::TILE_N - 1) / marlin_moe::TILE_N;
    const int n_tiles_down = (hiddenSize + marlin_moe::TILE_N - 1) / marlin_moe::TILE_N;

    // Parse scratch buffer
    char* scratch_ptr = (char*)scratch;
    int32_t* sorted_token_ids = (int32_t*)scratch_ptr;
    scratch_ptr += numTokensTopK * sizeof(int32_t);
    int32_t* expert_offsets = (int32_t*)scratch_ptr;
    scratch_ptr += (expertsPerRank + 1) * sizeof(int32_t);
    int32_t* expert_counts = (int32_t*)scratch_ptr;
    scratch_ptr += expertsPerRank * sizeof(int32_t);
    int32_t* expert_slot_counts = (int32_t*)scratch_ptr;
    scratch_ptr += expertsPerRank * sizeof(int32_t);
    nv_bfloat16* gate_up_out = (nv_bfloat16*)scratch_ptr;
    scratch_ptr += (size_t)numTokensTopK * gateUpSize * sizeof(nv_bfloat16);
    nv_bfloat16* activated = (nv_bfloat16*)scratch_ptr;
    scratch_ptr += (size_t)numTokensTopK * intermediateSize * sizeof(nv_bfloat16);
    float* output_f32 = (float*)scratch_ptr;

    // Step 1: Count and sort tokens by expert
    cudaMemsetAsync(expert_counts, 0, expertsPerRank * sizeof(int32_t), stream);
    cudaMemsetAsync(expert_slot_counts, 0, expertsPerRank * sizeof(int32_t), stream);
    cudaMemsetAsync(output_f32, 0, numTokens * hiddenSize * sizeof(float), stream);

    {
        int threads = 256;
        int blocks = (numTokensTopK + threads - 1) / threads;
        count_expert_tokens_kernel<<<blocks, threads, 0, stream>>>(
            (const int32_t*)expert_indices, expert_counts, numTokensTopK,
            expertsPerRank * worldSize, expertsPerRank, rank
        );
    }

    compute_expert_offsets_kernel<<<1, 1, 0, stream>>>(
        expert_counts, expert_offsets, expertsPerRank
    );

    {
        int threads = 256;
        int blocks = (numTokensTopK + threads - 1) / threads;
        sort_tokens_kernel<<<blocks, threads, 0, stream>>>(
            (const int32_t*)expert_indices, expert_offsets, expert_slot_counts,
            sorted_token_ids, numTokensTopK, expertsPerRank * worldSize,
            expertsPerRank, rank
        );
    }

    // Step 2: Prepare work distribution (need sync to read counts)
    std::vector<marlin_moe::ExpertWorkInfo> h_expert_info(expertsPerRank);
    std::vector<int32_t> h_expert_offsets(expertsPerRank + 1);
    std::vector<int32_t> h_expert_counts(expertsPerRank);

    cudaMemcpyAsync(h_expert_offsets.data(), expert_offsets,
                    (expertsPerRank + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_expert_counts.data(), expert_counts,
                    expertsPerRank * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Calculate work items for gate_up (larger N dimension)
    int total_work_gate_up = 0;
    for (int i = 0; i < expertsPerRank; i++) {
        int token_count = h_expert_counts[i];
        int m_tiles = (token_count + marlin_moe::TILE_M - 1) / marlin_moe::TILE_M;

        h_expert_info[i].token_start = h_expert_offsets[i];
        h_expert_info[i].token_count = token_count;
        h_expert_info[i].m_tiles = m_tiles;
        h_expert_info[i].work_offset = total_work_gate_up;

        total_work_gate_up += m_tiles * n_tiles_gate_up;
    }

    // Step 3: Gate-up projection
    if (total_work_gate_up > 0) {
        cudaMemcpyToSymbolAsync(marlin_moe::c_expert_info, h_expert_info.data(),
                                expertsPerRank * sizeof(marlin_moe::ExpertWorkInfo),
                                0, cudaMemcpyHostToDevice, stream);
        cudaMemcpyToSymbolAsync(marlin_moe::c_num_experts, &expertsPerRank,
                                sizeof(int), 0, cudaMemcpyHostToDevice, stream);
        cudaMemcpyToSymbolAsync(marlin_moe::c_n_tiles, &n_tiles_gate_up,
                                sizeof(int), 0, cudaMemcpyHostToDevice, stream);
        cudaMemcpyToSymbolAsync(marlin_moe::c_total_work_items, &total_work_gate_up,
                                sizeof(int), 0, cudaMemcpyHostToDevice, stream);

        marlin_moe::gate_up_kernel<<<total_work_gate_up, marlin_moe::THREADS, 0, stream>>>(
            (const nv_bfloat16*)hidden,
            (const uint8_t*)gate_up_blocks,
            (const uint8_t*)gate_up_scales,
            (const nv_bfloat16*)gate_up_bias,
            sorted_token_ids,
            gate_up_out,
            hiddenSize, gateUpSize, numBlocksIn, topK,
            gate_up_bias != nullptr
        );
    }

    // Step 4: Activation
    int totalSlots = h_expert_offsets[expertsPerRank];
    if (totalSlots > 0) {
        int threads = 256;
        int blocks = (totalSlots * intermediateSize + threads - 1) / threads;
        marlin_moe::activation_kernel<<<blocks, threads, 0, stream>>>(
            gate_up_out, activated, totalSlots, intermediateSize, 1.702f, 88.0f
        );
    }

    // Step 5: Down projection with work-stealing
    int total_work_down = 0;
    for (int i = 0; i < expertsPerRank; i++) {
        h_expert_info[i].work_offset = total_work_down;
        total_work_down += h_expert_info[i].m_tiles * n_tiles_down;
    }

    if (total_work_down > 0) {
        cudaMemcpyToSymbolAsync(marlin_moe::c_expert_info, h_expert_info.data(),
                                expertsPerRank * sizeof(marlin_moe::ExpertWorkInfo),
                                0, cudaMemcpyHostToDevice, stream);
        cudaMemcpyToSymbolAsync(marlin_moe::c_n_tiles, &n_tiles_down,
                                sizeof(int), 0, cudaMemcpyHostToDevice, stream);
        cudaMemcpyToSymbolAsync(marlin_moe::c_total_work_items, &total_work_down,
                                sizeof(int), 0, cudaMemcpyHostToDevice, stream);

        marlin_moe::down_kernel<<<total_work_down, marlin_moe::THREADS, 0, stream>>>(
            activated,
            (const uint8_t*)down_blocks,
            (const uint8_t*)down_scales,
            (const nv_bfloat16*)down_bias,
            sorted_token_ids,
            (const nv_bfloat16*)expert_weights,
            output_f32,
            hiddenSize, intermediateSize, numBlocksInter, topK,
            down_bias != nullptr
        );
    }

    // Step 6: Convert F32 to BF16
    {
        int threads = 256;
        int blocks = (numTokens * hiddenSize + threads - 1) / threads;
        marlin_moe::f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
            output_f32, (nv_bfloat16*)output, numTokens * hiddenSize
        );
    }
}
