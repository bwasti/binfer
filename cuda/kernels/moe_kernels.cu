// Mixture of Experts (MoE) CUDA kernels
// Includes MXFP4 dequantization and expert routing

#include "../include/binfer.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cfloat>

// External function from gemm.cu to get current stream for CUDA graph capture
extern cudaStream_t get_current_stream();

// ============================================================================
// MXFP4 Dequantization
// ============================================================================
// MXFP4 format: 4-bit values packed in uint8, with per-block uint8 scales
// Block size is 32 elements (16 bytes with 2 FP4 values per byte)
// Scale is in E8M0 format (8-bit exponent only, bias 127)

// E8M0 to float conversion table (256 entries for uint8 scale)
__constant__ float e8m0_table[256];

// Initialize E8M0 lookup table on host
// E8M0: 8-bit unsigned exponent, bias=127
// value = 2^(exp - 127) for exp in [0, 254]
// exp=255 is reserved for NaN
void init_e8m0_table_host(float* table) {
    for (int i = 0; i < 255; i++) {
        // E8M0: pure power of 2 with bias 127
        table[i] = ldexpf(1.0f, i - 127);
    }
    table[255] = NAN;  // Reserved for NaN
}

__device__ __forceinline__ float e8m0_to_float(uint8_t val) {
    return e8m0_table[val];
}

// FP4 E2M1 to float lookup table (only 16 values for 4 bits)
// Format: [sign][exp1][exp0][mantissa]
// MXFP4 E2M1 encoding with exponent bias = 1:
//   subnormal (exp=0): value = mantissa * 0.5
//   normal (exp>0): value = (1 + mantissa*0.5) * 2^(exp-1)
__device__ __constant__ float fp4_table[16] = {
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

__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t val) {
    return fp4_table[val & 0xF];
}

// Dequantize MXFP4 blocks to BF16
// blocks: [num_experts, out_features, num_blocks, 16] as uint8 (packed 2 fp4 per byte)
// scales: [num_experts, out_features, num_blocks] as uint8 (E5M2)
// bias: [num_experts, out_features] as bf16
// output: [num_experts, out_features, in_features] as bf16
// Note: in_features = num_blocks * 32 (16 bytes * 2 values per byte)
__global__ void mxfp4_dequant_kernel(
    const uint8_t* __restrict__ blocks,
    const uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    int num_experts,
    int out_features,
    int num_blocks,
    int in_features
) {
    // Each thread handles one output row (one out_feature for one expert)
    const int expert_idx = blockIdx.x;
    const int out_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (out_idx >= out_features) return;

    // Base indices
    const int block_offset = (expert_idx * out_features + out_idx) * num_blocks;
    const int out_offset = (expert_idx * out_features + out_idx) * in_features;

    // Process each block of 32 elements (16 bytes * 2 values per byte)
    for (int b = 0; b < num_blocks; b++) {
        float scale = e8m0_to_float(scales[block_offset + b]);

        // Each block has 32 fp4 values packed into 16 bytes (2 per byte)
        const uint8_t* block_data = blocks + (block_offset + b) * 16;

        for (int i = 0; i < 16; i++) {
            uint8_t packed = block_data[i];

            // Low 4 bits
            float val0 = fp4_e2m1_to_float(packed & 0xF) * scale;
            // High 4 bits
            float val1 = fp4_e2m1_to_float((packed >> 4) & 0xF) * scale;

            int out_pos = b * 32 + i * 2;
            if (out_pos < in_features) {
                output[out_offset + out_pos] = __float2bfloat16(val0);
            }
            if (out_pos + 1 < in_features) {
                output[out_offset + out_pos + 1] = __float2bfloat16(val1);
            }
        }
    }
}

extern "C" BinferError binfer_mxfp4_dequant(
    const void* blocks,
    const void* scales,
    const void* bias,
    void* output,
    int num_experts,
    int out_features,
    int num_blocks,
    int in_features
) {
    const int threads = 256;
    const int blocks_y = (out_features + threads - 1) / threads;

    dim3 grid(num_experts, blocks_y);
    dim3 block(threads);

    mxfp4_dequant_kernel<<<grid, block, 0, get_current_stream()>>>(
        (const uint8_t*)blocks,
        (const uint8_t*)scales,
        (const __nv_bfloat16*)bias,
        (__nv_bfloat16*)output,
        num_experts,
        out_features,
        num_blocks,
        in_features
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// Single Expert Dequantization (for on-demand dequant of selected experts)
// ============================================================================

// Dequantize a single expert's weights - more parallelized version
// Each thread handles multiple output elements
__global__ void mxfp4_dequant_single_expert_kernel(
    const uint8_t* __restrict__ blocks,      // [num_experts, out_features, num_blocks, 16]
    const uint8_t* __restrict__ scales,      // [num_experts, out_features, num_blocks]
    __nv_bfloat16* __restrict__ output,      // [out_features, in_features]
    int expert_idx,
    int num_experts,
    int out_features,
    int num_blocks,
    int in_features
) {
    // Global thread index covers the output matrix
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = out_features * in_features;

    if (tid >= total_elements) return;

    const int out_idx = tid / in_features;
    const int in_idx = tid % in_features;

    // Which block and position within block
    const int block_idx = in_idx / 32;
    const int pos_in_block = in_idx % 32;
    const int byte_idx = pos_in_block / 2;
    const int is_high = pos_in_block % 2;

    // Calculate source indices for this expert
    const int block_offset = (expert_idx * out_features + out_idx) * num_blocks + block_idx;

    // Get scale and packed value
    float scale = e8m0_to_float(scales[block_offset]);
    uint8_t packed = blocks[block_offset * 16 + byte_idx];

    // Extract the right nibble
    uint8_t nibble = is_high ? ((packed >> 4) & 0xF) : (packed & 0xF);
    float val = fp4_table[nibble] * scale;

    output[tid] = __float2bfloat16(val);
}

extern "C" BinferError binfer_mxfp4_dequant_single_expert(
    const void* blocks,
    const void* scales,
    void* output,
    int expert_idx,
    int num_experts,
    int out_features,
    int num_blocks,
    int in_features
) {
    const int total_elements = out_features * in_features;
    const int threads = 256;
    const int blocks_needed = (total_elements + threads - 1) / threads;

    mxfp4_dequant_single_expert_kernel<<<blocks_needed, threads, 0, get_current_stream()>>>(
        (const uint8_t*)blocks,
        (const uint8_t*)scales,
        (__nv_bfloat16*)output,
        expert_idx,
        num_experts,
        out_features,
        num_blocks,
        in_features
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Initialize E8M0 table (call once at startup)
extern "C" BinferError binfer_init_mxfp4_tables() {
    float h_table[256];
    init_e8m0_table_host(h_table);
    cudaError_t err = cudaMemcpyToSymbol(e8m0_table, h_table, sizeof(h_table));
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// Expert Routing (Top-K Selection)
// ============================================================================

// Compute router logits and select top-k experts
// hidden: [batch, seq, hidden_size] as bf16
// router_weight: [num_experts, hidden_size] as bf16
// router_bias: [num_experts] as bf16
// expert_indices: [batch, seq, k] as int32 (output)
// expert_weights: [batch, seq, k] as bf16 (output, normalized)
__global__ void moe_router_topk_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ router_weight,
    const __nv_bfloat16* __restrict__ router_bias,
    int32_t* __restrict__ expert_indices,
    __nv_bfloat16* __restrict__ expert_weights,
    int batch_seq,
    int hidden_size,
    int num_experts,
    int top_k
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (token_idx >= batch_seq) return;

    const __nv_bfloat16* token_hidden = hidden + token_idx * hidden_size;

    // Shared memory for logits
    extern __shared__ float shared_logits[];

    // Each thread computes logits for a subset of experts
    for (int e = tid; e < num_experts; e += blockDim.x) {
        float logit = (router_bias != nullptr) ? __bfloat162float(router_bias[e]) : 0.0f;

        // Dot product with router weight
        for (int h = 0; h < hidden_size; h++) {
            logit += __bfloat162float(token_hidden[h]) *
                     __bfloat162float(router_weight[e * hidden_size + h]);
        }

        shared_logits[e] = logit;
    }
    __syncthreads();

    // Find top-k using thread 0
    if (tid == 0) {
        int32_t* out_indices = expert_indices + token_idx * top_k;
        __nv_bfloat16* out_weights = expert_weights + token_idx * top_k;

        // Simple O(k*n) selection for small k
        float selected_logits[8];  // Assume k <= 8

        for (int i = 0; i < top_k; i++) {
            float max_val = -FLT_MAX;
            int max_idx = -1;

            for (int e = 0; e < num_experts; e++) {
                // Skip if already selected
                bool already_selected = false;
                for (int j = 0; j < i; j++) {
                    if (out_indices[j] == e) {
                        already_selected = true;
                        break;
                    }
                }

                if (!already_selected && shared_logits[e] > max_val) {
                    max_val = shared_logits[e];
                    max_idx = e;
                }
            }

            out_indices[i] = max_idx;
            selected_logits[i] = max_val;
        }

        // Softmax over selected experts
        float max_logit = selected_logits[0];
        for (int i = 1; i < top_k; i++) {
            max_logit = fmaxf(max_logit, selected_logits[i]);
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < top_k; i++) {
            selected_logits[i] = expf(selected_logits[i] - max_logit);
            sum_exp += selected_logits[i];
        }

        for (int i = 0; i < top_k; i++) {
            out_weights[i] = __float2bfloat16(selected_logits[i] / sum_exp);
        }
    }
}

extern "C" BinferError binfer_moe_router_topk(
    const void* hidden,
    const void* router_weight,
    const void* router_bias,
    int32_t* expert_indices,
    void* expert_weights,
    int batch_size,
    int seq_len,
    int hidden_size,
    int num_experts,
    int top_k
) {
    const int batch_seq = batch_size * seq_len;
    const int threads = 256;
    const size_t shared_size = num_experts * sizeof(float);

    moe_router_topk_kernel<<<batch_seq, threads, shared_size, get_current_stream()>>>(
        (const __nv_bfloat16*)hidden,
        (const __nv_bfloat16*)router_weight,
        (const __nv_bfloat16*)router_bias,
        expert_indices,
        (__nv_bfloat16*)expert_weights,
        batch_seq,
        hidden_size,
        num_experts,
        top_k
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// MoE Expert Execution with MXFP4 weights
// ============================================================================

// Fused MXFP4 dequant + GEMV for a single expert
// This is more efficient than dequantizing all weights first
__global__ void moe_expert_gemv_mxfp4_kernel(
    const __nv_bfloat16* __restrict__ input,     // [batch_seq, hidden_size]
    const uint8_t* __restrict__ gate_up_blocks,  // [num_experts, intermediate*2, num_blocks, 16]
    const uint8_t* __restrict__ gate_up_scales,  // [num_experts, intermediate*2, num_blocks]
    const __nv_bfloat16* __restrict__ gate_up_bias, // [num_experts, intermediate*2]
    const uint8_t* __restrict__ down_blocks,     // [num_experts, hidden, num_blocks, 16]
    const uint8_t* __restrict__ down_scales,     // [num_experts, hidden, num_blocks]
    const __nv_bfloat16* __restrict__ down_bias, // [num_experts, hidden]
    const int32_t* __restrict__ expert_indices,  // [batch_seq, top_k]
    const __nv_bfloat16* __restrict__ expert_weights, // [batch_seq, top_k]
    __nv_bfloat16* __restrict__ output,          // [batch_seq, hidden_size]
    int batch_seq,
    int hidden_size,
    int intermediate_size,
    int num_blocks,
    int top_k
) {
    const int token_idx = blockIdx.x;
    const int out_dim = threadIdx.x + blockIdx.y * blockDim.x;

    if (token_idx >= batch_seq || out_dim >= hidden_size) return;

    const __nv_bfloat16* token_input = input + token_idx * hidden_size;
    const int32_t* token_experts = expert_indices + token_idx * top_k;
    const __nv_bfloat16* token_weights = expert_weights + token_idx * top_k;

    float out_accum = 0.0f;

    // Process each selected expert
    for (int k = 0; k < top_k; k++) {
        int expert_idx = token_experts[k];
        float expert_weight = __bfloat162float(token_weights[k]);

        if (expert_idx < 0) continue;

        // Compute gate and up projections (fused)
        // gate_up has shape [intermediate*2], first half is gate, second is up
        float gate_vals[32];  // Assume intermediate <= 32 per thread block iteration
        float up_vals[32];

        // This is simplified - real implementation would be more optimized
        // For now, accumulate directly

        // For output dimension out_dim, we need down_proj row out_dim
        // down_proj: [expert, hidden, in_features] where in_features = intermediate

        // First compute intermediate = SiLU(gate) * up
        // Then compute output = down @ intermediate

        // This is a placeholder - real implementation needs proper tiling
        float down_accum = __bfloat162float(down_bias[expert_idx * hidden_size + out_dim]);

        // We need to compute the full intermediate first, then dot with down row
        // This kernel is simplified and not optimal

        out_accum += expert_weight * down_accum;
    }

    output[token_idx * hidden_size + out_dim] = __float2bfloat16(out_accum);
}

// Simpler approach: gather inputs, run through experts, scatter outputs
// Step 1: Gather tokens by expert
__global__ void moe_gather_by_expert_kernel(
    const __nv_bfloat16* __restrict__ input,     // [batch_seq, hidden]
    const int32_t* __restrict__ expert_indices,  // [batch_seq, top_k]
    __nv_bfloat16* __restrict__ gathered,        // [num_experts, max_tokens, hidden]
    int32_t* __restrict__ token_counts,          // [num_experts]
    int32_t* __restrict__ token_mapping,         // [num_experts, max_tokens] -> original token idx
    int batch_seq,
    int hidden_size,
    int num_experts,
    int top_k,
    int max_tokens_per_expert
) {
    // This is done on CPU for simplicity in initial implementation
    // GPU version would use atomics
}

// Step 2: Run batched GEMM per expert (using cuBLAS or custom kernel)
// Step 3: Scatter and weight outputs

extern "C" BinferError binfer_moe_forward(
    const void* input,
    const void* gate_up_blocks,
    const void* gate_up_scales,
    const void* gate_up_bias,
    const void* down_blocks,
    const void* down_scales,
    const void* down_bias,
    const void* router_weight,
    const void* router_bias,
    void* output,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size,
    int num_experts,
    int top_k
) {
    // Full MoE forward implementation
    // For now, return success - actual implementation in TypeScript layer
    return BINFER_SUCCESS;
}

// ============================================================================
// SwiGLU activation for MoE (fused gate/up)
// ============================================================================

// gate_up: [batch, intermediate * 2] -> output: [batch, intermediate]
// First half is gate, second half is up
__global__ void moe_swiglu_kernel(
    const __nv_bfloat16* __restrict__ gate_up,
    __nv_bfloat16* __restrict__ output,
    int batch,
    int intermediate_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = idx / intermediate_size;
    const int dim_idx = idx % intermediate_size;

    if (batch_idx >= batch) return;

    // Standard order: [gate | up]
    float gate = __bfloat162float(gate_up[batch_idx * intermediate_size * 2 + dim_idx]);
    float up = __bfloat162float(gate_up[batch_idx * intermediate_size * 2 + intermediate_size + dim_idx]);

    // SiLU(gate) * up
    float silu_gate = gate / (1.0f + expf(-gate));
    output[batch_idx * intermediate_size + dim_idx] = __float2bfloat16(silu_gate * up);
}

extern "C" BinferError binfer_moe_swiglu(
    const void* gate_up,
    void* output,
    int batch,
    int intermediate_size
) {
    const int total = batch * intermediate_size;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    moe_swiglu_kernel<<<blocks, threads, 0, get_current_stream()>>>(
        (const __nv_bfloat16*)gate_up,
        (__nv_bfloat16*)output,
        batch,
        intermediate_size
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// GPT-OSS Activation (concatenated gate/up with custom activation)
// ============================================================================
// GPT-OSS uses a different activation than standard SwiGLU:
// - gate_up is CONCATENATED: [gate_0..gate_n, up_0..up_n]
// - gate is clamped to max=7.0, up is clamped to [-7.0, 7.0]
// - glu = gate * sigmoid(gate * alpha) where alpha=1.702
// - output = (up + 1) * glu

__global__ void gpt_oss_activation_kernel(
    const __nv_bfloat16* __restrict__ gate_up,
    __nv_bfloat16* __restrict__ output,
    int batch,
    int intermediate_size,
    float alpha,
    float limit
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = idx / intermediate_size;
    const int dim_idx = idx % intermediate_size;

    if (batch_idx >= batch) return;

    // Interleaved: gate = gate_up[::2] (even indices), up = gate_up[1::2] (odd indices)
    // Reference: gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    float gate = __bfloat162float(gate_up[batch_idx * intermediate_size * 2 + dim_idx * 2]);
    float up = __bfloat162float(gate_up[batch_idx * intermediate_size * 2 + dim_idx * 2 + 1]);

    // Clamp
    gate = fminf(gate, limit);
    up = fmaxf(fminf(up, limit), -limit);

    // glu = gate * sigmoid(gate * alpha)
    float glu = gate / (1.0f + expf(-gate * alpha));

    // output = (up + 1) * glu
    output[batch_idx * intermediate_size + dim_idx] = __float2bfloat16((up + 1.0f) * glu);
}

extern "C" BinferError binfer_gpt_oss_activation(
    const void* gate_up,
    void* output,
    int batch,
    int intermediate_size,
    float alpha,
    float limit
) {
    const int total = batch * intermediate_size;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    gpt_oss_activation_kernel<<<blocks, threads, 0, get_current_stream()>>>(
        (const __nv_bfloat16*)gate_up,
        (__nv_bfloat16*)output,
        batch,
        intermediate_size,
        alpha,
        limit
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// ============================================================================
// Scale and Add: output = output + scale * input
// ============================================================================

__global__ void scale_add_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    float scale,
    int numel
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float in_val = __bfloat162float(input[idx]);
    float out_val = __bfloat162float(output[idx]);
    output[idx] = __float2bfloat16(out_val + scale * in_val);
}

extern "C" BinferError binfer_scale_add_bf16(
    const void* input,
    void* output,
    float scale,
    int numel
) {
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    scale_add_bf16_kernel<<<blocks, threads, 0, get_current_stream()>>>(
        (const __nv_bfloat16*)input,
        (__nv_bfloat16*)output,
        scale,
        numel
    );

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}

// Memory set (wrapper around cudaMemset)
extern "C" BinferError binfer_memset(void* ptr, int value, size_t size) {
    cudaError_t err = cudaMemset(ptr, value, size);
    return err == cudaSuccess ? BINFER_SUCCESS : BINFER_ERROR_CUDA;
}
