// Model loader - loads HuggingFace models via Python subprocess
// Uses POSIX shared memory for zero-copy weight transfer

import { spawn } from "bun";
import { existsSync, readFileSync } from "fs";
import { join } from "path";
import { LlamaConfig, parseLlamaConfig, estimateMemory } from "./config";
import { getCudaBackend, CudaBackend } from "../backend/cuda/bindings";

export interface WeightInfo {
  shape: number[];
  dtype: string;
  offset: number;
  size: number;
}

export interface SharedMemoryInfo {
  shm_name: string;
  shm_size: number;
  tensors: Record<string, WeightInfo>;
}

export interface LoadedModel {
  config: LlamaConfig;
  weights: Map<string, { ptr: bigint; info: WeightInfo }>;
  localPath: string;
}

export class ModelLoader {
  private pythonPath: string;
  private cuda: CudaBackend;

  constructor(pythonPath: string = "python3") {
    this.pythonPath = pythonPath;
    this.cuda = getCudaBackend();
  }

  /**
   * Load a model from a local path or download from HuggingFace Hub.
   */
  async load(modelPath: string): Promise<LoadedModel> {
    // Check if it's a HuggingFace model ID or local path
    let localPath: string;
    if (existsSync(modelPath)) {
      localPath = modelPath;
    } else {
      // Download from HuggingFace Hub
      console.log(`Downloading model ${modelPath} from HuggingFace Hub...`);
      localPath = await this.downloadModel(modelPath);
    }

    // Load config.json
    const configPath = join(localPath, "config.json");
    if (!existsSync(configPath)) {
      throw new Error(`config.json not found at ${configPath}`);
    }
    const configJson = JSON.parse(readFileSync(configPath, "utf-8"));
    const config = parseLlamaConfig(configJson);

    console.log(`Model: ${config.modelType}`);
    const memInfo = estimateMemory(config);
    console.log(
      `  Parameters: ${memInfo.parametersB.toFixed(2)}B`,
      `| Memory: ${memInfo.memoryGB.toFixed(2)}GB`,
      `| KV/token: ${memInfo.kvCachePerTokenMB.toFixed(3)}MB`
    );

    // Load weights via Python subprocess
    console.log("Loading weights into shared memory...");
    const { info: shmInfo, proc } = await this.loadWeightsToShm(localPath);

    // Copy weights from shared memory to GPU
    console.log("Copying weights to GPU...");
    const weights = await this.copyWeightsToGpu(shmInfo, proc);

    // Signal Python to cleanup shared memory
    // (This is handled by the Python process exiting)

    return { config, weights, localPath };
  }

  private async downloadModel(modelId: string): Promise<string> {
    const scriptPath = join(import.meta.dir, "../../python/hf_download.py");

    const proc = spawn([this.pythonPath, scriptPath, "download", modelId], {
      stdout: "pipe",
      stderr: "inherit",
    });

    const output = await new Response(proc.stdout).text();
    const exitCode = await proc.exited;

    if (exitCode !== 0) {
      throw new Error(`Failed to download model: exit code ${exitCode}`);
    }

    const result = JSON.parse(output.trim());
    return result.path;
  }

  private async loadWeightsToShm(modelPath: string): Promise<{ info: SharedMemoryInfo; proc: ReturnType<typeof spawn> }> {
    const scriptPath = join(import.meta.dir, "../../python/weight_loader.py");

    const proc = spawn([this.pythonPath, scriptPath, modelPath, "--wait"], {
      stdout: "pipe",
      stderr: "inherit",
      stdin: "pipe",
    });

    // Read the first line (JSON output) from stdout
    const reader = proc.stdout.getReader();
    let output = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      output += new TextDecoder().decode(value);
      // Check if we have a complete JSON line
      if (output.includes("\n")) {
        break;
      }
    }
    reader.releaseLock();

    // Parse the JSON output (first line)
    const firstLine = output.trim().split('\n')[0];
    const info = JSON.parse(firstLine) as SharedMemoryInfo;

    return { info, proc };
  }

  private async copyWeightsToGpu(
    shmInfo: SharedMemoryInfo,
    proc: ReturnType<typeof spawn>
  ): Promise<Map<string, { ptr: bigint; info: WeightInfo }>> {
    const weights = new Map<string, { ptr: bigint; info: WeightInfo }>();

    // Open shared memory file using Bun's file API (memory-mapped)
    const shmPath = `/dev/shm${shmInfo.shm_name}`;
    const file = Bun.file(shmPath);

    console.log("Copying weights to GPU...");

    // Copy each weight separately to avoid loading entire file into memory
    for (const [name, info] of Object.entries(shmInfo.tensors)) {
      // Allocate GPU memory
      const gpuPtr = this.cuda.malloc(info.size);

      // Read only this weight's slice from the file
      const tensorBlob = file.slice(info.offset, info.offset + info.size);
      const tensorData = await tensorBlob.arrayBuffer();

      // Copy to GPU
      this.cuda.memcpyH2D(gpuPtr, tensorData, info.size);

      weights.set(name, { ptr: gpuPtr, info });

      // Log progress for large models
      if (Object.keys(shmInfo.tensors).length > 100) {
        const progress = ((weights.size / Object.keys(shmInfo.tensors).length) * 100).toFixed(1);
        if (weights.size % 10 === 0) {
          process.stdout.write(`\r  Progress: ${progress}% (${weights.size}/${Object.keys(shmInfo.tensors).length} tensors)`);
        }
      }
    }

    if (Object.keys(shmInfo.tensors).length > 100) {
      console.log("\n  All weights copied to GPU");
    }

    // Sync to ensure all copies complete
    this.cuda.synchronize();

    // Signal Python to cleanup shared memory
    proc.stdin.write("\n");
    proc.stdin.end();
    await proc.exited;

    return weights;
  }

  /**
   * Free all GPU memory for a loaded model.
   */
  freeModel(model: LoadedModel): void {
    for (const [, { ptr }] of model.weights) {
      this.cuda.free(ptr);
    }
  }
}

// Weight name mapping for LLaMA models
export const LLAMA_WEIGHT_NAMES = {
  embedTokens: "model.embed_tokens.weight",
  norm: "model.norm.weight",
  lmHead: "lm_head.weight",

  // Per-layer weights (replace {i} with layer index)
  inputLayernorm: "model.layers.{i}.input_layernorm.weight",
  postAttentionLayernorm: "model.layers.{i}.post_attention_layernorm.weight",
  qProj: "model.layers.{i}.self_attn.q_proj.weight",
  kProj: "model.layers.{i}.self_attn.k_proj.weight",
  vProj: "model.layers.{i}.self_attn.v_proj.weight",
  oProj: "model.layers.{i}.self_attn.o_proj.weight",
  qProjBias: "model.layers.{i}.self_attn.q_proj.bias",
  kProjBias: "model.layers.{i}.self_attn.k_proj.bias",
  vProjBias: "model.layers.{i}.self_attn.v_proj.bias",
  oProjBias: "model.layers.{i}.self_attn.o_proj.bias",

  // Dense MLP weights
  gateProj: "model.layers.{i}.mlp.gate_proj.weight",
  upProj: "model.layers.{i}.mlp.up_proj.weight",
  downProj: "model.layers.{i}.mlp.down_proj.weight",

  // MoE (Mixture of Experts) weights - MXFP4 quantized
  moeRouterWeight: "model.layers.{i}.mlp.router.weight",
  moeRouterBias: "model.layers.{i}.mlp.router.bias",
  moeGateUpBlocks: "model.layers.{i}.mlp.experts.gate_up_proj_blocks",
  moeGateUpScales: "model.layers.{i}.mlp.experts.gate_up_proj_scales",
  moeGateUpBias: "model.layers.{i}.mlp.experts.gate_up_proj_bias",
  moeDownBlocks: "model.layers.{i}.mlp.experts.down_proj_blocks",
  moeDownScales: "model.layers.{i}.mlp.experts.down_proj_scales",
  moeDownBias: "model.layers.{i}.mlp.experts.down_proj_bias",
};

export function getLayerWeightName(
  template: string,
  layerIdx: number
): string {
  return template.replace("{i}", layerIdx.toString());
}
