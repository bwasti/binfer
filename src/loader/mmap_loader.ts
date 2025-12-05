// Native MMAP-based weight loader for fast model loading
// Bypasses Python entirely - loads safetensors directly to GPU

import { existsSync, readdirSync, statSync } from "fs";
import { join } from "path";
import { parseSafetensorsHeader, dtypeToString, dtypeByteSize } from "./safetensors";
import { LlamaConfig, parseLlamaConfig, estimateMemory } from "../model/config";
import { getCudaBackend, CudaBackend } from "../backend/cuda/bindings";

export interface WeightInfo {
  shape: number[];
  dtype: string;
  offset: number;
  size: number;
}

export interface LoadedModel {
  config: LlamaConfig;
  weights: Map<string, { ptr: bigint; info: WeightInfo }>;
  localPath: string;
}

export interface LoadOptions {
  // Force a specific dtype for weights. If null, uses the model's native dtype.
  // "float16" will convert BF16 weights to FP16
  // "bfloat16" will keep BF16 weights as-is (requires BF16 kernels)
  dtype?: "float16" | "bfloat16" | null;
  // Status callback for progress updates
  status?: { update: (msg: string) => void };
}

/**
 * Native MMAP-based model loader.
 * Loads safetensors files directly using memory mapping for minimal copies.
 */
export class MmapModelLoader {
  private cuda: CudaBackend;
  private mmapFiles: Map<string, Uint8Array> = new Map();

  constructor() {
    this.cuda = getCudaBackend();
  }

  /**
   * Load a model from a local path or download from HuggingFace Hub.
   */
  async load(modelPath: string, options: LoadOptions = {}): Promise<LoadedModel> {
    const status = options.status;

    // Check if it's a HuggingFace model ID or local path
    let localPath: string;
    if (existsSync(modelPath)) {
      localPath = modelPath;
    } else {
      // Download from HuggingFace Hub using Python helper
      status?.update(`Downloading ${modelPath}...`);
      localPath = await this.downloadModel(modelPath, status);
    }

    // Load config.json
    const configPath = join(localPath, "config.json");
    if (!existsSync(configPath)) {
      throw new Error(`config.json not found at ${configPath}`);
    }
    const configFile = Bun.file(configPath);
    const configJson = await configFile.json();
    const config = parseLlamaConfig(configJson);

    // Find safetensors files
    const safetensorFiles = this.findSafetensorFiles(localPath);
    if (safetensorFiles.length === 0) {
      throw new Error(`No safetensors files found in ${localPath}`);
    }

    // Load weights using MMAP
    status?.update("Loading weights...");
    const weights = await this.loadWeightsWithMmap(safetensorFiles, config, options);

    // Cleanup mmap references
    this.cleanup();

    return { config, weights, localPath };
  }

  /**
   * Find all safetensor files in a directory.
   */
  private findSafetensorFiles(modelPath: string): string[] {
    const files: string[] = [];

    // Check for single model.safetensors
    const singleFile = join(modelPath, "model.safetensors");
    if (existsSync(singleFile)) {
      return [singleFile];
    }

    // Check for sharded files
    const allFiles = readdirSync(modelPath);
    for (const file of allFiles) {
      if (file.endsWith(".safetensors")) {
        files.push(join(modelPath, file));
      }
    }

    // Sort to ensure consistent ordering
    return files.sort();
  }

  /**
   * Load weights using file slicing for reliable access to large files.
   */
  private async loadWeightsWithMmap(
    safetensorFiles: string[],
    config: LlamaConfig,
    options: LoadOptions
  ): Promise<Map<string, { ptr: bigint; info: WeightInfo }>> {
    const weights = new Map<string, { ptr: bigint; info: WeightInfo }>();

    // Collect all tensors across files
    const allTensors: Array<{
      name: string;
      dtype: string;
      shape: number[];
      byteOffset: number;
      byteLength: number;
      file: ReturnType<typeof Bun.file>;
    }> = [];

    let totalBytes = 0;
    let detectedDtype: string | null = null;

    for (const filePath of safetensorFiles) {
      const file = Bun.file(filePath);

      // Read header (first 1MB should be enough)
      const headerSlice = await file.slice(0, Math.min(file.size, 1024 * 1024)).arrayBuffer();
      const parsed = parseSafetensorsHeader(new Uint8Array(headerSlice));

      for (const [name, info] of parsed.tensors) {
        allTensors.push({
          name,
          dtype: info.dtype,
          shape: info.shape,
          byteOffset: info.byteOffset,
          byteLength: info.byteLength,
          file,
        });
        totalBytes += info.byteLength;

        // Detect dtype from weight tensors (not embeddings which might differ)
        if (!detectedDtype && (info.dtype === "BF16" || info.dtype === "F16")) {
          detectedDtype = info.dtype;
        }
      }
    }

    // Update config dtype based on detected weights and user preference
    const targetDtype = options.dtype ?? null;

    if (detectedDtype === "BF16") {
      if (targetDtype === "float16") {
        config.dtype = "float16";
      } else {
        config.dtype = "bfloat16";
      }
    } else if (detectedDtype === "F16") {
      config.dtype = "float16";
    }

    // Copy tensors to GPU
    let copiedCount = 0;
    let copiedBytes = 0;
    const needsConversion = targetDtype === "float16";

    for (const tensor of allTensors) {
      // Read tensor data using file slicing (works reliably on large files)
      const tensorData = await tensor.file.slice(
        tensor.byteOffset,
        tensor.byteOffset + tensor.byteLength
      ).arrayBuffer();
      const byteLength = tensor.byteLength;
      const bytesPerElement = dtypeByteSize(tensor.dtype);
      const numElements = byteLength / bytesPerElement;
      let outputDtype = tensor.dtype;
      let gpuPtr: bigint;

      if (tensor.dtype === "BF16" && needsConversion) {
        // GPU-based conversion: upload BF16, convert to FP16 on GPU
        const bf16Ptr = this.cuda.malloc(byteLength);
        this.cuda.memcpyH2D(bf16Ptr, tensorData, byteLength);

        // Allocate FP16 output and convert
        gpuPtr = this.cuda.malloc(byteLength);
        this.cuda.convertBf16ToFp16(bf16Ptr, gpuPtr, numElements);

        // Free BF16 temporary
        this.cuda.free(bf16Ptr);
        outputDtype = "F16";
      } else {
        // Direct copy without conversion (works for all dtypes including U8)
        gpuPtr = this.cuda.malloc(byteLength);
        this.cuda.memcpyH2D(gpuPtr, tensorData, byteLength);
      }

      // Store weight info
      const dtypeStr = dtypeToString(outputDtype);
      weights.set(tensor.name, {
        ptr: gpuPtr,
        info: {
          shape: tensor.shape,
          dtype: dtypeStr,
          offset: 0, // Not used for GPU weights
          size: byteLength,
        },
      });

      copiedCount++;
      copiedBytes += byteLength;

      // Progress for large models
      if (allTensors.length > 50 && copiedCount % 10 === 0) {
        const progress = ((copiedCount / allTensors.length) * 100).toFixed(0);
        options.status?.update(`Loading weights... ${progress}%`);
      }
    }

    // Sync to ensure all copies complete
    this.cuda.synchronize();

    return weights;
  }

  /**
   * Download model from HuggingFace Hub using Python helper.
   */
  private async downloadModel(modelId: string, status?: { update: (msg: string) => void }): Promise<string> {
    const { spawn } = await import("bun");
    const scriptPath = join(import.meta.dir, "../../python/hf_download.py");

    const proc = spawn(["python3", scriptPath, "download", modelId], {
      stdout: "pipe",
      stderr: "pipe",
    });

    // Process stderr to extract download progress
    const stderrReader = proc.stderr.getReader();
    let stderrBuffer = "";

    const processStderr = async () => {
      while (true) {
        const { done, value } = await stderrReader.read();
        if (done) break;

        const text = new TextDecoder().decode(value);
        stderrBuffer += text;

        // Extract progress info from HuggingFace output
        // HF outputs lines like: "Downloading model.safetensors: 45%|████      | 1.2G/2.7G"
        const lines = stderrBuffer.split("\n");
        for (const line of lines) {
          // Look for percentage progress
          const percentMatch = line.match(/(\d+)%\|/);
          const fileMatch = line.match(/Downloading\s+(\S+)/);

          if (percentMatch && fileMatch) {
            status?.update(`Downloading ${fileMatch[1]}... ${percentMatch[1]}%`);
          } else if (line.includes("Fetching")) {
            status?.update("Fetching model info...");
          }
        }

        // Keep only the last partial line
        stderrBuffer = lines[lines.length - 1];
      }
    };

    // Start processing stderr in background
    processStderr().catch(() => {});

    const output = await new Response(proc.stdout).text();
    const exitCode = await proc.exited;

    if (exitCode !== 0) {
      throw new Error(`Failed to download model: exit code ${exitCode}`);
    }

    const result = JSON.parse(output.trim());
    return result.path;
  }

  /**
   * Cleanup mmap references.
   */
  private cleanup(): void {
    // Bun.mmap returns a Uint8Array backed by the file
    // Just clear our references and let GC handle it
    this.mmapFiles.clear();
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
