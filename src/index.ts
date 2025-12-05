#!/usr/bin/env bun
// Binfer: Bun-based HuggingFace Inference Framework
// Unified entry point with tensor parallelism support

// ============================================================================
// Early CLI handling - runs before any heavy imports
// ============================================================================

const VALID_FLAGS = new Set([
  "--tp", "--max-tokens", "--dtype", "-p", "--prompt", "--chat", "--bench",
  "--profile", "--trace", "--no-cache", "-q", "--quiet", "--temperature",
  "--temp", "--top-k", "--top-p", "-i", "--input", "-o", "--output",
  "--help", "-h", "--verbose", "-v", "--no-cuda-graphs",
  "--use-template"
]);

const FLAGS_WITH_VALUES = new Set([
  "--tp", "--max-tokens", "--dtype", "-p", "--prompt", "--trace",
  "--temperature", "--temp", "--top-k", "--top-p", "-i", "--input", "-o", "--output"
]);

function printUsage() {
  console.log(`
Binfer - Bun-based HuggingFace Inference

Usage: bun run src/index.ts <model> [prompt] [options]

Options:
  --tp <n|auto>       Tensor parallel degree (default: auto)
  --max-tokens <n>    Maximum tokens to generate (default: 100)
  --dtype <type>      Weight dtype: fp16 or bf16 (default: model native)
  -p, --prompt <text> Prompt text
  --chat              Interactive chat mode
  --bench             Run benchmark
  --profile           Show timing breakdown summary
  --trace <file>      Write detailed JSON trace to file for analysis
  --no-cache          Disable pre-sharded weight cache (for TP mode)
  -q, --quiet         Suppress loading output
  -v, --verbose       Show startup timing breakdown
  --no-cuda-graphs    Disable CUDA graph capture for decode
  --use-template      Apply chat template to prompt (for single-shot generation)
  --temperature <f>   Sampling temperature (default: 0 = greedy)
  --top-k <n>         Top-k sampling (default: 0 = disabled)
  --top-p <f>         Nucleus sampling threshold (default: 1.0 = disabled)
  -i, --input <file>  Input file with prompts (one per line), processed in batch
  -o, --output <file> Output file for batch results (one per line)

Examples:
  bun run src/index.ts meta-llama/Llama-3.2-1B-Instruct -p "Hello"
  bun run src/index.ts meta-llama/Llama-3.1-70B-Instruct --chat
  bun run src/index.ts model -p "Hello" --trace trace.json
  bun run src/index.ts model -i prompts.txt -o outputs.txt --max-tokens 50
`);
}

// Pre-parse: handle --help and validate flags before loading anything
function preParseArgs(): { shouldExit: boolean; exitCode: number } {
  const args = process.argv.slice(2);

  // Check for help flag first
  if (args.includes("--help") || args.includes("-h")) {
    printUsage();
    return { shouldExit: true, exitCode: 0 };
  }

  // Validate all flags
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg.startsWith("-")) {
      if (!VALID_FLAGS.has(arg)) {
        console.error(`Error: Unknown flag '${arg}'`);
        console.error(`Run with --help to see available options.`);
        return { shouldExit: true, exitCode: 1 };
      }
      // Skip the value for flags that take values
      if (FLAGS_WITH_VALUES.has(arg)) {
        i++;
      }
    }
  }

  return { shouldExit: false, exitCode: 0 };
}

// Run pre-parse immediately
const preParseResult = preParseArgs();
if (preParseResult.shouldExit) {
  process.exit(preParseResult.exitCode);
}

// ============================================================================
// Heavy imports - only loaded after pre-parse passes
// Note: imports load function definitions, actual CUDA init happens when called
// ============================================================================

import { getCudaBackend } from "./backend/cuda/bindings";
import { MmapModelLoader } from "./loader/mmap_loader";
import { TensorParallelModelLoader } from "./parallel/loader";
import { TensorParallelContext } from "./engine/tp_context";
import { createBatchedEngine, BatchedInferenceEngine } from "./engine/batched_engine";
import { createTokenizer, Tokenizer } from "./model/tokenizer";
import { CudaProfiler } from "./profiler/profiler";
import { parseLlamaConfig, autoDetectTP, estimateMemory, LlamaConfig } from "./model/config";
import { existsSync } from "fs";
import { join } from "path";

interface CliArgs {
  model: string;
  prompt: string;
  maxTokens: number;
  tensorParallel: number | "auto";
  dtype: "float16" | "bfloat16" | null;
  chat: boolean;
  bench: boolean;
  profile: boolean;
  trace: string | null;  // Output path for trace file
  noCache: boolean;  // Disable pre-sharded weight cache
  quiet: boolean;    // Suppress loading output
  verbose: boolean;  // Show startup timing
  cudaGraphs: boolean; // Enable CUDA graph capture for decode
  // Sampling parameters
  temperature: number;
  topK: number;
  topP: number;
  // Batch file processing
  inputFile: string | null;
  outputFile: string | null;
}

function parseArgs(): CliArgs {
  const args = process.argv.slice(2);
  const result: CliArgs = {
    model: "",
    prompt: "",
    maxTokens: 100,
    tensorParallel: "auto",  // Default to auto-detection
    dtype: null,
    chat: false,
    bench: false,
    profile: false,
    trace: null,
    noCache: false,
    quiet: false,
    verbose: false,
    cudaGraphs: true,
    useTemplate: false,
    // Default to greedy sampling
    temperature: 0,
    topK: 0,
    topP: 1.0,
    // Batch file processing
    inputFile: null,
    outputFile: null,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--tp" && args[i + 1]) {
      const tpArg = args[++i];
      if (tpArg === "auto") {
        result.tensorParallel = "auto";
      } else {
        result.tensorParallel = parseInt(tpArg, 10);
      }
    } else if (arg === "--max-tokens" && args[i + 1]) {
      result.maxTokens = parseInt(args[++i], 10);
    } else if (arg === "--dtype" && args[i + 1]) {
      const dtype = args[++i];
      if (dtype === "fp16" || dtype === "float16") {
        result.dtype = "float16";
      } else if (dtype === "bf16" || dtype === "bfloat16") {
        result.dtype = "bfloat16";
      } else {
        console.error(`Unknown dtype: ${dtype}. Use fp16 or bf16.`);
        process.exit(1);
      }
    } else if (arg === "-p" || arg === "--prompt") {
      result.prompt = args[++i] || "";
    } else if (arg === "--chat") {
      result.chat = true;
    } else if (arg === "--bench") {
      result.bench = true;
    } else if (arg === "--profile") {
      result.profile = true;
    } else if (arg === "--trace" && args[i + 1]) {
      result.trace = args[++i];
    } else if (arg === "--no-cache") {
      result.noCache = true;
    } else if (arg === "--quiet" || arg === "-q") {
      result.quiet = true;
    } else if (arg === "--verbose" || arg === "-v") {
      result.verbose = true;
    } else if (arg === "--no-cuda-graphs") {
      result.cudaGraphs = false;
    } else if (arg === "--use-template") {
      result.useTemplate = true;
    } else if ((arg === "--temperature" || arg === "--temp") && args[i + 1]) {
      result.temperature = parseFloat(args[++i]);
    } else if (arg === "--top-k" && args[i + 1]) {
      result.topK = parseInt(args[++i], 10);
    } else if (arg === "--top-p" && args[i + 1]) {
      result.topP = parseFloat(args[++i]);
    } else if ((arg === "--input" || arg === "-i") && args[i + 1]) {
      result.inputFile = args[++i];
    } else if ((arg === "--output" || arg === "-o") && args[i + 1]) {
      result.outputFile = args[++i];
    } else if (!arg.startsWith("-") && !result.model) {
      result.model = arg;
    } else if (!arg.startsWith("-")) {
      result.prompt = arg;
    }
  }

  return result;
}

// Animated status indicator using a subprocess for truly independent animation
class StatusLine {
  private proc: ReturnType<typeof Bun.spawn> | null = null;
  private quiet: boolean;
  private lineCount: number = 0;

  constructor(quiet: boolean = false) {
    this.quiet = quiet;
  }

  start(message: string): void {
    if (this.quiet) return;

    // Spawn a subprocess that handles its own animation timing
    this.proc = Bun.spawn(["bun", "-e", `
      const frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
      let msg = ${JSON.stringify(message)};
      const start = Date.now();
      let running = true;

      const rl = require("readline").createInterface({ input: process.stdin });
      rl.on("line", (line) => {
        if (line === "STOP") {
          running = false;
          process.exit(0);
        } else if (line.startsWith("MSG:")) {
          msg = line.slice(4);
        }
      });

      const loop = () => {
        if (!running) return;
        const frame = Math.floor((Date.now() - start) / 80) % frames.length;
        process.stdout.write("\\r\\x1b[K" + frames[frame] + " " + msg);
        setTimeout(loop, 80);
      };
      loop();
    `], {
      stdin: "pipe",
      stdout: "inherit",
      stderr: "inherit",
    });
  }

  update(message: string): void {
    if (this.quiet || !this.proc) return;
    this.proc.stdin.write(`MSG:${message}\n`);
  }

  stop(finalMessage?: string): void {
    if (this.quiet) return;
    if (this.proc) {
      try {
        this.proc.stdin.write("STOP\n");
        this.proc.stdin.end();
      } catch (e) {
        // Ignore errors if process already exited
      }
      // Kill the process to ensure it stops
      this.proc.kill();
      this.proc = null;
    }
    if (finalMessage) {
      process.stdout.write(`\r\x1b[K${finalMessage}\n`);
      this.lineCount++;
    } else {
      process.stdout.write(`\r\x1b[K`);
    }
  }

  // Clear all lines we've written
  clear(): void {
    if (this.quiet) return;
    for (let i = 0; i < this.lineCount; i++) {
      process.stdout.write(`\x1b[A\x1b[K`);
    }
    this.lineCount = 0;
  }

  // Log a message while keeping the spinner active
  log(message: string): void {
    if (this.quiet) return;
    process.stdout.write(`\r\x1b[K${message}\n`);
    this.lineCount++;
  }

  // Update status inline
  progress(message: string): void {
    if (this.quiet) return;
    this.update(message);
  }
}

async function checkCuda(tp: number | "auto", status: StatusLine): Promise<{
  success: boolean;
  deviceCount: number;
  gpuMemoryGB: number;
}> {
  const cuda = getCudaBackend();
  const deviceCount = cuda.getDeviceCount();

  // Get the minimum GPU memory across all GPUs
  let minMemoryGB = Infinity;
  const numToCheck = tp === "auto" ? deviceCount : Math.min(deviceCount, tp as number);

  for (let i = 0; i < numToCheck; i++) {
    const props = cuda.getDeviceProperties(i);
    const memGB = props.totalMemory / 1024 / 1024 / 1024;
    minMemoryGB = Math.min(minMemoryGB, memGB);
  }

  if (tp !== "auto" && deviceCount < tp) {
    status.stop();
    console.error(`Error: Need ${tp} GPUs but only ${deviceCount} available`);
    return { success: false, deviceCount, gpuMemoryGB: minMemoryGB };
  }

  return { success: true, deviceCount, gpuMemoryGB: minMemoryGB };
}

/**
 * Resolve model path and load config for auto-TP detection.
 */
async function resolveModelPath(modelId: string): Promise<string> {
  if (existsSync(modelId)) {
    return modelId;
  }

  // Download from HuggingFace Hub
  const { spawn } = await import("bun");
  const scriptPath = join(import.meta.dir, "../python/hf_download.py");

  const proc = spawn(["python3", scriptPath, "download", modelId], {
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

async function loadConfig(localPath: string): Promise<LlamaConfig> {
  const configPath = join(localPath, "config.json");
  const configJson = JSON.parse(await Bun.file(configPath).text());
  return parseLlamaConfig(configJson);
}

async function loadModel(
  modelPath: string,
  tp: number,
  dtype: "float16" | "bfloat16" | null,
  noCache: boolean = false,
  status?: StatusLine,
  profile: boolean = false
): Promise<{
  ctx: TensorParallelContext;
  localPath: string;
  cleanup: () => void;
}> {
  if (tp === 1) {
    // Single GPU - use native MMAP loader
    const loader = new MmapModelLoader();
    const loadedModel = await loader.load(modelPath, { dtype, status });

    // Wrap weights in per-device map format
    const perDeviceWeights = new Map<number, Map<string, { ptr: bigint; info: any }>>();
    perDeviceWeights.set(0, loadedModel.weights);

    const ctx = new TensorParallelContext(
      loadedModel.config,
      perDeviceWeights,
      1
    );

    return {
      ctx,
      localPath: loadedModel.localPath,
      cleanup: () => loader.freeModel(loadedModel),
    };
  } else {
    // Multi-GPU - use tensor parallel loader
    // Create logger that integrates with status spinner
    const logger = status ? {
      log: (msg: string) => status.log(msg),
      warn: (msg: string) => status.log(msg),
      progress: (msg: string) => status.update(msg),
    } : undefined;

    let t0 = Date.now();
    status?.update("Loading model weights...");
    const loader = new TensorParallelModelLoader({ useCache: !noCache, logger });
    const tpModel = await loader.load(modelPath, tp);
    const loadTime = Date.now() - t0;

    t0 = Date.now();
    status?.update("Creating tensor context...");
    await new Promise(r => setImmediate(r));  // Allow spinner to update
    const ctx = new TensorParallelContext(
      tpModel.config,
      tpModel.perDeviceWeights,
      tp
    );
    const ctxCreateTime = Date.now() - t0;

    // Initialize NCCL communicators
    // NOTE: True parallelism with weight loading would require worker threads,
    // but CUDA contexts are thread-local, making this complex. For now, NCCL
    // init happens sequentially after weight loading.
    t0 = Date.now();
    status?.update("Initializing NCCL communicators...");
    await new Promise(r => setImmediate(r));  // Allow spinner to update
    await ctx.initCommunicators((msg) => status?.update(msg));
    const ncclTime = Date.now() - t0;

    if (profile) {
      logger?.log(`  Weights: ${(loadTime / 1000).toFixed(2)}s, Context: ${(ctxCreateTime / 1000).toFixed(2)}s, NCCL: ${(ncclTime / 1000).toFixed(2)}s`);
    }

    return {
      ctx,
      localPath: tpModel.localPath,
      cleanup: () => loader.freeModel(tpModel),
    };
  }
}

async function chat(
  engine: BatchedInferenceEngine,
  tokenizer: Tokenizer,
  samplingParams: { temperature: number; topK: number; topP: number }
): Promise<void> {
  const readline = await import("readline");

  // Enable raw mode to capture Ctrl+C during generation
  if (process.stdin.isTTY) {
    process.stdin.setRawMode(false);
  }

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const messages: Array<{ role: "user" | "assistant"; content: string }> = [];
  let isGenerating = false;
  let abortGeneration = false;

  // Handle Ctrl+C gracefully
  const handleInterrupt = () => {
    if (isGenerating) {
      abortGeneration = true;
      process.stdout.write("\n[interrupted]\n");
    } else {
      console.log("\n\nExiting chat...");
      rl.close();
      process.exit(0);
    }
  };

  // Catch SIGINT at process level
  process.on("SIGINT", handleInterrupt);

  // Also listen on readline
  rl.on("SIGINT", handleInterrupt);

  // Handle close event
  rl.on("close", () => {
    process.off("SIGINT", handleInterrupt);
  });

  console.log("\nChat mode started. Type 'exit' to quit, Ctrl+C to interrupt.\n");

  const askQuestion = (): Promise<void> => {
    return new Promise((resolve) => {
      rl.question("You: ", async (input) => {
        if (input === null || input.toLowerCase() === "exit") {
          rl.close();
          resolve();
          return;
        }

        messages.push({ role: "user", content: input });

        const prompt = tokenizer.applyChatTemplate(messages);
        const inputIds = tokenizer.encode(prompt);

        process.stdout.write("Assistant: ");

        let generatedText = "";
        isGenerating = true;
        abortGeneration = false;

        await engine.generate(inputIds, 512, (token) => {
          if (abortGeneration) return false;
          if (token === engine.config.eosTokenId) return;
          const text = tokenizer.decode([token]);
          process.stdout.write(text);
          generatedText += text;
        }, samplingParams);

        isGenerating = false;
        console.log("\n");

        if (!abortGeneration) {
          messages.push({ role: "assistant", content: generatedText });
        }

        resolve();
      });
    });
  };

  // Main chat loop
  while (true) {
    await askQuestion();
    if (!rl.terminal) break; // readline was closed
  }
}

async function runBenchmark(
  engine: BatchedInferenceEngine,
  tokenizer: Tokenizer,
  profile: boolean = false
) {
  console.log();
  console.log("Running benchmark...");
  console.log();

  const promptLengths = [128, 256, 512];
  const genLength = 128;

  for (const promptLen of promptLengths) {
    const dummyText = "Hello ".repeat(promptLen / 2);
    const inputIds = tokenizer.encode(dummyText, { maxLength: promptLen });

    // Reset engine state between runs
    engine.reset();

    const startTime = performance.now();
    let tokenCount = 0;

    await engine.generate(inputIds, genLength, () => {
      tokenCount++;
    });

    const elapsed = (performance.now() - startTime) / 1000;
    const tokensPerSec = tokenCount / elapsed;

    console.log(
      `Prompt: ${inputIds.length} tokens, Generated: ${tokenCount} tokens, ` +
        `Time: ${elapsed.toFixed(2)}s, Speed: ${tokensPerSec.toFixed(1)} tok/s`
    );

    // Print per-run profiling if enabled
    if (profile) {
      const metrics = engine.getProfileMetrics();
      if (metrics.size > 0) {
        console.log(CudaProfiler.formatReport(metrics, tokenCount, elapsed * 1000));
        console.log();
      }
    }
  }
}

/**
 * Process a batch of prompts from a file and write results to output file.
 * Uses true GPU batching: all sequences are processed together in each decode step.
 */
async function runBatchFile(
  engine: BatchedInferenceEngine,
  tokenizer: Tokenizer,
  inputFile: string,
  outputFile: string,
  maxTokens: number,
  profile: boolean = false
) {
  // Read input file
  const inputContent = await Bun.file(inputFile).text();
  const prompts = inputContent.trim().split("\n").filter(line => line.trim().length > 0);

  if (prompts.length === 0) {
    console.error("No prompts found in input file");
    process.exit(1);
  }

  console.log(`Processing ${prompts.length} prompts with true GPU batching...`);
  const startTime = performance.now();

  // Tokenize all prompts
  const inputTokens = prompts.map(p => tokenizer.encode(p));
  const eosTokenId = engine.config.eosTokenId;

  // Allocate KV states for all sequences
  const kvStates: ReturnType<typeof engine.allocateSequence>[] = [];
  for (let i = 0; i < prompts.length; i++) {
    const promptLen = inputTokens[i].length;
    const totalLen = promptLen + maxTokens;
    const kvState = engine.allocateSequence(i, promptLen);
    if (!kvState) {
      console.error(`Failed to allocate KV state for sequence ${i}`);
      // Free already allocated
      for (const state of kvStates) {
        if (state) engine.freeSequence(state);
      }
      process.exit(1);
    }
    kvStates.push(kvState);
  }

  // Track generated tokens per sequence
  const generatedTokens: number[][] = prompts.map(() => []);
  const completed: boolean[] = prompts.map(() => false);
  let totalTokens = 0;
  let activeCount = prompts.length;

  // 1. Prefill all sequences
  const prefillResult = await engine.prefillBatch(inputTokens, kvStates as any);

  // Update numTokens after prefill (critical: allocateSequence sets numTokens=0)
  for (let i = 0; i < prompts.length; i++) {
    kvStates[i]!.numTokens = inputTokens[i].length;
  }

  // Sample first token for each sequence (greedy - argmax)
  for (let i = 0; i < prompts.length; i++) {
    const logits = prefillResult.logits[i];
    // Greedy argmax
    let maxIdx = 0;
    let maxVal = logits[0];
    for (let j = 1; j < logits.length; j++) {
      if (logits[j] > maxVal) {
        maxVal = logits[j];
        maxIdx = j;
      }
    }
    const token = maxIdx;

    if (token === eosTokenId) {
      completed[i] = true;
      activeCount--;
    } else {
      generatedTokens[i].push(token);
      totalTokens++;
    }

    // Extend KV state for the generated token
    engine.extendSequence(kvStates[i]!, 1);
  }

  // 2. Decode loop - process all active sequences together
  while (activeCount > 0) {
    // Build batch of active sequences
    const activeIndices: number[] = [];
    const activeTokens: number[] = [];
    const activeKvStates: typeof kvStates = [];

    for (let i = 0; i < prompts.length; i++) {
      if (!completed[i]) {
        activeIndices.push(i);
        activeTokens.push(generatedTokens[i][generatedTokens[i].length - 1]);
        activeKvStates.push(kvStates[i]);
        // Extend KV state BEFORE decode (decodeBatchGreedy expects this)
        engine.extendSequence(kvStates[i]!, 1);
      }
    }

    // Decode batch with GPU greedy sampling
    const decodeResult = await engine.decodeBatchGreedy(activeTokens, activeKvStates as any);

    // Process results
    for (let j = 0; j < activeIndices.length; j++) {
      const i = activeIndices[j];
      const token = decodeResult.tokenIds[j];

      if (token === eosTokenId || generatedTokens[i].length >= maxTokens) {
        completed[i] = true;
        activeCount--;
      } else {
        generatedTokens[i].push(token);
        totalTokens++;
      }
    }

    // Progress update
    const elapsed = (performance.now() - startTime) / 1000;
    const tokPerSec = totalTokens / elapsed;
    const completedCount = prompts.length - activeCount;
    process.stdout.write(`\rProgress: ${completedCount}/${prompts.length} done | ${totalTokens} tokens | ${tokPerSec.toFixed(1)} tok/s  `);
  }

  const elapsed = (performance.now() - startTime) / 1000;
  const tokPerSec = totalTokens / elapsed;

  console.log(`\n\nCompleted ${prompts.length} prompts in ${elapsed.toFixed(2)}s`);
  console.log(`Total: ${totalTokens} tokens, ${tokPerSec.toFixed(1)} tok/s`);

  // Decode results
  const results = generatedTokens.map(tokens => tokenizer.decode(tokens));

  // Free all KV states
  for (const state of kvStates) {
    if (state) engine.freeSequence(state);
  }

  // Write output file
  await Bun.write(outputFile, results.join("\n") + "\n");
  console.log(`Results written to ${outputFile}`);

  // Print profiling if enabled
  if (profile) {
    const metrics = engine.getProfileMetrics();
    if (metrics.size > 0) {
      console.log();
      console.log(CudaProfiler.formatReport(metrics, totalTokens, elapsed * 1000));
    }
  }
}

async function main() {
  const startupStart = Date.now();
  const args = parseArgs();

  if (!args.model) {
    printUsage();
    process.exit(1);
  }

  const status = new StatusLine(args.quiet);
  const timings: { [key: string]: number } = {};

  // Check CUDA
  let t0 = Date.now();
  status.start("Checking CUDA availability...");
  const cudaCheck = await checkCuda(args.tensorParallel, status);
  if (!cudaCheck.success) {
    process.exit(1);
  }
  timings["CUDA check"] = Date.now() - t0;

  // Resolve TP (auto-detect if needed)
  let tp: number;
  try {
    t0 = Date.now();
    if (args.tensorParallel === "auto") {
      status.update("Detecting optimal configuration...");
      const localPath = await resolveModelPath(args.model);
      const config = await loadConfig(localPath);
      tp = autoDetectTP(config, cudaCheck.gpuMemoryGB, cudaCheck.deviceCount);
    } else {
      tp = args.tensorParallel;
    }
    timings["TP detection"] = Date.now() - t0;
  } catch (error) {
    status.stop();
    console.error(`Error: ${(error as Error).message}`);
    process.exit(1);
  }

  try {
    // Load model (single or multi-GPU)
    t0 = Date.now();
    status.update("Loading model weights...");
    const { ctx, localPath, cleanup } = await loadModel(args.model, tp, args.dtype, args.noCache, status, args.profile);
    timings["Model loading"] = Date.now() - t0;

    // Load tokenizer
    t0 = Date.now();
    status.update("Loading tokenizer...");
    const tokenizer = await createTokenizer(localPath);
    timings["Tokenizer"] = Date.now() - t0;

    // Create inference engine
    t0 = Date.now();
    const engine = createBatchedEngine(ctx, { quiet: true, cudaGraphs: args.cudaGraphs });
    timings["Engine creation"] = Date.now() - t0;

    // Enable profiling if requested
    if (args.profile) {
      engine.enableProfiling();
    }

    // Clear all loading output before starting
    status.stop();
    status.clear();

    // Print startup timing breakdown
    timings["Total startup"] = Date.now() - startupStart;
    if (args.verbose || args.profile) {
      console.log("Startup timing:");
      for (const [name, ms] of Object.entries(timings)) {
        console.log(`  ${name}: ${(ms / 1000).toFixed(2)}s`);
      }
      console.log();
    }

    if (args.chat) {
      await chat(engine, tokenizer, {
        temperature: args.temperature,
        topK: args.topK,
        topP: args.topP,
      });
    } else if (args.bench) {
      await runBenchmark(engine, tokenizer, args.profile);
    } else if (args.inputFile) {
      // Batch file processing
      const outputFile = args.outputFile || args.inputFile.replace(/\.txt$/, "") + "_output.txt";
      await runBatchFile(engine, tokenizer, args.inputFile, outputFile, args.maxTokens, args.profile);
    } else {
      // Single generation
      const rawPrompt = args.prompt || "Once upon a time, there was a";

      let prompt: string;
      if (args.useTemplate) {
        // Apply chat template to treat prompt as a user message
        prompt = tokenizer.applyChatTemplate([{ role: "user", content: rawPrompt }], { addGenerationPrompt: true });
      } else {
        prompt = rawPrompt;
      }

      const inputIds = tokenizer.encode(prompt);

      const startTime = Date.now();
      const generatedTokens = await engine.generate(
        inputIds,
        args.maxTokens,
        (token) => {
          const text = tokenizer.decode([token]);
          process.stdout.write(text);
        },
        {
          temperature: args.temperature,
          topK: args.topK,
          topP: args.topP,
        }
      );
      const elapsed = (Date.now() - startTime) / 1000;

      // Print stats (unless quiet)
      if (!args.quiet) {
        console.log();
        console.log();
        console.log(`[${generatedTokens.length} tokens, ${elapsed.toFixed(2)}s, ${(generatedTokens.length / elapsed).toFixed(1)} tok/s]`);
      } else {
        console.log();
      }

      // Print profiling report if enabled
      if (args.profile) {
        const metrics = engine.getProfileMetrics();
        if (metrics.size > 0) {
          console.log();
          console.log(CudaProfiler.formatReport(
            metrics,
            generatedTokens.length,
            elapsed * 1000
          ));
        }
      }
    }

    // Cleanup
    await tokenizer.stop();
    ctx.dispose();
    cleanup();

  } catch (error) {
    status.stop();
    console.error("Error:", error);
    process.exit(1);
  }
}

// Exports for programmatic use
export { ModelLoader } from "./model/loader";
export { MmapModelLoader } from "./loader/mmap_loader";
export { TensorParallelModelLoader } from "./parallel/loader";
export { parseLlamaConfig } from "./model/config";
export type { LlamaConfig } from "./model/config";
export { Tensor } from "./tensor/tensor";
export { DType } from "./tensor/dtype";
export { CudaBackend, getCudaBackend } from "./backend/cuda/bindings";
export { BackendType } from "./backend/interface";
export type { Backend } from "./backend/interface";
export { InferenceEngine } from "./engine/engine_legacy";
export { BatchedInferenceEngine, createBatchedEngine } from "./engine/batched_engine";
export { TensorParallelContext } from "./engine/tp_context";
export type { DeviceContext } from "./engine/device_context";
export { SingleDeviceContext, MultiDeviceContext } from "./engine/device_context";
export { Tokenizer, createTokenizer } from "./model/tokenizer";
export { Sampler } from "./engine/sampler";
export type { SamplingParams } from "./engine/sampler";
export { KVCache, createKVCache } from "./kv/manager";
export type { IKVCache } from "./kv/manager";

// Run if executed directly
main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
