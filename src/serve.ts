#!/usr/bin/env bun
// Binfer Server with Continuous Batching
// OpenAI-compatible API server with high-throughput dynamic batching

import { getCudaBackend } from "./backend/cuda/bindings";
import { MmapModelLoader } from "./loader/mmap_loader";
import { TensorParallelModelLoader } from "./parallel/loader";
import { TensorParallelContext } from "./engine/tp_context";
import { BatchedInferenceEngine, createBatchedEngine, SequenceKVState } from "./engine/batched_engine";
import { InferenceEngine } from "./engine/engine_legacy";
import { createTokenizer, Tokenizer } from "./model/tokenizer";
import { LlamaConfig } from "./model/config";
import { Sampler, SamplingParams } from "./engine/sampler";

interface ServerArgs {
  model: string;
  port: number;
  maxTokens: number;
  tp: number;
  continuous: boolean;  // Use continuous batching
  maxBatchSize: number;
  gpuUtil: number;  // Fraction of free GPU memory to use for KV cache (0.0-1.0)
  profile: "off" | "basic" | "detailed";  // Profiling mode
}

function parseArgs(): ServerArgs {
  const args = process.argv.slice(2);
  const result: ServerArgs = {
    model: "",
    port: 8000,
    maxTokens: 256,
    tp: 1,
    continuous: true,  // Always use continuous batching (more robust)
    maxBatchSize: 64,
    gpuUtil: 0.5,  // Default: use 50% of free GPU memory (leave headroom for activations)
    profile: "off",
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--port" && args[i + 1]) {
      result.port = parseInt(args[++i], 10);
    } else if (arg === "--max-tokens" && args[i + 1]) {
      result.maxTokens = parseInt(args[++i], 10);
    } else if (arg === "--tp" && args[i + 1]) {
      result.tp = parseInt(args[++i], 10);
    } else if (arg === "--max-batch-size" && args[i + 1]) {
      result.maxBatchSize = parseInt(args[++i], 10);
    } else if (arg === "--gpu-util" && args[i + 1]) {
      result.gpuUtil = parseFloat(args[++i]);
      if (result.gpuUtil <= 0 || result.gpuUtil > 1.0) {
        console.error("--gpu-util must be between 0.0 and 1.0");
        process.exit(1);
      }
    } else if (arg === "--profile") {
      const next = args[i + 1];
      if (next === "detailed") {
        result.profile = "detailed";
        i++;
      } else {
        result.profile = "basic";
      }
    } else if (!arg.startsWith("--")) {
      result.model = arg;
    }
  }

  return result;
}

function printUsage() {
  console.log(`
Usage: bun run src/serve.ts <model> [options]

Options:
  --port <port>           Server port (default: 8000)
  --max-tokens <n>        Default max tokens per request (default: 256)
  --tp <n>                Tensor parallelism degree (default: 1)
  --max-batch-size <n>    Maximum concurrent sequences (default: 64)
  --gpu-util <0.0-1.0>    Fraction of free GPU memory for KV cache (default: 0.5)
  --profile [detailed]    Enable live profiling stats (add 'detailed' for op breakdown)

Examples:
  bun run src/serve.ts openai/gpt-oss-20b
  bun run src/serve.ts openai/gpt-oss-120b --tp 2 --gpu-util 0.8
  bun run src/serve.ts openai/gpt-oss-20b --profile
  bun run src/serve.ts openai/gpt-oss-20b --profile detailed
`);
}

// ============================================================================
// Live Profiling Stats (EMA)
// ============================================================================

class EmaStats {
  private alpha: number;
  private values: Map<string, number> = new Map();
  private counts: Map<string, number> = new Map();

  constructor(alpha: number = 0.1) {
    this.alpha = alpha;
  }

  update(name: string, value: number) {
    const prev = this.values.get(name);
    if (prev === undefined) {
      this.values.set(name, value);
    } else {
      this.values.set(name, this.alpha * value + (1 - this.alpha) * prev);
    }
    this.counts.set(name, (this.counts.get(name) ?? 0) + 1);
  }

  get(name: string): number | undefined {
    return this.values.get(name);
  }

  getCount(name: string): number {
    return this.counts.get(name) ?? 0;
  }

  getAll(): Map<string, number> {
    return new Map(this.values);
  }
}

// Profiling state
let profilingEnabled = false;
let profilingDetailed = false;
let emaStats: EmaStats | null = null;
let profileDisplayLines = 0;  // Track how many lines we've printed for refresh

// Tracking vars for per-batch stats
let lastBatchTime = 0;
let lastBatchTokens = 0;
let totalTokensGenerated = 0;
let totalRequestsCompleted = 0;

// Helper to pad a line to fixed width with right border
function padLine(content: string, width: number = 57): string {
  const padding = width - content.length;
  if (padding < 0) return content.slice(0, width) + "│";
  return content + " ".repeat(padding) + "│";
}

function updateProfilingDisplay() {
  if (!profilingEnabled || !emaStats || !batchedEngine) return;

  const stats = batchedEngine.getStats();
  const tokPerSec = emaStats.get("tok/s") ?? 0;
  const decodeMs = emaStats.get("decode_ms") ?? 0;
  const prefillMs = emaStats.get("prefill_ms") ?? 0;
  const batchSize = emaStats.get("batch_size") ?? 0;
  const kvUsed = stats.totalBlocks - stats.freeBlocks;
  const kvPct = (kvUsed / stats.totalBlocks) * 100;

  // Move cursor up and clear lines, then reprint
  if (profileDisplayLines > 0) {
    process.stdout.write(`\x1b[${profileDisplayLines}A\x1b[J`);
  }

  const W = 57;
  const border = "─".repeat(W);
  const lines: string[] = [
    `┌${border}┐`,
    padLine("│ PROFILING (EMA α=0.1)"),
    `├${border}┤`,
    padLine(`│  Throughput: ${tokPerSec.toFixed(1).padStart(7)} tok/s   Batch size: ${batchSize.toFixed(1).padStart(5)}`),
    padLine(`│  Decode:     ${decodeMs.toFixed(2).padStart(7)} ms      Prefill:    ${prefillMs.toFixed(2).padStart(7)} ms`),
    padLine(`│  KV cache:   ${kvUsed.toString().padStart(6)} / ${stats.totalBlocks} blocks (${kvPct.toFixed(1).padStart(5)}%)`),
    padLine(`│  Pending:    ${pendingRequests.size.toString().padStart(6)}         Running:    ${runningRequests.size.toString().padStart(6)}`),
    padLine(`│  Generated:  ${totalTokensGenerated.toString().padStart(6)} tokens   Completed:  ${totalRequestsCompleted.toString().padStart(6)} reqs`),
  ];

  // Add detailed breakdown if enabled
  if (profilingDetailed) {
    lines.push(`├${border}┤`);
    lines.push(padLine("│ DECODE BREAKDOWN"));

    const qkvMs = emaStats.get("qkv_proj_ms") ?? 0;
    const attnMs = emaStats.get("attn_ms") ?? 0;
    const ffnMs = emaStats.get("ffn_ms") ?? 0;
    const otherMs = Math.max(0, decodeMs - qkvMs - attnMs - ffnMs);

    const total = qkvMs + attnMs + ffnMs + otherMs;
    const qkvPct = total > 0 ? (qkvMs / total) * 100 : 0;
    const attnPct = total > 0 ? (attnMs / total) * 100 : 0;
    const ffnPct = total > 0 ? (ffnMs / total) * 100 : 0;
    const otherPct = total > 0 ? (otherMs / total) * 100 : 0;

    lines.push(padLine(`│  QKV proj:   ${qkvMs.toFixed(2).padStart(7)} ms  (${qkvPct.toFixed(0).padStart(3)}%)`));
    lines.push(padLine(`│  Attention:  ${attnMs.toFixed(2).padStart(7)} ms  (${attnPct.toFixed(0).padStart(3)}%)`));
    lines.push(padLine(`│  FFN/MoE:    ${ffnMs.toFixed(2).padStart(7)} ms  (${ffnPct.toFixed(0).padStart(3)}%)`));
    lines.push(padLine(`│  Other:      ${otherMs.toFixed(2).padStart(7)} ms  (${otherPct.toFixed(0).padStart(3)}%)`));
  }

  lines.push(`└${border}┘`);

  console.log(lines.join("\n"));
  profileDisplayLines = lines.length;
}

// ============================================================================
// Request Queue for Continuous Batching
// ============================================================================

// Queue configuration - sane defaults for production
const MAX_CONCURRENT_PREFILLS = parseInt(process.env.MAX_PREFILLS || "1", 10);  // 1 for large models, 2-4 for smaller
const MAX_QUEUE_SIZE = parseInt(process.env.MAX_QUEUE_SIZE || "100", 10);  // Max pending requests
const QUEUE_TIMEOUT_MS = parseInt(process.env.QUEUE_TIMEOUT_MS || "30000", 10);  // 30 second default

interface PendingRequest {
  id: string;
  promptTokenIds: number[];
  maxTokens: number;
  temperature: number;
  resolve: (response: Response) => void;
  reject: (error: Error) => void;
  stream: boolean;
  controller?: ReadableStreamDefaultController;
  encoder?: TextEncoder;
  generatedTokens: number[];
  kvState: SequenceKVState | null;
  prefillComplete: boolean;
  queuedAt: number;  // Timestamp when request was queued
}

// Global state
let engine: InferenceEngine | null = null;
let batchedEngine: BatchedInferenceEngine | null = null;
let tokenizer: Tokenizer;
let config: LlamaConfig;
let modelId: string;
let defaultMaxTokens: number;
let useContinuousBatching: boolean = false;

// Request management
const pendingRequests: Map<string, PendingRequest> = new Map();
const runningRequests: Map<string, PendingRequest> = new Map();
let nextRequestId = 0;
let processing = false;

// Simple request queue for single-request mode
const requestQueue: Array<{
  resolve: (response: Response) => void;
  reject: (error: Error) => void;
  handler: () => Promise<Response>;
}> = [];

async function processQueue() {
  if (processing || requestQueue.length === 0) return;
  processing = true;

  while (requestQueue.length > 0) {
    const item = requestQueue.shift()!;
    try {
      const response = await item.handler();
      item.resolve(response);
    } catch (error) {
      item.reject(error as Error);
    }
  }

  processing = false;
}

function queueRequest(handler: () => Promise<Response>): Promise<Response> {
  return new Promise((resolve, reject) => {
    requestQueue.push({ resolve, reject, handler });
    processQueue();
  });
}

// ============================================================================
// Continuous Batching Inference Loop
// ============================================================================

let inferenceLoopRunning = false;

async function continuousBatchingLoop() {
  if (!batchedEngine) return;

  const sampler = new Sampler({ temperature: 0.7 });

  while (inferenceLoopRunning) {
    // Check if there's work to do
    if (pendingRequests.size === 0 && runningRequests.size === 0) {
      await Bun.sleep(1);
      continue;
    }

    const now = Date.now();

    // Check for timed-out pending requests
    for (const [id, req] of pendingRequests) {
      if (now - req.queuedAt > QUEUE_TIMEOUT_MS) {
        pendingRequests.delete(id);
        const timeoutError = new Error(`Request timed out after ${QUEUE_TIMEOUT_MS}ms in queue`);
        if (req.stream && req.controller) {
          try {
            const errorChunk = {
              error: { message: timeoutError.message, type: "timeout_error" }
            };
            req.controller.enqueue(req.encoder!.encode(`data: ${JSON.stringify(errorChunk)}\n\n`));
            req.controller.close();
          } catch (e) {
            // Controller may already be closed
          }
        } else {
          req.resolve(Response.json(
            { error: { message: timeoutError.message, type: "timeout_error" } },
            { status: 408, headers: { "Access-Control-Allow-Origin": "*" } }
          ));
        }
        console.log(`Request ${id} timed out in queue`);
      }
    }

    // Count how many prefills are currently in flight
    let prefillsInFlight = 0;
    for (const req of runningRequests.values()) {
      if (!req.prefillComplete) {
        prefillsInFlight++;
      }
    }

    // Move pending requests to running (if we have capacity)
    // Limit to MAX_CONCURRENT_PREFILLS to prevent OOM
    for (const [id, req] of pendingRequests) {
      if (prefillsInFlight >= MAX_CONCURRENT_PREFILLS) {
        break;  // Already at max prefills, wait for current ones to complete
      }

      // Try to allocate KV blocks
      const totalTokens = req.promptTokenIds.length + req.maxTokens;
      if (!batchedEngine.canAllocate(totalTokens)) {
        continue;  // No memory, wait
      }

      const kvState = batchedEngine.allocateSequence(
        nextRequestId++,
        req.promptTokenIds.length
      );

      if (!kvState) {
        continue;  // Allocation failed
      }

      req.kvState = kvState;
      pendingRequests.delete(id);
      runningRequests.set(id, req);
      prefillsInFlight++;
    }

    // Get prefill and decode batches
    const prefillBatch: PendingRequest[] = [];
    const decodeBatch: PendingRequest[] = [];

    for (const req of runningRequests.values()) {
      if (!req.prefillComplete) {
        prefillBatch.push(req);
      } else {
        decodeBatch.push(req);
      }
    }

    // Process prefill batch
    if (prefillBatch.length > 0) {
      const inputTokens = prefillBatch.map(r => r.promptTokenIds);
      const kvStates = prefillBatch.map(r => r.kvState!);

      try {
        const prefillStart = performance.now();
        const results = await batchedEngine.prefillBatch(inputTokens, kvStates);
        const prefillMs = performance.now() - prefillStart;

        if (profilingEnabled && emaStats) {
          emaStats.update("prefill_ms", prefillMs);
        }

        for (let i = 0; i < prefillBatch.length; i++) {
          const req = prefillBatch[i];
          const logits = results.logits[i];
          const nextToken = sampler.sample(logits, req.generatedTokens);

          req.generatedTokens.push(nextToken);
          req.prefillComplete = true;

          // Extend KV state for generated token
          batchedEngine.extendSequence(req.kvState!, 1);

          // Stream the token if streaming
          if (req.stream && req.controller) {
            const text = tokenizer.decode([nextToken]);
            const chunk = {
              id: req.id,
              object: "chat.completion.chunk",
              created: Math.floor(Date.now() / 1000),
              model: modelId,
              choices: [{
                index: 0,
                delta: { content: text },
                finish_reason: null,
              }],
            };
            req.controller.enqueue(req.encoder!.encode(`data: ${JSON.stringify(chunk)}\n\n`));
          }
        }
      } catch (error) {
        console.error("Prefill error:", error);
        for (const req of prefillBatch) {
          req.reject(error as Error);
          runningRequests.delete(req.id);
          if (req.kvState) {
            batchedEngine.freeSequence(req.kvState);
          }
        }
      }
    }

    // Process decode batch - use GPU greedy sampling for speed
    if (decodeBatch.length > 0) {
      const tokens = decodeBatch.map(r => r.generatedTokens[r.generatedTokens.length - 1]);
      const kvStates = decodeBatch.map(r => r.kvState!);

      try {
        const decodeStart = performance.now();
        // Use GPU greedy sampling - avoids 2MB D2H copy per batch
        const results = await batchedEngine.decodeBatchGreedy(tokens, kvStates);
        const decodeMs = performance.now() - decodeStart;

        if (profilingEnabled && emaStats) {
          emaStats.update("decode_ms", decodeMs);
          emaStats.update("batch_size", decodeBatch.length);
          // Calculate tok/s: tokens in this batch / time
          const tokPerSec = (decodeBatch.length * 1000) / decodeMs;
          emaStats.update("tok/s", tokPerSec);
          totalTokensGenerated += decodeBatch.length;

          // Collect detailed breakdown if enabled
          if (profilingDetailed) {
            const metrics = batchedEngine.getProfileMetrics();

            // Aggregate QKV projection time
            let qkvMs = 0;
            qkvMs += metrics.get("qkv_proj") ?? 0;
            qkvMs += metrics.get("rope") ?? 0;
            qkvMs += metrics.get("kv_update") ?? 0;
            if (qkvMs > 0) emaStats.update("qkv_proj_ms", qkvMs);

            // Aggregate attention time
            let attnMs = 0;
            attnMs += metrics.get("attention") ?? 0;
            attnMs += metrics.get("o_proj") ?? 0;
            attnMs += metrics.get("attn_allreduce") ?? 0;
            if (attnMs > 0) emaStats.update("attn_ms", attnMs);

            // Aggregate FFN/MoE time
            let ffnMs = 0;
            ffnMs += metrics.get("moe") ?? 0;
            ffnMs += metrics.get("mlp_gate_up") ?? 0;
            ffnMs += metrics.get("swiglu") ?? 0;
            ffnMs += metrics.get("mlp_down") ?? 0;
            ffnMs += metrics.get("mlp_allreduce") ?? 0;
            if (ffnMs > 0) emaStats.update("ffn_ms", ffnMs);
          }
        }

        for (let i = 0; i < decodeBatch.length; i++) {
          const req = decodeBatch[i];
          const nextToken = results.tokenIds[i];

          req.generatedTokens.push(nextToken);

          // Extend KV state
          batchedEngine.extendSequence(req.kvState!, 1);

          // Check for completion
          const isComplete =
            nextToken === config.eosTokenId ||
            req.generatedTokens.length >= req.maxTokens;

          if (req.stream && req.controller) {
            const text = tokenizer.decode([nextToken]);
            const chunk = {
              id: req.id,
              object: "chat.completion.chunk",
              created: Math.floor(Date.now() / 1000),
              model: modelId,
              choices: [{
                index: 0,
                delta: isComplete ? {} : { content: text },
                finish_reason: isComplete ? "stop" : null,
              }],
            };
            req.controller.enqueue(req.encoder!.encode(`data: ${JSON.stringify(chunk)}\n\n`));

            if (isComplete) {
              req.controller.enqueue(req.encoder!.encode("data: [DONE]\n\n"));
              req.controller.close();
            }
          }

          if (isComplete) {
            // Complete the request
            if (!req.stream) {
              const generatedText = tokenizer.decode(req.generatedTokens);
              req.resolve(Response.json({
                id: req.id,
                object: "chat.completion",
                created: Math.floor(Date.now() / 1000),
                model: modelId,
                choices: [{
                  index: 0,
                  message: { role: "assistant", content: generatedText },
                  finish_reason: "stop",
                }],
                usage: {
                  prompt_tokens: req.promptTokenIds.length,
                  completion_tokens: req.generatedTokens.length,
                  total_tokens: req.promptTokenIds.length + req.generatedTokens.length,
                },
              }, {
                headers: { "Access-Control-Allow-Origin": "*" },
              }));
            }

            runningRequests.delete(req.id);
            batchedEngine.freeSequence(req.kvState!);
            if (profilingEnabled) {
              totalRequestsCompleted++;
            }
          }
        }
      } catch (error) {
        console.error("Decode error:", error);
        for (const req of decodeBatch) {
          req.reject(error as Error);
          runningRequests.delete(req.id);
          if (req.kvState) {
            batchedEngine.freeSequence(req.kvState);
          }
        }
      }
    }

    // Yield to event loop
    await Bun.sleep(0);
  }
}

// ============================================================================
// HTTP Handler
// ============================================================================

interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
}

async function handleRequest(req: Request): Promise<Response> {
  const url = new URL(req.url);

  // CORS preflight
  if (req.method === "OPTIONS") {
    return new Response(null, {
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
      },
    });
  }

  try {
    if (url.pathname === "/v1/models" && req.method === "GET") {
      return Response.json({
        object: "list",
        data: [{
          id: modelId,
          object: "model",
          created: Date.now(),
          owned_by: "binfer",
        }],
      }, {
        headers: { "Access-Control-Allow-Origin": "*" },
      });
    }

    if (url.pathname === "/v1/chat/completions" && req.method === "POST") {
      // Parse JSON with better error handling
      let body: ChatCompletionRequest;
      try {
        const text = await req.text();
        if (!text || text.trim() === "") {
          return Response.json(
            { error: { message: "Request body is empty", type: "invalid_request_error" } },
            { status: 400, headers: { "Access-Control-Allow-Origin": "*" } }
          );
        }
        body = JSON.parse(text) as ChatCompletionRequest;
      } catch (parseError) {
        return Response.json(
          { error: { message: "Invalid JSON in request body", type: "invalid_request_error" } },
          { status: 400, headers: { "Access-Control-Allow-Origin": "*" } }
        );
      }

      if (useContinuousBatching) {
        return handleContinuousBatchingRequest(body);
      }

      if (body.stream) {
        return handleStreamingChat(body);
      }

      return queueRequest(() => handleNonStreamingChat(body));
    }

    if (url.pathname === "/stats" && req.method === "GET") {
      if (batchedEngine) {
        const stats = batchedEngine.getStats();
        // Count prefills in flight
        let prefillsInFlight = 0;
        for (const req of runningRequests.values()) {
          if (!req.prefillComplete) {
            prefillsInFlight++;
          }
        }
        return Response.json({
          mode: "continuous_batching",
          pending_requests: pendingRequests.size,
          running_requests: runningRequests.size,
          prefills_in_flight: prefillsInFlight,
          max_concurrent_prefills: MAX_CONCURRENT_PREFILLS,
          max_queue_size: MAX_QUEUE_SIZE,
          queue_timeout_ms: QUEUE_TIMEOUT_MS,
          kv_cache: stats,
        }, {
          headers: { "Access-Control-Allow-Origin": "*" },
        });
      }
      return Response.json({
        mode: "single_request",
        queue_length: requestQueue.length,
      }, {
        headers: { "Access-Control-Allow-Origin": "*" },
      });
    }

    if (url.pathname === "/health" && req.method === "GET") {
      return Response.json({ status: "ok" }, {
        headers: { "Access-Control-Allow-Origin": "*" },
      });
    }

    return new Response("Not Found", { status: 404 });
  } catch (error) {
    console.error("Request error:", error);
    return Response.json(
      { error: { message: String(error), type: "server_error" } },
      { status: 500, headers: { "Access-Control-Allow-Origin": "*" } }
    );
  }
}

function handleContinuousBatchingRequest(body: ChatCompletionRequest): Response {
  // Check queue size - return 503 if overloaded
  if (pendingRequests.size >= MAX_QUEUE_SIZE) {
    return Response.json(
      {
        error: {
          message: `Server overloaded: ${pendingRequests.size} requests queued. Please retry later.`,
          type: "server_overloaded"
        }
      },
      { status: 503, headers: { "Access-Control-Allow-Origin": "*", "Retry-After": "5" } }
    );
  }

  const prompt = tokenizer.applyChatTemplate(body.messages);
  const inputIds = tokenizer.encode(prompt);
  const maxTokens = body.max_tokens ?? defaultMaxTokens;
  const temperature = body.temperature ?? 0.7;
  const requestId = `chatcmpl-${Date.now()}-${nextRequestId++}`;
  const queuedAt = Date.now();

  if (body.stream) {
    const encoder = new TextEncoder();

    const stream = new ReadableStream({
      start: (controller) => {
        // Send initial chunk
        const initialChunk = {
          id: requestId,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: modelId,
          choices: [{
            index: 0,
            delta: { role: "assistant", content: "" },
            finish_reason: null,
          }],
        };
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(initialChunk)}\n\n`));

        // Create pending request
        const pendingReq: PendingRequest = {
          id: requestId,
          promptTokenIds: inputIds,
          maxTokens,
          temperature,
          resolve: () => {},
          reject: (e) => controller.error(e),
          stream: true,
          controller,
          encoder,
          generatedTokens: [],
          kvState: null,
          prefillComplete: false,
          queuedAt,
        };

        pendingRequests.set(requestId, pendingReq);
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
      },
    });
  }

  // Non-streaming
  return new Promise((resolve, reject) => {
    const pendingReq: PendingRequest = {
      id: requestId,
      promptTokenIds: inputIds,
      maxTokens,
      temperature,
      resolve,
      reject,
      stream: false,
      generatedTokens: [],
      kvState: null,
      prefillComplete: false,
      queuedAt,
    };

    pendingRequests.set(requestId, pendingReq);
  }) as unknown as Response;
}

async function handleNonStreamingChat(body: ChatCompletionRequest): Promise<Response> {
  const prompt = tokenizer.applyChatTemplate(body.messages);
  const inputIds = tokenizer.encode(prompt);
  const maxTokens = body.max_tokens ?? defaultMaxTokens;
  const temperature = body.temperature ?? 0.7;

  // Reset KV cache before each generation
  engine!.reset();

  const startTime = performance.now();
  const outputIds = await engine!.generate(inputIds, maxTokens, undefined, { temperature });
  const elapsed = performance.now() - startTime;

  // Get only the generated tokens (not the input)
  const generatedIds = outputIds.slice(inputIds.length);
  const generatedText = tokenizer.decode(generatedIds);

  const tokensPerSecond = generatedIds.length / (elapsed / 1000);
  console.log(`Generated ${generatedIds.length} tokens in ${elapsed.toFixed(0)}ms (${tokensPerSecond.toFixed(1)} tok/s)`);

  return Response.json({
    id: `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: modelId,
    choices: [{
      index: 0,
      message: { role: "assistant", content: generatedText },
      finish_reason: "stop",
    }],
    usage: {
      prompt_tokens: inputIds.length,
      completion_tokens: generatedIds.length,
      total_tokens: inputIds.length + generatedIds.length,
    },
  }, {
    headers: { "Access-Control-Allow-Origin": "*" },
  });
}

function handleStreamingChat(body: ChatCompletionRequest): Response {
  const encoder = new TextEncoder();
  const requestId = `chatcmpl-${Date.now()}`;

  const stream = new ReadableStream({
    start: async (controller) => {
      try {
        const prompt = tokenizer.applyChatTemplate(body.messages);
        const inputIds = tokenizer.encode(prompt);
        const maxTokens = body.max_tokens ?? defaultMaxTokens;
        const temperature = body.temperature ?? 0.7;

        // Reset KV cache before each generation
        engine!.reset();

        // Send initial chunk with role
        const initialChunk = {
          id: requestId,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: modelId,
          choices: [{
            index: 0,
            delta: { role: "assistant", content: "" },
            finish_reason: null,
          }],
        };
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(initialChunk)}\n\n`));

        const startTime = performance.now();
        let tokenCount = 0;

        const onToken = (token: number) => {
          const text = tokenizer.decode([token]);
          tokenCount++;

          const chunk = {
            id: requestId,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model: modelId,
            choices: [{
              index: 0,
              delta: { content: text },
              finish_reason: null,
            }],
          };
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
        };

        await engine!.generate(inputIds, maxTokens, onToken, { temperature });

        const elapsed = performance.now() - startTime;
        const tokensPerSecond = tokenCount / (elapsed / 1000);
        console.log(`Streamed ${tokenCount} tokens in ${elapsed.toFixed(0)}ms (${tokensPerSecond.toFixed(1)} tok/s)`);

        // Send final chunk
        const finalChunk = {
          id: requestId,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: modelId,
          choices: [{
            index: 0,
            delta: {},
            finish_reason: "stop",
          }],
        };
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(finalChunk)}\n\n`));
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      } catch (error) {
        console.error("Streaming error:", error);
        controller.error(error);
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      "Access-Control-Allow-Origin": "*",
    },
  });
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.log("╔════════════════════════════════════════════════════════════╗");
  console.log("║  Binfer Server v0.3.0                                      ║");
  console.log("║  OpenAI-compatible API | Continuous Batching               ║");
  console.log("╚════════════════════════════════════════════════════════════╝");
  console.log();

  const args = parseArgs();

  if (!args.model) {
    printUsage();
    process.exit(1);
  }

  modelId = args.model;
  defaultMaxTokens = args.maxTokens;
  useContinuousBatching = args.continuous;

  // Check CUDA
  const cuda = getCudaBackend();
  if (!(await cuda.isAvailable())) {
    console.error("CUDA not available. Build CUDA library: bun run build:cuda");
    process.exit(1);
  }

  const deviceCount = cuda.getDeviceCount();
  console.log(`✓ CUDA: ${deviceCount} GPU(s) available`);
  for (let i = 0; i < deviceCount; i++) {
    const props = cuda.getDeviceProperties(i);
    const memGB = (props.totalMemory / 1024 / 1024 / 1024).toFixed(1);
    console.log(`  └─ GPU ${i}: ${props.name} (${memGB}GB)`);
  }

  // Initialize cuBLAS handles for all devices upfront
  // This avoids lazy initialization issues with multi-GPU
  cuda.initCublas(args.tp);

  console.log();

  // Load model
  console.log(`Loading model: ${args.model} (TP=${args.tp})`);

  let ctx: TensorParallelContext;
  let localPath: string;

  if (args.tp === 1) {
    // Single GPU - use MmapModelLoader directly
    const loader = new MmapModelLoader();
    const model = await loader.load(args.model, {});
    config = model.config;
    localPath = model.localPath;

    // Wrap weights for TP context
    const perDeviceWeights = new Map<number, Map<string, { ptr: bigint; info: any }>>();
    perDeviceWeights.set(0, model.weights);
    ctx = new TensorParallelContext(config, perDeviceWeights, 1);
  } else {
    // Multi-GPU - use TensorParallelModelLoader
    const loader = new TensorParallelModelLoader({ useCache: true });
    const tpModel = await loader.load(args.model, args.tp);
    config = tpModel.config;
    localPath = tpModel.localPath;
    ctx = new TensorParallelContext(config, tpModel.perDeviceWeights, tpModel.worldSize);
  }

  console.log();
  console.log("Model configuration:");
  console.log(`  Type: ${config.modelType}`);
  console.log(`  Layers: ${config.numHiddenLayers}`);
  console.log(`  Hidden: ${config.hiddenSize}`);
  console.log(`  Heads: ${config.numAttentionHeads} (${config.numKeyValueHeads} KV)`);
  console.log(`  Vocab: ${config.vocabSize}`);
  console.log(`  Dtype: ${config.dtype}`);
  if (config.numExperts) {
    console.log(`  MoE: ${config.numExperts} experts (top-${config.numExpertsPerTok})`);
  }

  // Load tokenizer
  console.log();
  console.log("Loading tokenizer...");
  tokenizer = await createTokenizer(localPath);
  console.log(`  Vocab size: ${tokenizer.getVocabSize()}`);

  // Create inference engine
  console.log();
  if (useContinuousBatching) {
    // Auto-compute KV memory from available GPU memory
    // The KV cache is allocated on the device with the least free memory,
    // so we need to use the minimum free memory across all devices
    let minFreeMemory = Infinity;
    for (let device = 0; device < args.tp; device++) {
      cuda.setDevice(device);
      const memInfo = cuda.getMemInfo();
      console.log(`  GPU ${device}: ${(memInfo.free / 1e9).toFixed(2)}GB free / ${(memInfo.total / 1e9).toFixed(2)}GB total`);
      if (memInfo.free < minFreeMemory) {
        minFreeMemory = memInfo.free;
      }
    }
    cuda.setDevice(0);  // Reset to primary device

    // Apply gpu-util fraction to the minimum per-GPU free memory
    // Each GPU needs its own copy of the KV cache for paged attention
    const usableMemory = minFreeMemory * args.gpuUtil;
    const kvMemoryGB = usableMemory / 1e9;
    console.log(`  KV cache: ${(minFreeMemory / 1e9).toFixed(2)}GB × ${(args.gpuUtil * 100).toFixed(0)}% = ${kvMemoryGB.toFixed(2)}GB`);

    console.log("Initializing batched inference engine...");
    batchedEngine = createBatchedEngine(ctx, {
      maxKvMemoryGB: kvMemoryGB,
      maxSeqLen: 4096,
      maxBatchSize: args.maxBatchSize,
    });

    // Start inference loop
    inferenceLoopRunning = true;
    continuousBatchingLoop();

    // Initialize profiling if enabled
    if (args.profile !== "off") {
      profilingEnabled = true;
      profilingDetailed = args.profile === "detailed";
      emaStats = new EmaStats(0.1);
      const mode = profilingDetailed ? "detailed" : "basic";
      console.log(`Profiling enabled (${mode}) - stats will update periodically\n`);

      // Enable engine-level profiling for detailed breakdown
      if (profilingDetailed) {
        batchedEngine.enableProfiling();
      }
    }
  } else {
    console.log("Initializing inference engine...");
    engine = new InferenceEngine(ctx);
  }

  // Start server
  console.log();
  const server = Bun.serve({
    port: args.port,
    fetch: handleRequest,
  });

  console.log("API endpoints:");
  console.log(`  POST http://localhost:${args.port}/v1/chat/completions`);
  console.log(`  GET  http://localhost:${args.port}/v1/models`);
  console.log(`  GET  http://localhost:${args.port}/stats`);
  console.log(`  GET  http://localhost:${args.port}/health`);
  console.log();
  console.log(`Mode: ${useContinuousBatching ? "Continuous Batching" : "Single Request"}`);
  if (useContinuousBatching && batchedEngine) {
    const stats = batchedEngine.getStats();
    console.log(`  Max batch size: ${args.maxBatchSize}`);
    console.log(`  KV cache: ${stats.totalBlocks} blocks (${stats.freeBlocks} free)`);
  }
  console.log();
  console.log("Press Ctrl+C to stop");

  // Start periodic profiling display if enabled
  let profileInterval: Timer | null = null;
  if (args.profile !== "off" && useContinuousBatching) {
    console.log();  // Extra line for profiling display
    profileInterval = setInterval(() => {
      updateProfilingDisplay();
    }, 500);  // Update every 500ms - not too fast to avoid overhead
  }

  // Handle shutdown
  process.on("SIGINT", () => {
    if (profileInterval) {
      clearInterval(profileInterval);
    }
    console.log("\nShutting down...");
    inferenceLoopRunning = false;
    server.stop();
    ctx.dispose();
    tokenizer.stop();
    if (batchedEngine) {
      batchedEngine.dispose();
    }
    process.exit(0);
  });

  // Keep process alive
  await new Promise(() => {});
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
