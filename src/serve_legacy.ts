#!/usr/bin/env bun
// Binfer Server - OpenAI-compatible API server
// Simple single-request inference using InferenceEngine

import { CudaBackend, getCudaBackend } from "./backend/cuda/bindings";
import { MmapModelLoader } from "./loader/mmap_loader";
import { TensorParallelModelLoader } from "./parallel/loader";
import { TensorParallelContext } from "./engine/tp_context";
import { InferenceEngine } from "./engine/engine_legacy";
import { createTokenizer, Tokenizer } from "./model/tokenizer";
import { LlamaConfig } from "./model/config";

interface ServerArgs {
  model: string;
  port: number;
  maxTokens: number;
  tp: number;  // Tensor parallelism
}

function parseArgs(): ServerArgs {
  const args = process.argv.slice(2);
  const result: ServerArgs = {
    model: "",
    port: 8000,
    maxTokens: 256,
    tp: 1,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--port" && args[i + 1]) {
      result.port = parseInt(args[++i], 10);
    } else if (arg === "--max-tokens" && args[i + 1]) {
      result.maxTokens = parseInt(args[++i], 10);
    } else if (arg === "--tp" && args[i + 1]) {
      result.tp = parseInt(args[++i], 10);
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

Examples:
  bun run src/serve.ts openai/gpt-oss-20b
  bun run src/serve.ts openai/gpt-oss-120b --tp 2 --port 8080
`);
}

// Global state
let engine: InferenceEngine;
let tokenizer: Tokenizer;
let config: LlamaConfig;
let modelId: string;
let defaultMaxTokens: number;

// Request queue for serializing inference
const requestQueue: Array<{
  resolve: (response: Response) => void;
  reject: (error: Error) => void;
  handler: () => Promise<Response>;
}> = [];
let processing = false;

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

// Chat completion types
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

// Handle requests
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
      const body = await req.json() as ChatCompletionRequest;

      if (body.stream) {
        return handleStreamingChat(body);
      }

      return queueRequest(() => handleNonStreamingChat(body));
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

async function handleNonStreamingChat(body: ChatCompletionRequest): Promise<Response> {
  const prompt = tokenizer.applyChatTemplate(body.messages);
  const inputIds = tokenizer.encode(prompt);
  const maxTokens = body.max_tokens ?? defaultMaxTokens;
  const temperature = body.temperature ?? 0.7;

  // Reset KV cache before each generation
  engine.reset();

  const startTime = performance.now();
  const outputIds = await engine.generate(inputIds, maxTokens, undefined, { temperature });
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
        engine.reset();

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

        await engine.generate(inputIds, maxTokens, onToken, { temperature });

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

async function main() {
  console.log("╔════════════════════════════════════════════════════════════╗");
  console.log("║  Binfer Server v0.2.0                                      ║");
  console.log("║  OpenAI-compatible API | Tensor Parallel Inference         ║");
  console.log("╚════════════════════════════════════════════════════════════╝");
  console.log();

  const args = parseArgs();

  if (!args.model) {
    printUsage();
    process.exit(1);
  }

  modelId = args.model;
  defaultMaxTokens = args.maxTokens;

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
  console.log("Initializing inference engine...");
  engine = new InferenceEngine(ctx);

  // Start server
  console.log();
  const server = Bun.serve({
    port: args.port,
    fetch: handleRequest,
  });

  console.log("API endpoints:");
  console.log(`  POST http://localhost:${args.port}/v1/chat/completions`);
  console.log(`  GET  http://localhost:${args.port}/v1/models`);
  console.log(`  GET  http://localhost:${args.port}/health`);
  console.log();
  console.log("Press Ctrl+C to stop");

  // Handle shutdown
  process.on("SIGINT", () => {
    console.log("\nShutting down...");
    server.stop();
    ctx.dispose();
    tokenizer.stop();
    process.exit(0);
  });

  // Keep process alive
  await new Promise(() => {});
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
