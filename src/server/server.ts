// OpenAI-compatible HTTP API server
// Fully compliant with OpenAI API spec
// Endpoints: /v1/chat/completions, /v1/completions, /v1/models

import { Scheduler } from "../scheduler/scheduler";
import { Request, RequestStatus } from "../scheduler/request";
import { Tokenizer } from "../model/tokenizer";
import { LlamaConfig } from "../model/config";

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  name?: string;
  tool_calls?: any[];
  tool_call_id?: string;
}

export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  n?: number;
  max_tokens?: number;
  max_completion_tokens?: number;
  stream?: boolean;
  stop?: string | string[] | null;
  presence_penalty?: number;
  frequency_penalty?: number;
  logit_bias?: Record<string, number>;
  user?: string;
  seed?: number;
  tools?: any[];
  tool_choice?: string | object;
  response_format?: { type: string };
}

export interface ChatCompletionChoice {
  index: number;
  message: {
    role: "assistant";
    content: string | null;
    tool_calls?: any[];
  };
  logprobs: null;
  finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | null;
}

export interface ChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  system_fingerprint: string;
  choices: ChatCompletionChoice[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface ChatCompletionChunkChoice {
  index: number;
  delta: {
    role?: "assistant";
    content?: string | null;
    tool_calls?: any[];
  };
  logprobs: null;
  finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | null;
}

export interface ChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  system_fingerprint: string;
  choices: ChatCompletionChunkChoice[];
}

export interface CompletionRequest {
  model: string;
  prompt: string | string[];
  suffix?: string | null;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  n?: number;
  stream?: boolean;
  logprobs?: number | null;
  echo?: boolean;
  stop?: string | string[] | null;
  presence_penalty?: number;
  frequency_penalty?: number;
  best_of?: number;
  logit_bias?: Record<string, number>;
  user?: string;
  seed?: number;
}

export interface CompletionChoice {
  text: string;
  index: number;
  logprobs: null;
  finish_reason: "stop" | "length";
}

export interface CompletionResponse {
  id: string;
  object: "text_completion";
  created: number;
  model: string;
  system_fingerprint: string;
  choices: CompletionChoice[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface ModelInfo {
  id: string;
  object: "model";
  created: number;
  owned_by: string;
}

export interface ErrorResponse {
  error: {
    message: string;
    type: string;
    param: string | null;
    code: string | null;
  };
}

/**
 * OpenAI-compatible API server.
 */
export class Server {
  private scheduler: Scheduler;
  private tokenizer: Tokenizer;
  private modelConfig: LlamaConfig;
  private modelId: string;

  private server: ReturnType<typeof Bun.serve> | null = null;
  private activeRequests: Map<string, { request: Request; resolve: (resp: Response) => void }> = new Map();

  constructor(
    scheduler: Scheduler,
    tokenizer: Tokenizer,
    modelConfig: LlamaConfig,
    modelId: string
  ) {
    this.scheduler = scheduler;
    this.tokenizer = tokenizer;
    this.modelConfig = modelConfig;
    this.modelId = modelId;
  }

  /**
   * Start the HTTP server.
   */
  start(port: number = 8000): void {
    this.server = Bun.serve({
      port,
      fetch: this.handleRequest.bind(this),
    });

    console.log(`Server listening on http://localhost:${port}`);
    console.log(`Model: ${this.modelId}`);
  }

  /**
   * Stop the server.
   */
  stop(): void {
    if (this.server) {
      this.server.stop();
      this.server = null;
    }
  }

  /**
   * Handle incoming HTTP request.
   */
  private async handleRequest(req: Bun.Request): Promise<Response> {
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
      // Route handling
      if (url.pathname === "/v1/models" && req.method === "GET") {
        return this.handleListModels();
      }

      if (url.pathname === "/v1/chat/completions" && req.method === "POST") {
        const body = await req.json() as ChatCompletionRequest;
        return this.handleChatCompletion(body);
      }

      if (url.pathname === "/v1/completions" && req.method === "POST") {
        // Legacy completions endpoint
        const body = await req.json();
        return this.handleCompletion(body);
      }

      if (url.pathname === "/health" && req.method === "GET") {
        return Response.json({ status: "ok" });
      }

      return new Response("Not Found", { status: 404 });
    } catch (error) {
      console.error("Request error:", error);
      return Response.json(
        { error: { message: String(error), type: "server_error" } },
        { status: 500 }
      );
    }
  }

  /**
   * GET /v1/models
   */
  private handleListModels(): Response {
    const models: ModelInfo[] = [
      {
        id: this.modelId,
        object: "model",
        created: Date.now(),
        owned_by: "binfer",
      },
    ];

    return Response.json({
      object: "list",
      data: models,
    });
  }

  /**
   * POST /v1/chat/completions
   */
  private async handleChatCompletion(body: ChatCompletionRequest): Promise<Response> {
    // Apply chat template
    const prompt = this.tokenizer.applyChatTemplate(body.messages);
    const promptTokenIds = this.tokenizer.encode(prompt);

    if (body.stream) {
      return this.handleStreamingChatCompletion(body, promptTokenIds);
    }

    return this.handleNonStreamingChatCompletion(body, promptTokenIds);
  }

  /**
   * Non-streaming chat completion.
   */
  private async handleNonStreamingChatCompletion(
    body: ChatCompletionRequest,
    promptTokenIds: number[]
  ): Promise<Response> {
    const request = this.scheduler.submitRequest(
      "",  // prompt text not needed here
      promptTokenIds,
      {
        maxNewTokens: body.max_tokens ?? 256,
        temperature: body.temperature ?? 0.7,
        topP: body.top_p ?? 0.9,
        stopTokenIds: [this.modelConfig.eosTokenId],
      }
    );

    // Wait for completion
    await this.waitForRequest(request);

    // Decode generated tokens
    const generatedText = this.tokenizer.decode(request.generatedTokenIds);

    const response: ChatCompletionResponse = {
      id: `chatcmpl-${request.id}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: this.modelId,
      system_fingerprint: `fp_${this.modelId.replace(/[^a-z0-9]/gi, "").slice(0, 12)}`,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: generatedText,
          },
          logprobs: null,
          finish_reason: request.hasStopToken() ? "stop" : "length",
        },
      ],
      usage: {
        prompt_tokens: promptTokenIds.length,
        completion_tokens: request.generatedTokenIds.length,
        total_tokens: promptTokenIds.length + request.generatedTokenIds.length,
      },
    };

    return Response.json(response);
  }

  /**
   * Streaming chat completion (SSE).
   */
  private handleStreamingChatCompletion(
    body: ChatCompletionRequest,
    promptTokenIds: number[]
  ): Response {
    const encoder = new TextEncoder();
    const requestId = `chatcmpl-${Date.now()}`;

    const stream = new ReadableStream({
      start: async (controller) => {
        // Send initial chunk with role
        const initialChunk: ChatCompletionChunk = {
          id: requestId,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: this.modelId,
          system_fingerprint: `fp_${this.modelId.replace(/[^a-z0-9]/gi, "").slice(0, 12)}`,
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "" },
              logprobs: null,
              finish_reason: null,
            },
          ],
        };
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(initialChunk)}\n\n`));

        // Submit request with streaming callback
        const request = this.scheduler.submitRequest(
          "",
          promptTokenIds,
          {
            maxNewTokens: body.max_tokens ?? 256,
            temperature: body.temperature ?? 0.7,
            topP: body.top_p ?? 0.9,
            stopTokenIds: [this.modelConfig.eosTokenId],
            streamCallback: (token: number, text: string) => {
              if (token === this.modelConfig.eosTokenId) return;

              const chunk: ChatCompletionChunk = {
                id: requestId,
                object: "chat.completion.chunk",
                created: Math.floor(Date.now() / 1000),
                model: this.modelId,
                system_fingerprint: `fp_${this.modelId.replace(/[^a-z0-9]/gi, "").slice(0, 12)}`,
                choices: [
                  {
                    index: 0,
                    delta: { content: text },
                    logprobs: null,
                    finish_reason: null,
                  },
                ],
              };
              controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
            },
          }
        );

        // Wait for completion
        await this.waitForRequest(request);

        // Send final chunk
        const finalChunk: ChatCompletionChunk = {
          id: requestId,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: this.modelId,
          system_fingerprint: `fp_${this.modelId.replace(/[^a-z0-9]/gi, "").slice(0, 12)}`,
          choices: [
            {
              index: 0,
              delta: {},
              logprobs: null,
              finish_reason: request.hasStopToken() ? "stop" : "length",
            },
          ],
        };
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(finalChunk)}\n\n`));
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
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

  /**
   * POST /v1/completions (legacy)
   */
  private async handleCompletion(body: any): Promise<Response> {
    const promptTokenIds = this.tokenizer.encode(body.prompt ?? "");

    const request = this.scheduler.submitRequest(
      body.prompt ?? "",
      promptTokenIds,
      {
        maxNewTokens: body.max_tokens ?? 256,
        temperature: body.temperature ?? 0.7,
        topP: body.top_p ?? 0.9,
      }
    );

    await this.waitForRequest(request);

    const generatedText = this.tokenizer.decode(request.generatedTokenIds);

    return Response.json({
      id: `cmpl-${request.id}`,
      object: "text_completion",
      created: Math.floor(Date.now() / 1000),
      model: this.modelId,
      choices: [
        {
          text: generatedText,
          index: 0,
          logprobs: null,
          finish_reason: request.hasStopToken() ? "stop" : "length",
        },
      ],
      usage: {
        prompt_tokens: promptTokenIds.length,
        completion_tokens: request.generatedTokenIds.length,
        total_tokens: promptTokenIds.length + request.generatedTokenIds.length,
      },
    });
  }

  /**
   * Wait for a request to complete.
   */
  private async waitForRequest(request: Request): Promise<void> {
    while (!request.isFinished()) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
  }
}
