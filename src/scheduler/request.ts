// Request state management for continuous batching

export enum RequestStatus {
  PENDING = "pending",       // Waiting to be scheduled
  RUNNING = "running",       // Currently being processed
  PREEMPTED = "preempted",   // Swapped out due to memory pressure
  COMPLETED = "completed",   // Generation finished (EOS or max tokens)
  FAILED = "failed",         // Error during generation
}

export interface GenerationParams {
  maxNewTokens: number;
  temperature: number;
  topK: number;
  topP: number;
  stopTokenIds: number[];
  streamCallback?: (token: number, text: string) => void;
}

export const DEFAULT_GENERATION_PARAMS: GenerationParams = {
  maxNewTokens: 256,
  temperature: 0.7,
  topK: 50,
  topP: 0.9,
  stopTokenIds: [],
};

/**
 * Represents a single generation request.
 */
export class Request {
  readonly id: string;
  readonly prompt: string;
  readonly promptTokenIds: number[];
  readonly params: GenerationParams;
  readonly createdAt: number;

  status: RequestStatus;
  generatedTokenIds: number[];
  numPrefillTokens: number;
  prefillComplete: boolean;

  // For streaming
  private onToken?: (token: number, text: string) => void;

  constructor(
    id: string,
    prompt: string,
    promptTokenIds: number[],
    params: Partial<GenerationParams> = {}
  ) {
    this.id = id;
    this.prompt = prompt;
    this.promptTokenIds = promptTokenIds;
    this.params = { ...DEFAULT_GENERATION_PARAMS, ...params };
    this.createdAt = Date.now();

    this.status = RequestStatus.PENDING;
    this.generatedTokenIds = [];
    this.numPrefillTokens = 0;
    this.prefillComplete = false;
    this.onToken = params.streamCallback;
  }

  /**
   * Get total number of tokens (prompt + generated).
   */
  get totalTokens(): number {
    return this.promptTokenIds.length + this.generatedTokenIds.length;
  }

  /**
   * Get number of tokens processed so far.
   */
  get processedTokens(): number {
    if (!this.prefillComplete) {
      return this.numPrefillTokens;
    }
    return this.promptTokenIds.length + this.generatedTokenIds.length;
  }

  /**
   * Check if generation is complete.
   */
  isFinished(): boolean {
    return (
      this.status === RequestStatus.COMPLETED ||
      this.status === RequestStatus.FAILED
    );
  }

  /**
   * Check if we've hit max tokens.
   */
  hasReachedMaxTokens(): boolean {
    return this.generatedTokenIds.length >= this.params.maxNewTokens;
  }

  /**
   * Check if the last token is a stop token.
   */
  hasStopToken(): boolean {
    if (this.generatedTokenIds.length === 0) return false;
    const lastToken = this.generatedTokenIds[this.generatedTokenIds.length - 1];
    return this.params.stopTokenIds.includes(lastToken);
  }

  /**
   * Add a generated token.
   */
  addToken(token: number, text: string): void {
    this.generatedTokenIds.push(token);
    if (this.onToken) {
      this.onToken(token, text);
    }
  }

  /**
   * Mark prefill as complete.
   */
  markPrefillComplete(): void {
    this.prefillComplete = true;
    this.numPrefillTokens = this.promptTokenIds.length;
  }

  /**
   * Mark request as completed.
   */
  markCompleted(): void {
    this.status = RequestStatus.COMPLETED;
  }

  /**
   * Mark request as failed.
   */
  markFailed(error: string): void {
    this.status = RequestStatus.FAILED;
    console.error(`Request ${this.id} failed: ${error}`);
  }
}

/**
 * Queue of pending requests with priority support.
 */
export class RequestQueue {
  private queue: Request[] = [];

  /**
   * Add a request to the queue.
   */
  add(request: Request): void {
    request.status = RequestStatus.PENDING;
    this.queue.push(request);
  }

  /**
   * Get the next request to process.
   * Uses FIFO for now, could add priority later.
   */
  pop(): Request | undefined {
    return this.queue.shift();
  }

  /**
   * Peek at the next request without removing it.
   */
  peek(): Request | undefined {
    return this.queue[0];
  }

  /**
   * Get number of pending requests.
   */
  get length(): number {
    return this.queue.length;
  }

  /**
   * Check if queue is empty.
   */
  isEmpty(): boolean {
    return this.queue.length === 0;
  }

  /**
   * Get all pending requests.
   */
  getAll(): Request[] {
    return [...this.queue];
  }
}

/**
 * Generate a unique request ID.
 */
export function generateRequestId(): string {
  return `req-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}
