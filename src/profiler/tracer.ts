// GPU Tracer - Captures detailed timeline for analysis
// Outputs JSON trace format compatible with chrome://tracing and Claude Code analysis

import { getCudaBackend, CudaBackend } from "../backend/cuda/bindings";

export interface TraceEvent {
  name: string;
  cat: string;           // Category (e.g., "gpu", "cpu", "memory")
  ph: "B" | "E" | "X";   // Phase: Begin, End, or Complete
  ts: number;            // Timestamp in microseconds
  dur?: number;          // Duration in microseconds (for "X" events)
  pid: number;           // Process ID (we use 0 for GPU, 1 for CPU)
  tid: number;           // Thread/stream ID
  args?: Record<string, unknown>;
}

export interface TraceFile {
  traceEvents: TraceEvent[];
  metadata: {
    model: string;
    tokensGenerated: number;
    totalTimeMs: number;
    tokensPerSecond: number;
    deviceName?: string;
    timestamp: string;
  };
  summary: {
    totalGpuTimeMs: number;
    totalIdleTimeMs: number;
    gpuUtilization: number;
    operationBreakdown: Record<string, { totalMs: number; count: number; avgMs: number }>;
    slowestOperations: Array<{ name: string; durationMs: number; layer?: number }>;
  };
}

interface PendingSpan {
  name: string;
  category: string;
  cpuStartUs: number;
  gpuStartEvent: bigint;
  layer?: number;
  args?: Record<string, unknown>;
}

/**
 * GPU Tracer for detailed performance analysis.
 *
 * Captures both CPU and GPU timestamps to identify:
 * - GPU kernel execution times
 * - CPU overhead between kernels
 * - Memory transfer times
 * - Idle gaps where GPU is waiting
 */
export class GpuTracer {
  private cuda: CudaBackend;
  private enabled: boolean = false;
  private events: TraceEvent[] = [];
  private pendingSpans: Map<string, PendingSpan> = new Map();
  private eventPool: bigint[] = [];
  private baseTimeUs: number = 0;
  private gpuBaseEvent: bigint | null = null;
  private allGpuEvents: Array<{ event: bigint; cpuTimeUs: number }> = [];

  constructor() {
    this.cuda = getCudaBackend();
  }

  enable(): void {
    this.enabled = true;
    this.events = [];
    this.pendingSpans.clear();
    this.allGpuEvents = [];

    // Record base time
    this.baseTimeUs = performance.now() * 1000;

    // Record GPU base event
    this.gpuBaseEvent = this.getEvent();
    this.cuda.eventRecord(this.gpuBaseEvent);
    this.cuda.synchronize();
  }

  disable(): void {
    this.enabled = false;
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  private getEvent(): bigint {
    if (this.eventPool.length > 0) {
      return this.eventPool.pop()!;
    }
    return this.cuda.eventCreate();
  }

  private returnEvent(event: bigint): void {
    this.eventPool.push(event);
  }

  private nowUs(): number {
    return performance.now() * 1000 - this.baseTimeUs;
  }

  /**
   * Start a traced span.
   */
  begin(name: string, category: string = "gpu", layer?: number, args?: Record<string, unknown>): void {
    if (!this.enabled) return;

    const cpuStartUs = this.nowUs();
    const gpuStartEvent = this.getEvent();
    this.cuda.eventRecord(gpuStartEvent);

    const key = layer !== undefined ? `${name}_L${layer}` : name;
    this.pendingSpans.set(key, {
      name,
      category,
      cpuStartUs,
      gpuStartEvent,
      layer,
      args,
    });
  }

  /**
   * End a traced span.
   */
  end(name: string, layer?: number): void {
    if (!this.enabled) return;

    const key = layer !== undefined ? `${name}_L${layer}` : name;
    const span = this.pendingSpans.get(key);
    if (!span) return;

    const cpuEndUs = this.nowUs();
    const gpuEndEvent = this.getEvent();
    this.cuda.eventRecord(gpuEndEvent);

    // Store for later resolution
    this.allGpuEvents.push(
      { event: span.gpuStartEvent, cpuTimeUs: span.cpuStartUs },
      { event: gpuEndEvent, cpuTimeUs: cpuEndUs }
    );

    // Create CPU timeline event (shows when CPU issued the work)
    this.events.push({
      name: span.layer !== undefined ? `${span.name} [L${span.layer}]` : span.name,
      cat: span.category,
      ph: "X",
      ts: span.cpuStartUs,
      dur: cpuEndUs - span.cpuStartUs,
      pid: 1,  // CPU
      tid: 0,
      args: {
        ...span.args,
        layer: span.layer,
        type: "cpu_issue",
      },
    });

    this.pendingSpans.delete(key);
  }

  /**
   * Add a simple instant event.
   */
  instant(name: string, category: string = "marker", args?: Record<string, unknown>): void {
    if (!this.enabled) return;

    this.events.push({
      name,
      cat: category,
      ph: "X",
      ts: this.nowUs(),
      dur: 1,
      pid: 1,
      tid: 0,
      args,
    });
  }

  /**
   * Finalize and generate the trace file.
   */
  finalize(metadata: {
    model: string;
    tokensGenerated: number;
    totalTimeMs: number;
    deviceName?: string;
  }): TraceFile {
    // Synchronize GPU to get accurate timings
    this.cuda.synchronize();

    // Resolve GPU event timings relative to base
    const gpuEvents: TraceEvent[] = [];
    const operationStats: Map<string, { totalMs: number; count: number; durations: number[] }> = new Map();

    // Process events pairwise (start, end)
    for (let i = 0; i < this.events.length; i++) {
      const cpuEvent = this.events[i];
      if (cpuEvent.args?.type !== "cpu_issue") continue;

      // Find corresponding GPU events
      const gpuStartIdx = i * 2;
      const gpuEndIdx = gpuStartIdx + 1;

      if (gpuEndIdx < this.allGpuEvents.length) {
        const startEvt = this.allGpuEvents[gpuStartIdx];
        const endEvt = this.allGpuEvents[gpuEndIdx];

        // Get GPU timing relative to base
        const gpuStartMs = this.cuda.eventElapsedTime(this.gpuBaseEvent!, startEvt.event);
        const gpuEndMs = this.cuda.eventElapsedTime(this.gpuBaseEvent!, endEvt.event);
        const durationMs = gpuEndMs - gpuStartMs;

        // Add GPU event
        gpuEvents.push({
          name: cpuEvent.name,
          cat: "gpu",
          ph: "X",
          ts: gpuStartMs * 1000,  // Convert to microseconds
          dur: durationMs * 1000,
          pid: 0,  // GPU
          tid: 0,
          args: {
            ...cpuEvent.args,
            type: "gpu_exec",
            durationMs: durationMs.toFixed(3),
          },
        });

        // Track stats
        const baseName = cpuEvent.name.replace(/ \[L\d+\]$/, "");
        const stats = operationStats.get(baseName) || { totalMs: 0, count: 0, durations: [] };
        stats.totalMs += durationMs;
        stats.count += 1;
        stats.durations.push(durationMs);
        operationStats.set(baseName, stats);

        // Return events to pool
        this.returnEvent(startEvt.event);
        this.returnEvent(endEvt.event);
      }
    }

    // Calculate summary statistics
    let totalGpuTimeMs = 0;
    const operationBreakdown: Record<string, { totalMs: number; count: number; avgMs: number }> = {};

    for (const [name, stats] of operationStats) {
      totalGpuTimeMs += stats.totalMs;
      operationBreakdown[name] = {
        totalMs: stats.totalMs,
        count: stats.count,
        avgMs: stats.totalMs / stats.count,
      };
    }

    // Find slowest individual operations
    const slowestOperations = gpuEvents
      .filter(e => e.dur !== undefined)
      .map(e => ({
        name: e.name,
        durationMs: e.dur! / 1000,
        layer: e.args?.layer as number | undefined,
      }))
      .sort((a, b) => b.durationMs - a.durationMs)
      .slice(0, 20);

    // Calculate idle time (gaps between GPU operations)
    const sortedGpuEvents = [...gpuEvents].sort((a, b) => a.ts - b.ts);
    let totalIdleTimeMs = 0;
    for (let i = 1; i < sortedGpuEvents.length; i++) {
      const prev = sortedGpuEvents[i - 1];
      const curr = sortedGpuEvents[i];
      const gapUs = curr.ts - (prev.ts + (prev.dur || 0));
      if (gapUs > 0) {
        totalIdleTimeMs += gapUs / 1000;
      }
    }

    const gpuUtilization = totalGpuTimeMs / (totalGpuTimeMs + totalIdleTimeMs);

    // Combine all events
    const allEvents = [...this.events, ...gpuEvents];

    // Cleanup
    if (this.gpuBaseEvent) {
      this.returnEvent(this.gpuBaseEvent);
      this.gpuBaseEvent = null;
    }

    return {
      traceEvents: allEvents,
      metadata: {
        model: metadata.model,
        tokensGenerated: metadata.tokensGenerated,
        totalTimeMs: metadata.totalTimeMs,
        tokensPerSecond: metadata.tokensGenerated / (metadata.totalTimeMs / 1000),
        deviceName: metadata.deviceName,
        timestamp: new Date().toISOString(),
      },
      summary: {
        totalGpuTimeMs,
        totalIdleTimeMs,
        gpuUtilization,
        operationBreakdown,
        slowestOperations,
      },
    };
  }

  /**
   * Dispose of all resources.
   */
  dispose(): void {
    for (const event of this.eventPool) {
      this.cuda.eventDestroy(event);
    }
    this.eventPool = [];
    if (this.gpuBaseEvent) {
      this.cuda.eventDestroy(this.gpuBaseEvent);
      this.gpuBaseEvent = null;
    }
  }
}

// Singleton tracer instance
let _tracer: GpuTracer | null = null;

export function getTracer(): GpuTracer {
  if (!_tracer) {
    _tracer = new GpuTracer();
  }
  return _tracer;
}

/**
 * Write trace to file with analysis-friendly format.
 */
export async function writeTrace(trace: TraceFile, path: string): Promise<void> {
  const output = {
    // Chrome tracing format
    traceEvents: trace.traceEvents,

    // Our metadata and summary for Claude Code analysis
    _binfer: {
      metadata: trace.metadata,
      summary: trace.summary,

      // Analysis hints for optimization
      analysisNotes: generateAnalysisNotes(trace),
    },
  };

  await Bun.write(path, JSON.stringify(output, null, 2));
}

function generateAnalysisNotes(trace: TraceFile): string[] {
  const notes: string[] = [];
  const { summary } = trace;

  // GPU utilization check
  if (summary.gpuUtilization < 0.8) {
    notes.push(
      `GPU utilization is ${(summary.gpuUtilization * 100).toFixed(1)}%. ` +
      `${summary.totalIdleTimeMs.toFixed(1)}ms of idle time detected. ` +
      `Look for CPU bottlenecks or synchronization issues.`
    );
  }

  // Find dominant operations
  const sortedOps = Object.entries(summary.operationBreakdown)
    .sort((a, b) => b[1].totalMs - a[1].totalMs);

  if (sortedOps.length > 0) {
    const [topOp, topStats] = sortedOps[0];
    const pct = (topStats.totalMs / summary.totalGpuTimeMs * 100).toFixed(1);
    notes.push(
      `"${topOp}" is the dominant operation at ${pct}% of GPU time ` +
      `(${topStats.count} calls, avg ${topStats.avgMs.toFixed(2)}ms each).`
    );
  }

  // Check for slow individual operations
  if (summary.slowestOperations.length > 0) {
    const slowest = summary.slowestOperations[0];
    notes.push(
      `Slowest single operation: "${slowest.name}" at ${slowest.durationMs.toFixed(2)}ms` +
      (slowest.layer !== undefined ? ` (layer ${slowest.layer})` : "")
    );
  }

  // Memory-bound vs compute-bound hints
  const attentionTime = summary.operationBreakdown["attention"]?.totalMs || 0;
  const mlpTime = (summary.operationBreakdown["mlp_gate_up"]?.totalMs || 0) +
                  (summary.operationBreakdown["mlp_down"]?.totalMs || 0);

  if (attentionTime > mlpTime * 1.5) {
    notes.push(
      `Attention-bound: attention takes ${attentionTime.toFixed(1)}ms vs ` +
      `MLP ${mlpTime.toFixed(1)}ms. Consider optimizing attention or using Flash Attention.`
    );
  } else if (mlpTime > attentionTime * 1.5) {
    notes.push(
      `MLP-bound: MLP takes ${mlpTime.toFixed(1)}ms vs attention ${attentionTime.toFixed(1)}ms. ` +
      `Consider GEMM optimization or reduced precision.`
    );
  }

  return notes;
}
