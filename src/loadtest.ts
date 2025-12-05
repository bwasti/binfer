#!/usr/bin/env bun
// Load test script for binfer server
// Measures total tokens generated per second (TPGS) under concurrent load

interface LoadTestArgs {
  url: string;
  concurrency: number;
  totalRequests: number;
  maxTokens: number;
  prompt: string;
}

function parseArgs(): LoadTestArgs {
  const args = process.argv.slice(2);
  const result: LoadTestArgs = {
    url: "http://localhost:8000",
    concurrency: 8,
    totalRequests: 32,
    maxTokens: 128,
    prompt: "Write a short story about a robot learning to paint.",
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--url" && args[i + 1]) {
      result.url = args[++i];
    } else if ((arg === "-c" || arg === "--concurrency") && args[i + 1]) {
      result.concurrency = parseInt(args[++i], 10);
    } else if ((arg === "-n" || arg === "--requests") && args[i + 1]) {
      result.totalRequests = parseInt(args[++i], 10);
    } else if (arg === "--max-tokens" && args[i + 1]) {
      result.maxTokens = parseInt(args[++i], 10);
    } else if ((arg === "-p" || arg === "--prompt") && args[i + 1]) {
      result.prompt = args[++i];
    } else if (arg === "--help" || arg === "-h") {
      printUsage();
      process.exit(0);
    }
  }

  return result;
}

function printUsage() {
  console.log(`
Load Test for Binfer Server

Usage: bun run src/loadtest.ts [options]

Options:
  --url <url>              Server URL (default: http://localhost:8000)
  -c, --concurrency <n>    Concurrent requests (default: 8)
  -n, --requests <n>       Total requests to make (default: 32)
  --max-tokens <n>         Max tokens per request (default: 128)
  -p, --prompt <text>      Prompt to use (default: story prompt)

Examples:
  bun run src/loadtest.ts -c 16 -n 64
  bun run src/loadtest.ts --concurrency 32 --max-tokens 256
`);
}

interface RequestResult {
  success: boolean;
  tokensGenerated: number;
  latencyMs: number;
  ttftMs: number;  // Time to first token
  error?: string;
}

async function makeRequest(
  url: string,
  prompt: string,
  maxTokens: number
): Promise<RequestResult> {
  const startTime = performance.now();
  let ttftMs = 0;
  let tokensGenerated = 0;

  try {
    const response = await fetch(`${url}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "default",
        messages: [{ role: "user", content: prompt }],
        max_tokens: maxTokens,
        stream: true,
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      return {
        success: false,
        tokensGenerated: 0,
        latencyMs: performance.now() - startTime,
        ttftMs: 0,
        error: `HTTP ${response.status}: ${text}`,
      };
    }

    const reader = response.body?.getReader();
    if (!reader) {
      return {
        success: false,
        tokensGenerated: 0,
        latencyMs: performance.now() - startTime,
        ttftMs: 0,
        error: "No response body",
      };
    }

    const decoder = new TextDecoder();
    let firstToken = true;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split("\n");

      for (const line of lines) {
        if (line.startsWith("data: ") && line !== "data: [DONE]") {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.choices?.[0]?.delta?.content) {
              if (firstToken) {
                ttftMs = performance.now() - startTime;
                firstToken = false;
              }
              tokensGenerated++;
            }
          } catch {
            // Skip parse errors for partial chunks
          }
        }
      }
    }

    return {
      success: true,
      tokensGenerated,
      latencyMs: performance.now() - startTime,
      ttftMs,
    };
  } catch (error) {
    return {
      success: false,
      tokensGenerated: 0,
      latencyMs: performance.now() - startTime,
      ttftMs: 0,
      error: String(error),
    };
  }
}

async function runLoadTest(args: LoadTestArgs): Promise<void> {
  console.log(`\nLoad Test Configuration:`);
  console.log(`  URL: ${args.url}`);
  console.log(`  Concurrency: ${args.concurrency}`);
  console.log(`  Total Requests: ${args.totalRequests}`);
  console.log(`  Max Tokens: ${args.maxTokens}`);
  console.log();

  // Check server health first
  try {
    const health = await fetch(`${args.url}/health`);
    if (!health.ok) {
      console.error(`Server not healthy: ${await health.text()}`);
      process.exit(1);
    }
    console.log("Server health: OK\n");
  } catch (error) {
    console.error(`Cannot connect to server: ${error}`);
    process.exit(1);
  }

  const results: RequestResult[] = [];
  let completed = 0;
  let inFlight = 0;
  let nextRequest = 0;

  const startTime = performance.now();

  // Progress indicator
  const updateProgress = () => {
    const elapsed = (performance.now() - startTime) / 1000;
    const totalTokens = results.reduce((sum, r) => sum + r.tokensGenerated, 0);
    const tpgs = elapsed > 0 ? totalTokens / elapsed : 0;
    process.stdout.write(
      `\rProgress: ${completed}/${args.totalRequests} | In-flight: ${inFlight} | Tokens: ${totalTokens} | TPGS: ${tpgs.toFixed(1)}    `
    );
  };

  const progressInterval = setInterval(updateProgress, 100);

  // Worker function
  const worker = async () => {
    while (nextRequest < args.totalRequests) {
      const requestId = nextRequest++;
      inFlight++;

      const result = await makeRequest(args.url, args.prompt, args.maxTokens);
      results.push(result);

      inFlight--;
      completed++;

      if (!result.success) {
        console.error(`\nRequest ${requestId} failed: ${result.error}`);
      }
    }
  };

  // Start workers
  const workers: Promise<void>[] = [];
  for (let i = 0; i < args.concurrency; i++) {
    workers.push(worker());
  }

  await Promise.all(workers);

  clearInterval(progressInterval);
  const totalTime = (performance.now() - startTime) / 1000;

  // Calculate stats
  const successfulResults = results.filter((r) => r.success);
  const totalTokens = successfulResults.reduce((sum, r) => sum + r.tokensGenerated, 0);
  const avgLatency = successfulResults.reduce((sum, r) => sum + r.latencyMs, 0) / successfulResults.length;
  const avgTtft = successfulResults.reduce((sum, r) => sum + r.ttftMs, 0) / successfulResults.length;
  const avgTokensPerRequest = totalTokens / successfulResults.length;

  // Latency percentiles
  const latencies = successfulResults.map((r) => r.latencyMs).sort((a, b) => a - b);
  const p50 = latencies[Math.floor(latencies.length * 0.5)] || 0;
  const p90 = latencies[Math.floor(latencies.length * 0.9)] || 0;
  const p99 = latencies[Math.floor(latencies.length * 0.99)] || 0;

  console.log(`\n\n${"=".repeat(50)}`);
  console.log("LOAD TEST RESULTS");
  console.log("=".repeat(50));
  console.log();
  console.log(`Requests:     ${results.length} total, ${successfulResults.length} successful, ${results.length - successfulResults.length} failed`);
  console.log(`Duration:     ${totalTime.toFixed(2)}s`);
  console.log(`Throughput:   ${(successfulResults.length / totalTime).toFixed(2)} req/s`);
  console.log();
  console.log(`Tokens:       ${totalTokens} total`);
  console.log(`TPGS:         ${(totalTokens / totalTime).toFixed(1)} tokens/s  ⬅️  KEY METRIC`);
  console.log(`Per Request:  ${avgTokensPerRequest.toFixed(1)} tokens avg`);
  console.log();
  console.log(`Latency (ms):`);
  console.log(`  Mean:       ${avgLatency.toFixed(0)}`);
  console.log(`  P50:        ${p50.toFixed(0)}`);
  console.log(`  P90:        ${p90.toFixed(0)}`);
  console.log(`  P99:        ${p99.toFixed(0)}`);
  console.log();
  console.log(`TTFT (ms):    ${avgTtft.toFixed(0)} avg (time to first token)`);
  console.log("=".repeat(50));
}

// Run
const args = parseArgs();
runLoadTest(args).catch(console.error);
