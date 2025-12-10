# binfer (Bun + infer)

Experimental Bun-based snappy LLM inference engine in TypeScript and CUDA.
Currently supports gpt-oss and qwen huggingface models.

![Screen Recording 2025-12-10 at 2 12 45 PM](https://github.com/user-attachments/assets/9c4a228f-4751-46d4-b863-b32e71c3b4db)


## Requirements

- CUDA 12.x
- Bun 1.x
- NVIDIA GPU (H100 recommended, A100 supported)
- NCCL (for multi-GPU)

## Build

First, ensure that all submodules are cloned:
```bash
git submodule update --init --recursive
```

Then build the CUDA kernels and install:
```bash
bun install
```

To rebuild manually: `bun run build:cuda`

## Usage

### Single Generation

```bash
bun run src/index.ts <model> [prompt] [options]

# Examples
bun run src/index.ts Qwen/Qwen3-1.7B "Hello world" --max-tokens 50
bun run src/index.ts openai/gpt-oss-20b "Tell me a joke" --tp 2
```

### Interactive Chat

```bash
bun run src/index.ts Qwen/Qwen3-8B --chat
```

### Batch Processing

```bash
bun run src/index.ts model -i prompts.txt -o outputs.txt --max-tokens 100
```

### OpenAI-Compatible Server

```bash
bun run src/serve.ts Qwen/Qwen3-1.7B --port 8000

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 20}'
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--tp <n\|auto>` | Tensor parallel degree (default: auto) |
| `--max-tokens <n>` | Maximum tokens to generate (default: 100) |
| `--dtype <fp16\|bf16>` | Weight precision (default: model native) |
| `--chat` | Interactive chat mode |
| `--bench` | Run benchmark |
| `--profile` | Show timing breakdown |
| `--trace <file>` | Write JSON trace (chrome://tracing format) |
| `--use-template` | Apply chat template to prompt |
| `--no-cuda-graphs` | Disable CUDA graph capture |
| `--temperature <f>` | Sampling temperature (default: 0 = greedy) |
| `--top-k <n>` | Top-k sampling (default: 0 = disabled) |
| `--top-p <f>` | Nucleus sampling (default: 1.0 = disabled) |
| `-i, --input <file>` | Input file with prompts (one per line) |
| `-o, --output <file>` | Output file for batch results |
| `-q, --quiet` | Suppress loading output |
| `-v, --verbose` | Show startup timing |

## Supported Models

- **Qwen** (including Qwen3 with QK-norm)
- **GPT-OSS** (20B, 120B)
- possibly more, I forget

Any HuggingFace model with LLaMA-style architecture.

## Features

### Inference
- Paged KV cache
- Continuous batching
- CUDA graph capture for decode
- Streaming token output

### Multi-GPU
- Tensor parallelism with NCCL
- Automatic TP degree detection
- Column/row parallel weight sharding
- Pre-sharded weight caching

### Quantization
- MXFP4 (4-bit with E8M0 scales)
- Native FP16/BF16

### Attention
- Flash Attention 3 (H100)
- Paged attention for variable-length batches
- Grouped Query Attention (GQA)
- RoPE scaling (linear, YaRN)
- Sliding window attention

### MoE (Mixture of Experts)
- Expert parallel across GPUs
- MXFP4 quantized experts
- Marlin kernel for batched inference
- Up to 128 experts

### Tokenizer
- Native JS tokenizer (@huggingface/tokenizers)
- Chat templates: ChatML, Llama, GPT-OSS formats
- No Python dependency

## Project Structure

```
src/
  index.ts          # CLI entry point
  serve.ts          # OpenAI-compatible server
  engine/           # Inference engine
  model/            # Config parsing, tokenizer
  parallel/         # Tensor parallelism, NCCL
  backend/cuda/     # CUDA bindings
cuda/
  kernels/          # CUDA kernel implementations
  include/          # C headers
```

## License

MIT
