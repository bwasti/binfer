// Native tokenizer using @huggingface/tokenizers (pure JS, no Python)

import { Tokenizer as HFTokenizer } from "@huggingface/tokenizers";
import { readFileSync, existsSync } from "fs";
import { join } from "path";

export interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface SpecialTokens {
  bos_token_id: number | null;
  eos_token_id: number | null;
  pad_token_id: number | null;
  unk_token_id: number | null;
}

/**
 * Native tokenizer using @huggingface/tokenizers.
 * No Python subprocess needed - pure JS implementation.
 */
export class Tokenizer {
  private tokenizer: HFTokenizer | null = null;
  private modelPath: string;
  private tokenizerConfig: any = null;
  private tokenizerJson: any = null;
  private chatTemplate: string | null = null;

  constructor(modelPath: string) {
    this.modelPath = modelPath;
  }

  /**
   * Load the tokenizer from model files.
   */
  async start(): Promise<void> {
    const tokenizerJsonPath = join(this.modelPath, "tokenizer.json");
    const tokenizerConfigPath = join(this.modelPath, "tokenizer_config.json");
    const chatTemplatePath = join(this.modelPath, "chat_template.jinja");

    if (!existsSync(tokenizerJsonPath)) {
      throw new Error(`tokenizer.json not found at ${tokenizerJsonPath}`);
    }

    this.tokenizerJson = JSON.parse(readFileSync(tokenizerJsonPath, "utf-8"));

    // Load config if available
    if (existsSync(tokenizerConfigPath)) {
      this.tokenizerConfig = JSON.parse(readFileSync(tokenizerConfigPath, "utf-8"));
      this.chatTemplate = this.tokenizerConfig.chat_template || null;
    }

    // Load external chat template file if available (GPT-OSS style)
    if (!this.chatTemplate && existsSync(chatTemplatePath)) {
      this.chatTemplate = readFileSync(chatTemplatePath, "utf-8");
    }

    this.tokenizer = new HFTokenizer(this.tokenizerJson, this.tokenizerConfig);
  }

  /**
   * Encode text to token IDs.
   */
  encode(
    text: string,
    options: { addSpecialTokens?: boolean; maxLength?: number } = {}
  ): number[] {
    if (!this.tokenizer) {
      throw new Error("Tokenizer not started");
    }

    const result = this.tokenizer.encode(text, {
      add_special_tokens: options.addSpecialTokens ?? true,
    });

    let ids = result.ids;

    // Truncate if maxLength specified
    if (options.maxLength && ids.length > options.maxLength) {
      ids = ids.slice(0, options.maxLength);
    }

    return ids;
  }

  /**
   * Decode token IDs to text.
   */
  decode(
    tokenIds: number[],
    options: { skipSpecialTokens?: boolean } = {}
  ): string {
    if (!this.tokenizer) {
      throw new Error("Tokenizer not started");
    }

    return this.tokenizer.decode(tokenIds, {
      skip_special_tokens: options.skipSpecialTokens ?? true,
    });
  }

  /**
   * Apply chat template to messages.
   * Uses Jinja2-style template from tokenizer_config.json
   */
  applyChatTemplate(
    messages: Message[],
    options: { addGenerationPrompt?: boolean } = {}
  ): string {
    if (!this.tokenizer) {
      throw new Error("Tokenizer not started");
    }

    // Check if tokenizer has native chat template support
    if (typeof (this.tokenizer as any).applyChatTemplate === 'function') {
      return (this.tokenizer as any).applyChatTemplate(messages, {
        add_generation_prompt: options.addGenerationPrompt ?? true,
      });
    }

    // Fallback: manual chat template application for common formats
    const addGenPrompt = options.addGenerationPrompt ?? true;

    // Try to use the chat template from config
    if (this.chatTemplate) {
      return this.applyJinjaTemplate(messages, addGenPrompt);
    }

    // Default ChatML format (used by many models including Qwen)
    let result = "";
    for (const msg of messages) {
      result += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
    }
    if (addGenPrompt) {
      result += "<|im_start|>assistant\n";
    }
    return result;
  }

  /**
   * Simple Jinja template parser for chat templates.
   * Handles the most common patterns used in HuggingFace chat templates.
   */
  private applyJinjaTemplate(messages: Message[], addGenerationPrompt: boolean): string {
    const template = this.chatTemplate!;

    // Check for common template patterns
    if (template.includes("<|im_start|>")) {
      // ChatML format (Qwen, etc.)
      let result = "";
      for (const msg of messages) {
        result += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
      }
      if (addGenerationPrompt) {
        result += "<|im_start|>assistant\n";
      }
      return result;
    }

    if (template.includes("[INST]")) {
      // Llama 2/3 format
      let result = "";
      for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        if (msg.role === "system") {
          result += `<<SYS>>\n${msg.content}\n<</SYS>>\n\n`;
        } else if (msg.role === "user") {
          result += `[INST] ${msg.content} [/INST]`;
        } else if (msg.role === "assistant") {
          result += ` ${msg.content} `;
        }
      }
      return result;
    }

    if (template.includes("<|start|>") && template.includes("<|message|>")) {
      // GPT-OSS / OpenAI format
      const today = new Date().toISOString().split('T')[0];
      let result = "";

      // System message with model identity
      result += `<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n`;
      result += `Knowledge cutoff: 2024-06\n`;
      result += `Current date: ${today}\n\n`;
      result += `Reasoning: medium\n\n`;
      result += `# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>`;

      // Check for system/developer message in input
      let loopMessages = messages;
      if (messages.length > 0 && (messages[0].role === "system" || messages[0].role === "developer")) {
        result += `<|start|>developer<|message|># Instructions\n\n${messages[0].content}<|end|>`;
        loopMessages = messages.slice(1);
      }

      // Render conversation
      for (let i = 0; i < loopMessages.length; i++) {
        const msg = loopMessages[i];
        if (msg.role === "user") {
          result += `<|start|>user<|message|>${msg.content}<|end|>`;
        } else if (msg.role === "assistant") {
          result += `<|start|>assistant<|channel|>final<|message|>${msg.content}<|end|>`;
        }
      }

      // Generation prompt
      if (addGenerationPrompt) {
        result += `<|start|>assistant`;
      }
      return result;
    }

    // Default: just concatenate with role prefixes
    let result = "";
    for (const msg of messages) {
      result += `${msg.role}: ${msg.content}\n`;
    }
    if (addGenerationPrompt) {
      result += "assistant: ";
    }
    return result;
  }

  /**
   * Get special token IDs.
   */
  getSpecialTokens(): SpecialTokens {
    const config = this.tokenizerConfig || {};
    return {
      bos_token_id: config.bos_token_id ?? null,
      eos_token_id: config.eos_token_id ?? null,
      pad_token_id: config.pad_token_id ?? null,
      unk_token_id: config.unk_token_id ?? null,
    };
  }

  /**
   * Get vocabulary size.
   */
  getVocabSize(): number {
    if (!this.tokenizer) {
      throw new Error("Tokenizer not started");
    }

    // Try various sources for vocab size
    if ((this.tokenizer as any).vocab_size) {
      return (this.tokenizer as any).vocab_size;
    }
    if (this.tokenizerConfig?.vocab_size) {
      return this.tokenizerConfig.vocab_size;
    }
    // Get from the model vocabulary in tokenizer.json
    if (this.tokenizerJson?.model?.vocab) {
      return Object.keys(this.tokenizerJson.model.vocab).length;
    }
    return 0;
  }

  /**
   * Stop the tokenizer (no-op for native implementation).
   */
  async stop(): Promise<void> {
    this.tokenizer = null;
  }
}

/**
 * Create and start a tokenizer.
 */
export async function createTokenizer(modelPath: string): Promise<Tokenizer> {
  const tokenizer = new Tokenizer(modelPath);
  await tokenizer.start();
  return tokenizer;
}
