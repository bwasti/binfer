#!/usr/bin/env python3
"""
Tokenizer server for Binfer.
Provides tokenization/detokenization via a simple JSON protocol over stdin/stdout.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

from transformers import AutoTokenizer


class TokenizerServer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Encode text to token IDs."""
        kwargs = {"add_special_tokens": add_special_tokens}
        if max_length:
            kwargs["max_length"] = max_length
            kwargs["truncation"] = True
        return self.tokenizer.encode(text, **kwargs)

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self,
        messages: List[dict],
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply chat template to messages."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def get_vocab_size(self) -> int:
        return len(self.tokenizer)

    def get_special_tokens(self) -> dict:
        return {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "unk_token_id": self.tokenizer.unk_token_id,
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: tokenizer.py <model_path>", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]
    server = TokenizerServer(model_path)

    # Signal ready
    print(json.dumps({"status": "ready", "vocab_size": server.get_vocab_size()}))
    sys.stdout.flush()

    # Process requests
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            cmd = request.get("cmd")

            if cmd == "encode":
                result = server.encode(
                    request["text"],
                    add_special_tokens=request.get("add_special_tokens", True),
                    max_length=request.get("max_length"),
                )
                response = {"token_ids": result}

            elif cmd == "decode":
                result = server.decode(
                    request["token_ids"],
                    skip_special_tokens=request.get("skip_special_tokens", True),
                )
                response = {"text": result}

            elif cmd == "apply_chat_template":
                result = server.apply_chat_template(
                    request["messages"],
                    add_generation_prompt=request.get("add_generation_prompt", True),
                )
                response = {"text": result}

            elif cmd == "get_special_tokens":
                response = server.get_special_tokens()

            elif cmd == "quit":
                break

            else:
                response = {"error": f"Unknown command: {cmd}"}

            print(json.dumps(response))
            sys.stdout.flush()

        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
