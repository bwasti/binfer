#!/usr/bin/env python3
"""
HuggingFace model downloader for Binfer.
Downloads models from the HuggingFace Hub with caching support.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from huggingface_hub import hf_hub_download, snapshot_download, HfApi


def download_model(
    model_id: str,
    revision: str = "main",
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    files: Optional[List[str]] = None,
) -> Path:
    """
    Download a model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-3B-Instruct")
        revision: Git revision (branch, tag, or commit)
        cache_dir: Local cache directory
        token: HuggingFace API token for gated models
        files: Specific files to download (None = all)

    Returns:
        Path to the downloaded model directory
    """
    # Use environment token if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if files:
        # Download specific files
        model_dir = None
        for filename in files:
            path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )
            if model_dir is None:
                model_dir = Path(path).parent
        return model_dir
    else:
        # Download entire model (safetensors + config)
        # Ignore unnecessary files
        ignore_patterns = [
            "*.bin",  # Prefer safetensors
            "*.msgpack",
            "*.h5",
            "optimizer*",
            "training_args*",
            "*.ot",
        ]

        model_dir = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            ignore_patterns=ignore_patterns,
        )
        return Path(model_dir)


def get_model_info(model_id: str, token: Optional[str] = None) -> dict:
    """Get model information from HuggingFace Hub."""
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    api = HfApi()
    info = api.model_info(model_id, token=token)

    return {
        "id": info.id,
        "author": info.author,
        "sha": info.sha,
        "pipeline_tag": info.pipeline_tag,
        "tags": info.tags,
        "downloads": info.downloads,
        "library_name": info.library_name,
    }


def list_files(model_id: str, token: Optional[str] = None) -> List[str]:
    """List files in a model repository."""
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    api = HfApi()
    files = api.list_repo_files(model_id, token=token)
    return list(files)


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace models")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("model_id", help="HuggingFace model ID")
    download_parser.add_argument("--revision", default="main", help="Git revision")
    download_parser.add_argument("--cache-dir", help="Cache directory")
    download_parser.add_argument("--token", help="HuggingFace API token")
    download_parser.add_argument("--files", nargs="+", help="Specific files to download")

    # Info command
    info_parser = subparsers.add_parser("info", help="Get model info")
    info_parser.add_argument("model_id", help="HuggingFace model ID")
    info_parser.add_argument("--token", help="HuggingFace API token")

    # List command
    list_parser = subparsers.add_parser("list", help="List model files")
    list_parser.add_argument("model_id", help="HuggingFace model ID")
    list_parser.add_argument("--token", help="HuggingFace API token")

    args = parser.parse_args()

    if args.command == "download":
        model_dir = download_model(
            args.model_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            token=args.token,
            files=args.files,
        )
        print(json.dumps({"path": str(model_dir)}))

    elif args.command == "info":
        info = get_model_info(args.model_id, token=args.token)
        print(json.dumps(info, indent=2))

    elif args.command == "list":
        files = list_files(args.model_id, token=args.token)
        for f in files:
            print(f)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
