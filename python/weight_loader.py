#!/usr/bin/env python3
"""
Weight loader for Binfer inference framework.
Loads safetensors weights and writes them to POSIX shared memory for zero-copy transfer to Bun.
"""

import json
import mmap
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from safetensors import safe_open


# Shared memory constants
SHM_PREFIX = "/binfer_"
HEADER_SIZE = 262144  # 256KB reserved for metadata (large models have many weights)


# Try to import torch for bfloat16 support
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_tensor_from_safetensors(f, name: str):
    """Get tensor, handling bfloat16 properly."""
    if HAS_TORCH:
        # Use PyTorch which handles bfloat16 natively
        tensor = f.get_tensor(name)
        if hasattr(tensor, 'numpy'):
            # It's already a numpy array
            if tensor.dtype == np.dtype('bfloat16') or str(tensor.dtype) == 'bfloat16':
                # This shouldn't happen with numpy, but just in case
                return tensor.view(np.uint16)
            return tensor
        return tensor
    else:
        # Fallback: use raw bytes for bfloat16
        return f.get_tensor(name)


def dtype_to_numpy(dtype_str: str) -> np.dtype:
    """Convert safetensors dtype string to numpy dtype."""
    mapping = {
        "F32": np.float32,
        "F16": np.float16,
        "BF16": np.float16,  # We'll keep as raw bytes (same size)
        "I32": np.int32,
        "I64": np.int64,
        "U8": np.uint8,
        "I8": np.int8,
    }
    return np.dtype(mapping.get(dtype_str, np.float32))


def dtype_size(dtype_str: str) -> int:
    """Get byte size of dtype."""
    sizes = {
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "I64": 8,
        "U8": 1,
        "I8": 1,
    }
    return sizes.get(dtype_str, 4)


class WeightInfo:
    """Information about a single weight tensor."""
    def __init__(self, name: str, shape: List[int], dtype: str, offset: int, size: int):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.offset = offset  # Offset in shared memory (after header)
        self.size = size  # Size in bytes


class SharedMemoryWeightLoader:
    """Loads weights into POSIX shared memory for zero-copy transfer."""

    def __init__(self, model_path: str, shm_name: Optional[str] = None, quiet: bool = False):
        self.model_path = Path(model_path)
        self.shm_name = shm_name or f"{SHM_PREFIX}{os.getpid()}"
        self.quiet = quiet
        self.weights: Dict[str, WeightInfo] = {}
        self.shm_fd: Optional[int] = None
        self.shm_size: int = 0
        self.mm: Optional[mmap.mmap] = None

    def load(self) -> Dict[str, WeightInfo]:
        """Load all safetensor files and write to shared memory."""
        # Find all safetensor files
        if self.model_path.is_file():
            safetensor_files = [self.model_path]
        else:
            safetensor_files = sorted(self.model_path.glob("*.safetensors"))
            if not safetensor_files:
                # Try model.safetensors specifically
                model_file = self.model_path / "model.safetensors"
                if model_file.exists():
                    safetensor_files = [model_file]

        if not safetensor_files:
            raise FileNotFoundError(f"No safetensor files found in {self.model_path}")

        # Determine framework to use
        framework = "pt" if HAS_TORCH else "numpy"

        # First pass: calculate total size and collect metadata
        total_size = HEADER_SIZE
        all_tensors = []

        for sf_path in safetensor_files:
            with safe_open(sf_path, framework=framework) as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    if HAS_TORCH and hasattr(tensor, 'numpy'):
                        # PyTorch tensor - keep bf16 as-is (CUDA kernels support bf16)
                        if tensor.dtype == torch.bfloat16:
                            size = tensor.numel() * 2  # bf16 is 2 bytes
                            dtype_str = "bfloat16"  # Keep bf16 natively
                        else:
                            size = tensor.numel() * tensor.element_size()
                            dtype_str = str(tensor.dtype).replace("torch.", "")
                        shape = list(tensor.shape)
                    else:
                        size = tensor.nbytes
                        dtype_str = str(tensor.dtype)
                        shape = list(tensor.shape)

                    # Align to 256 bytes for GPU efficiency
                    aligned_size = (size + 255) // 256 * 256
                    all_tensors.append({
                        "name": name,
                        "shape": shape,
                        "dtype": dtype_str,
                        "path": str(sf_path),
                        "offset": total_size,
                        "size": size,
                        "aligned_size": aligned_size,
                    })
                    total_size += aligned_size

        self.shm_size = total_size
        if not self.quiet:
            print(f"Total weight size: {total_size / 1024 / 1024:.2f} MB", file=sys.stderr)

        # Create shared memory
        self._create_shm()

        # Second pass: load and write tensors
        # Group tensors by file to avoid reopening
        tensors_by_file: Dict[str, List[dict]] = {}
        for tensor_info in all_tensors:
            path = tensor_info["path"]
            if path not in tensors_by_file:
                tensors_by_file[path] = []
            tensors_by_file[path].append(tensor_info)

        total_tensors = len(all_tensors)
        loaded_count = 0
        bytes_loaded = 0

        for sf_path, tensors in tensors_by_file.items():
            with safe_open(sf_path, framework=framework) as f:
                for tensor_info in tensors:
                    tensor = f.get_tensor(tensor_info["name"])
                    self._write_tensor_data(tensor, tensor_info["offset"], tensor_info["dtype"])

                    self.weights[tensor_info["name"]] = WeightInfo(
                        name=tensor_info["name"],
                        shape=tensor_info["shape"],
                        dtype=tensor_info["dtype"],
                        offset=tensor_info["offset"],
                        size=tensor_info["size"],
                    )

                    loaded_count += 1
                    bytes_loaded += tensor_info["size"]

                    # Progress for large models
                    if not self.quiet and total_tensors > 50 and loaded_count % 10 == 0:
                        pct = (loaded_count / total_tensors) * 100
                        mb_loaded = bytes_loaded / 1024 / 1024
                        print(f"\r  Loading: {pct:.1f}% ({loaded_count}/{total_tensors} tensors, {mb_loaded:.0f}MB)", end="", file=sys.stderr)

        if not self.quiet and total_tensors > 50:
            print("", file=sys.stderr)  # newline after progress

        # Write header with metadata
        self._write_header()

        return self.weights

    def _create_shm(self):
        """Create POSIX shared memory segment."""
        import posix_ipc

        # Remove existing if present
        try:
            shm = posix_ipc.SharedMemory(self.shm_name)
            shm.close_fd()
            shm.unlink()
        except posix_ipc.ExistentialError:
            pass

        # Create new shared memory
        shm = posix_ipc.SharedMemory(
            self.shm_name,
            posix_ipc.O_CREAT | posix_ipc.O_EXCL,
            size=self.shm_size
        )
        self.shm_fd = shm.fd

        # Memory map it
        self.mm = mmap.mmap(self.shm_fd, self.shm_size)

    def _write_tensor(self, tensor: np.ndarray, offset: int):
        """Write numpy tensor data to shared memory at given offset."""
        data = tensor.tobytes()
        self.mm[offset:offset + len(data)] = data

    def _write_tensor_data(self, tensor, offset: int, dtype_str: str):
        """Write tensor data to shared memory, handling both PyTorch and numpy tensors."""
        if HAS_TORCH and hasattr(tensor, 'cpu'):
            # PyTorch tensor - ensure it's on CPU
            tensor = tensor.cpu()
            if tensor.dtype == torch.bfloat16:
                # Keep bfloat16 as-is - CUDA kernels support bf16 natively
                # Convert to uint16 view (same bytes) for numpy compatibility
                data = tensor.view(torch.uint16).numpy().tobytes()
            else:
                # Regular PyTorch tensor - convert to numpy first
                data = tensor.numpy().tobytes()
        else:
            # Numpy array
            data = tensor.tobytes()
        self.mm[offset:offset + len(data)] = data

    def _write_header(self):
        """Write JSON header with tensor metadata."""
        header = {
            "version": 1,
            "shm_name": self.shm_name,
            "shm_size": self.shm_size,
            "tensors": {
                name: {
                    "shape": info.shape,
                    "dtype": info.dtype,
                    "offset": info.offset,
                    "size": info.size,
                }
                for name, info in self.weights.items()
            }
        }
        header_json = json.dumps(header).encode("utf-8")
        if len(header_json) > HEADER_SIZE - 8:
            raise ValueError(f"Header too large: {len(header_json)} bytes")

        # Write length prefix + JSON
        self.mm[0:4] = struct.pack("<I", len(header_json))
        self.mm[4:4 + len(header_json)] = header_json

    def get_shm_info(self) -> dict:
        """Get shared memory info for Bun to read."""
        return {
            "shm_name": self.shm_name,
            "shm_size": self.shm_size,
            "tensors": {
                name: {
                    "shape": info.shape,
                    "dtype": info.dtype,
                    "offset": info.offset,
                    "size": info.size,
                }
                for name, info in self.weights.items()
            }
        }

    def cleanup(self):
        """Unlink shared memory (call after Bun has copied to GPU)."""
        if self.mm:
            self.mm.close()
        if self.shm_fd is not None:
            import posix_ipc
            try:
                shm = posix_ipc.SharedMemory(self.shm_name)
                shm.unlink()
            except posix_ipc.ExistentialError:
                pass


def main():
    """CLI for weight loading."""
    import argparse

    parser = argparse.ArgumentParser(description="Load HuggingFace weights into shared memory")
    parser.add_argument("model_path", help="Path to model directory or safetensors file")
    parser.add_argument("--shm-name", help="Shared memory name (default: auto-generated)")
    parser.add_argument("--wait", action="store_true", help="Wait for signal before cleanup")
    parser.add_argument("--quiet", action="store_true", help="Suppress status messages")
    args = parser.parse_args()

    loader = SharedMemoryWeightLoader(args.model_path, args.shm_name, quiet=args.quiet)

    try:
        weights = loader.load()
        info = loader.get_shm_info()

        # Output JSON to stdout for Bun to read
        print(json.dumps(info))
        sys.stdout.flush()

        if args.wait:
            # Wait for signal from Bun that it's done copying
            if not args.quiet:
                print("Waiting for cleanup signal...", file=sys.stderr)
            input()  # Wait for newline from Bun

    finally:
        loader.cleanup()
        if not args.quiet:
            print("Cleaned up shared memory", file=sys.stderr)


if __name__ == "__main__":
    main()
