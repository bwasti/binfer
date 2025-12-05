#!/bin/bash
# Binfer setup and test script

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Binfer Setup Script                                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo

# 1. Install Bun dependencies
echo "Step 1: Installing Bun dependencies..."
bun install
echo "✓ Dependencies installed"
echo

# 2. Setup Python environment
echo "Step 2: Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r python/requirements.txt -q
pip install posix_ipc -q  # For shared memory
echo "✓ Python environment ready"
echo

# 3. Build CUDA library
echo "Step 3: Building CUDA library..."
if command -v nvcc &> /dev/null; then
    cd cuda
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cd ../..
    echo "✓ CUDA library built"
else
    echo "⚠ CUDA not found, skipping CUDA build"
    echo "  Install CUDA Toolkit to enable GPU acceleration"
fi
echo

# 4. Run TypeScript type check
echo "Step 4: Type checking..."
bun run typecheck || echo "⚠ Type check had issues (expected without CUDA)"
echo

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Setup Complete!                                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo
echo "Next steps:"
echo "  1. Run tests:        bun test"
echo "  2. Run main:         bun run src/index.ts <model_path>"
echo "  3. Example:          bun run src/index.ts meta-llama/Llama-3.2-1B-Instruct"
echo
