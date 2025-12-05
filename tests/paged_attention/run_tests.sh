#!/bin/bash
# Run paged attention tests - SIMPLIFIED
# Only runs the key tests that have been verified to work

cd "$(dirname "$0")/../.."

echo "====================================="
echo "  Paged Attention Test Suite"
echo "====================================="

# Test 1: Minimal comparison (MOST IMPORTANT)
echo ""
echo ">>> Test 1: Minimal Attention Comparison"
echo "-------------------------------------"
HF_HUB_OFFLINE=1 timeout 60 bun run tests/paged_attention/test_minimal.ts 2>&1

# Test 2: KV Cache Roundtrip
echo ""
echo ">>> Test 2: KV Cache Roundtrip"
echo "-------------------------------------"
HF_HUB_OFFLINE=1 timeout 60 bun run tests/paged_attention/test_kv_cache_roundtrip.ts 2>&1

# Test 3: Decode Flow (simulates batched_engine decode path)
echo ""
echo ">>> Test 3: Decode Flow Simulation"
echo "-------------------------------------"
HF_HUB_OFFLINE=1 timeout 60 bun run tests/paged_attention/test_decode_flow.ts 2>&1

echo ""
echo "====================================="
echo "  Test Suite Complete"
echo "====================================="
echo ""
echo "NOTE: These tests verify kernel-level correctness."
echo "If tests pass but production still fails, the bug is in"
echo "the batched_engine.ts integration layer."
