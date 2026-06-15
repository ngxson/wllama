#!/bin/bash

# Git bisect script for finding the llama.cpp commit that introduced the locale crash.
# Run from wllama root: git bisect run ./scripts/bisect_test.sh
#
# Start (good): aa46bda89b9a8378ae76bb15fc2ce2f571f0983c  (wllama master's llama.cpp)
# End   (bad):  dd4623a74                                  (current HEAD of submodule)

set -e

WLLAMA_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
cd "$WLLAMA_ROOT"

CONTEXT_H="$WLLAMA_ROOT/cpp/wllama-context.h"
FIT_H="$WLLAMA_ROOT/llama.cpp/common/fit.h"

# Adjust #if 0 / #if 1 around common_get_device_memory_data stub based on
# which signature fit.h declares at the current bisect commit.
if grep -q "common_device_memory_data_vec" "$FIT_H" 2>/dev/null; then
    # New signature (after d8a24ccee): enable the common_device_memory_data_vec block
    sed -i.bak 's/^#if [01]$/\#if 1/' "$CONTEXT_H"
else
    # Old signature (before d8a24ccee): enable the std::vector<llama_device_memory_data> block
    sed -i.bak 's/^#if [01]$/\#if 0/' "$CONTEXT_H"
fi
rm -f "${CONTEXT_H}.bak"

# Build and run tests. Exit 125 = skip (build infra broken at this commit).
rm -rf build

SKIP_COMPAT=1 npm run build:wasm 2>&1 || exit 125
npm run build 2>&1 || exit 125
AUTO=1 npm run test 2>&1
