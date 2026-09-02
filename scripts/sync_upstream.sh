#!/bin/bash

# Rebuild and regenerate everything that depends on the llama.cpp submodule.
# Run this after bumping the submodule, see the "Syncing llama.cpp upstream" section in AGENTS.md

set -e

CURRENT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $CURRENT_PATH/..

npm run build:wasm
npm run build
npm run format
