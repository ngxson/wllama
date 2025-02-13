#!/bin/bash

set -e

CURRENT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $CURRENT_PATH
cd ..

# we're on the root of the project

echo "// This file is auto-generated" > ./src/workers-code/generated.ts
echo "// To re-generate it, run: npm run build:worker" >> ./src/workers-code/generated.ts

process_file() {
  local file="$1"
  local content
  content=$(node -e "console.log(JSON.stringify(require('fs').readFileSync('$file', 'utf8').toString()))")
  echo "export const $2 = $content;" >> ./src/workers-code/generated.ts
  echo "" >> ./src/workers-code/generated.ts
}

process_file ./src/workers-code/llama-cpp.js  LLAMA_CPP_WORKER_CODE
process_file ./src/workers-code/opfs-utils.js OPFS_UTILS_WORKER_CODE

# emscripten
process_file ./src/multi-thread/wllama.js         WLLAMA_MULTI_THREAD_CODE
process_file ./src/single-thread/wllama.js        WLLAMA_SINGLE_THREAD_CODE

# build CDN paths
node ./scripts/generate_wasm_from_cdn.js
