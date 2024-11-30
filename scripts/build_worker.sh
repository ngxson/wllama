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
  content=$(cat "$file" | sed 's/`/\\`/g' | sed 's/\$/\\$/g')
  echo "export const $2 = \`$content\`;" >> ./src/workers-code/generated.ts
  echo "" >> ./src/workers-code/generated.ts
}

process_file ./src/workers-code/llama-cpp.js LLAMA_CPP_WORKER_CODE
