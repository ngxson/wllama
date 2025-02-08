#!/bin/bash

#set -e

CURRENT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $CURRENT_PATH

export D_UID=$UID
export D_GID=$GID

# temporary apply that viral x2 speedup PR lmao
wget https://patch-diff.githubusercontent.com/raw/ggerganov/llama.cpp/pull/11453.patch -O ../tmp.patch
cd ../llama.cpp
git apply ../tmp.patch
cd $CURRENT_PATH

docker compose up llamacpp-wasm-builder --exit-code-from llamacpp-wasm-builder

cd ../llama.cpp
git reset --hard HEAD
