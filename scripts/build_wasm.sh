#!/bin/bash

#set -e

export EMSDK_IMAGE_TAG="4.0.20"

CURRENT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $CURRENT_PATH

export D_UID=$UID
export D_GID=$GID

if [[ $(uname -m) == "arm64" ]]; then
  echo "Running on ARM64 processor"
  export DOCKER_DEFAULT_PLATFORM="linux/arm64"
  export EMSDK_IMAGE_TAG="${EMSDK_IMAGE_TAG}-arm64"
fi

if [[ "${WLLAMA_TEST_BACKEND}" == "1" ]]; then
  touch "$CURRENT_PATH/../IS_DEBUG_BUILD"
else
  rm -f "$CURRENT_PATH/../IS_DEBUG_BUILD"
fi

docker compose up llamacpp-wasm-builder --exit-code-from llamacpp-wasm-builder
