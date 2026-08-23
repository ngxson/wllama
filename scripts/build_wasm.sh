#!/bin/bash

#set -e

export EMSDK_IMAGE_TAG="4.0.20"

CURRENT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $CURRENT_PATH

# UID is set by bash, but GID is not (it is a zsh thing), fallback to id
export D_UID=${UID:-$(id -u)}
export D_GID=${GID:-$(id -g)}

if [[ $(uname -m) == "arm64" ]]; then
  echo "Running on ARM64 processor"
  export DOCKER_DEFAULT_PLATFORM="linux/arm64"
  export EMSDK_IMAGE_TAG="${EMSDK_IMAGE_TAG}-arm64"
fi

docker compose up llamacpp-wasm-builder --exit-code-from llamacpp-wasm-builder
