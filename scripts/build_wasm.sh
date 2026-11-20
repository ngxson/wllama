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

# only the compat build needs the patched binaryen, skip it (slow to build) with SKIP_COMPAT
# a new patch gives a new image tag, so the image is rebuilt
if [ -z "${SKIP_COMPAT}" ]; then
  export BINARYEN_PATCH=1
  export IMAGE_SUFFIX="-$(shasum binaryen-asyncify-try.patch | cut -c1-8)"
else
  export BINARYEN_PATCH=0
  export IMAGE_SUFFIX=""
fi

# the patched wasm-opt is also kept in .binaryen-cache (CI caches it), the name has the image tag so an old one is not used
mkdir -p .binaryen-cache
export WASM_OPT_CACHE="wasm-opt-${EMSDK_IMAGE_TAG}${IMAGE_SUFFIX}"

# one compose project per checkout, so parallel builds in different worktrees do not share a container
REPO_ROOT="$(cd "$CURRENT_PATH/.." && pwd)"
REPO_NAME="$(basename "$REPO_ROOT" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_-' '-' | sed 's/-*$//')"
REPO_HASH="$(printf '%s' "$REPO_ROOT" | shasum | cut -c1-8)"
export COMPOSE_PROJECT_NAME="wllama-${REPO_NAME}-${REPO_HASH}"

docker compose run --rm llamacpp-wasm-builder
