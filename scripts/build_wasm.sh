#!/bin/bash

set -e

CURRENT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $CURRENT_PATH

export D_UID=$UID
export D_GID=$GID

docker compose up llamacpp-wasm-builder --exit-code-from llamacpp-wasm-builder
