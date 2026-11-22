#!/bin/bash

# builds the main example: the llama.cpp UI (llama.cpp/ui) with our overlay and patches on top
# usage: ./build.sh [build|dev]

set -eo pipefail
cd "$(dirname "$0")"

MODE=${1:-build}
ROOT=$(cd ../.. && pwd)
OUT_DIR=$(pwd)/dist
UI_SRC=$ROOT/llama.cpp/ui
STAGE=.build/ui
APP=$STAGE/apps/chat

if [ ! -d "$UI_SRC/apps/chat" ]; then
  echo "llama.cpp/ui not found, run: git submodule update --init"
  exit 1
fi
if [ ! -f "$ROOT/esm/index.js" ]; then
  echo "esm/ not found, run 'npm run build' at the repo root"
  exit 1
fi

echo ">>> stage llama.cpp/ui"
mkdir -p $STAGE
rsync -a --delete --exclude node_modules --exclude .svelte-kit --exclude dist "$UI_SRC/" $STAGE/
for p in patches/*.patch; do
  [ -e "$p" ] || continue
  echo "applying $p"
  patch -s -p1 -d $STAGE < "$p"
done
rsync -a overlay/ $APP/

# npm ci only when the lock file changes
LOCK_HASH=$(shasum $APP/package-lock.json | cut -d' ' -f1)
if [ "$(cat $APP/node_modules/.lock-hash 2>/dev/null)" != "$LOCK_HASH" ]; then
  echo ">>> npm ci"
  # older npm 11 wants a peer that the upstream lock file does not have
  (cd $APP && npm ci --legacy-peer-deps)
  echo "$LOCK_HASH" > $APP/node_modules/.lock-hash
fi
# the app imports @wllama/wllama from this repo (esm/)
mkdir -p $APP/node_modules/@wllama
ln -sfn "$ROOT" $APP/node_modules/@wllama/wllama

cd $APP
export WLLAMA_ROOT=$ROOT
export VITE_PUBLIC_APP_NAME=wllama
if [ "$MODE" = "dev" ]; then
  npx vite dev --config vite.config.wllama.ts
else
  echo ">>> build"
  npm run build-pwa-assets
  rm -rf "$OUT_DIR"
  LLAMA_UI_OUT_DIR=$OUT_DIR npx vite build --config vite.config.wllama.ts
  echo ">>> done, output in $OUT_DIR"
fi
