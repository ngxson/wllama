#!/bin/bash

set -e

echo ">>> clone"
rm -rf _tmp_hf_space
git clone https://ngxson:${HF_TOKEN}@huggingface.co/spaces/ngxson/wllama --depth 1 _tmp_hf_space

echo ">>> build"
cd _tmp_hf_space

# pre-seed the source clone so we can build the wasm (it is not committed to git), build.sh will reuse it
git clone --recurse-submodules https://github.com/ngxson/wllama.git source
(cd source && ./scripts/build_wasm.sh)

./build.sh

echo ">>> push"
if [ -z "$(git status --porcelain)" ]; then
  echo "nothing changed, skipping..."
  exit 0
fi
git add -A
git commit -m "update"
git push

echo ">>> clean up"
cd ..
rm -rf _tmp_hf_space

echo ">>> done"
