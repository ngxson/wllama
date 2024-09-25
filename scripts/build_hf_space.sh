#!/bin/bash

set -e

echo ">>> clone"
rm -rf _tmp_hf_space
git clone https://ngxson:${HF_TOKEN}@huggingface.co/spaces/ngxson/wllama --depth 1 _tmp_hf_space

echo ">>> build"
cd _tmp_hf_space
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
