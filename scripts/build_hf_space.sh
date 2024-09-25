#!/bin/bash

set -e

# clone
rm -rf _tmp_hf_space
git clone https://ngxson:${HF_TOKEN}@huggingface.co/spaces/ngxson/wllama --depth 1 _tmp_hf_space

# build
cd _tmp_hf_space
./build.sh

# push
git add -A
git commit -m "update"
git push

# clean up
cd ..
rm -rf _tmp_hf_space
