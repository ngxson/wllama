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
# the example imports @wllama/wllama from esm/, which is not committed, build it
(cd source && npm ci --include=dev && npm run build)

./build.sh

echo ">>> push"
if [ -z "$(git status --porcelain)" ]; then
  echo "nothing changed, skipping..."
  exit 0
fi
# the space has a storage limit (1GB) and every build adds new wasm files: push one commit without history
git checkout -q --orphan build
git add -A
git commit -q -m "update"

# the LFS files of the old builds still count until they are deleted, delete them before uploading the new ones
git lfs ls-files --long | awk '{print $1}' > ../lfs_keep.txt
python3 - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
keep = set(open("../lfs_keep.txt").read().split())
old = [f for f in api.list_lfs_files("ngxson/wllama", repo_type="space") if f.file_oid not in keep]
print(f"deleting {len(old)} old LFS files ({sum(f.size for f in old) / 1e6:.0f} MB)")
if old:
    api.permanently_delete_lfs_files("ngxson/wllama", old, repo_type="space")
PY

git push --force origin build:main

echo ">>> clean up"
cd ..
rm -rf _tmp_hf_space

echo ">>> done"
