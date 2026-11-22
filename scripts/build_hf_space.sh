#!/bin/bash

set -eo pipefail

# deletes the LFS files of the space that are not in the given lists of SHA-256 (DRY_RUN=1: only print them)
delete_lfs_except() {
  python3 - "$@" <<'PY'
import os, sys
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_TOKEN"))
keep = set()
for path in sys.argv[1:]:
    keep |= set(open(path).read().split())
if not keep:
    sys.exit("no LFS file to keep, refusing to delete all of them")
old = [f for f in api.list_lfs_files("ngxson/wllama", repo_type="space") if f.file_oid not in keep]
print(f"deleting {len(old)} LFS files ({sum(f.size for f in old) / 1e6:.0f} MB), keeping {len(keep)}")
if old and not os.environ.get("DRY_RUN"):
    api.permanently_delete_lfs_files("ngxson/wllama", old, repo_type="space")
PY
}

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
git lfs ls-files --long HEAD | awk '{print $1}' > ../lfs_main.txt
git checkout -q --orphan build
git add -A
git commit -q -m "update"
git lfs ls-files --long | awk '{print $1}' > ../lfs_new.txt

# the LFS files of the old builds count until they are deleted: delete them before uploading the new ones,
# but keep the ones of the published build until the new one is pushed
delete_lfs_except ../lfs_main.txt ../lfs_new.txt
git push --force origin build:main
delete_lfs_except ../lfs_new.txt

echo ">>> clean up"
cd ..
rm -rf _tmp_hf_space

echo ">>> done"
