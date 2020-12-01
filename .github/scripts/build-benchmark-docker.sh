#!/bin/sh

set -euo pipefail

TODAY=$(date +"%Y%m%d")

# Get the latest pytorch docker image
TORCH_LATEST=$(docker images | grep "pytorch\s" | head -n 1)
TORCH_ID=$(echo $TORCH_LATEST | tr -s ' ' | cut -d ' ' -f3)

echo "Building benchmark docker using pytorch docker $TORCH_ID..."

sed -i 's,FROM pytorch,FROM '"$TORCH_ID"',' docker/Dockerfile
docker build -t $(id -un)/pytorch-benchmark:${TODAY}_${GITHUB_RUN_ID} docker
sed -i 's,FROM '"$TORCH_ID"',FROM pytorch,' docker/Dockerfile
