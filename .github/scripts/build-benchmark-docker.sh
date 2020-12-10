#!/bin/sh

set -euo pipefail

TODAY=$(date +"%Y%m%d")
IMAGE_NAME=torchbench/pytorch-benchmark
IMAGE_TAG=${TODAY}_gh${GITHUB_RUN_ID}
DATA_DIR=${HOME}/benchmark-results/gh${GITHUB_RUN_ID}

PYTORCH_IMAGE=$(cat $DATA_DIR/summary.txt |head -n 1)
echo "Building benchmark docker on pytorch image ${PYTORCH_IMAGE}..."

docker build -t ${IMAGE_NAME}:${IMAGE_TAG} --no-cache docker

# Preserve the image tag to summary.txt
echo "${IMAGE_NAME}:${IMAGE_TAG}" >> $DATA_DIR/summary.txt
# Retag to pytorch-benchmark:latest
docker image tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest
