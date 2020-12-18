#!/bin/sh

set -euo pipefail

TODAY=$(date +"%Y%m%d")
IMAGE_NAME=torchbench/pytorch-benchmark
IMAGE_TAG=${TODAY}_gh${GITHUB_RUN_ID}
DATA_DIR=${HOME}/benchmark-results/gh${GITHUB_RUN_ID}

docker build -t ${IMAGE_NAME}:${IMAGE_TAG} --no-cache docker

# Retag to pytorch-benchmark:latest
docker image tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest
