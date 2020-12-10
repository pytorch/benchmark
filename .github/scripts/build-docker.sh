#!/bin/sh

set -euo pipefail

TODAY=$(date +"%Y%m%d")
IMAGE_NAME=torchbench/pytorch
IMAGE_TAG=${TODAY}_gh${GITHUB_RUN_ID}
PYTORCH_GITGHUB="https://github.com/pytorch/pytorch.git"
PYTORCH_SRC=${HOME}/pytorch
DATA_DIR=${HOME}/benchmark-results/gh${GITHUB_RUN_ID}

# Checkout pytorch code
if [ -d $PYTORCH_SRC ]
then
    # Update the code
    pushd $PYTORCH_SRC && git pull origin master
    popd
else
    # Fetch the newest code
    git clone $PYTORCH_GITHUB $PYTORCH_SRC
fi

# Build the nightly docker
pushd $PYTORCH_SRC

make -f docker.Makefile PYTHON_VERSION=3.7 \
     CUDA_VERSION=10.2 CUDNN_VERSION=7 \
     INSTALL_CHANNEL=pytorch-nightly BUILD_TYPE=official \
     DOCKER_ORG=torchbench \
     DOCKER_TAG=${IMAGE_TAG} \
     EXTRA_DOCKER_BUILD_FLAGS=--no-cache devel-image
popd

# Build Successful. Save the pytorch image tag in benchmarking summary
mkdir -p $DATA_DIR
echo "${IMAGE_NAME}:${IMAGE_TAG}" > ${DATA_DIR}/summary.txt
# Retag the image
docker image tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest
