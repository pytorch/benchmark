#!/bin/sh

set -euo pipefail

# Version of the config
CONFIG_VER=v0

# Run Parameters
RUN_SCRIPT=$1
CONFIG_DIR=${PWD}/.github/configs/${CONFIG_VER}
CONFIG_ENV=${CONFIG_DIR}/config-${CONFIG_VER}.env
DATA_DIR=${HOME}/benchmark-results/gh${GITHUB_RUN_ID}

# Load environment variables
set -a;
source ${CONFIG_ENV}
set +a;

# Use the latest pytorch-benchmark image
TORCH_IMAGE=$(docker images | grep "pytorch-benchmark" | sed -n '1 p')
TORCH_IMAGE_ID=$(echo $TORCH_IMAGE | tr -s ' ' | cut -d' ' -f3)

echo "Running pytorch-benchmark image ${TORCH_IMAGE_ID}, config ${CONFIG_VER}"

mkdir -p ${DATA_DIR}

export CUDA_VISIBLE_DEVICES=${GPU_LIST}
# Nvidia won't let this run inside docker
# Make sure the Nvidia GPU is in persistence mode
sudo nvidia-smi -pm ENABLED -i ${GPU_LIST}
# Set the <memory, graphics> clock frequency
# Need further study on how Nvidia card throttling affect overall performance variance
sudo nvidia-smi -ac ${GPU_FREQUENCY}

docker run \
       --env GITHUB_RUN_ID \
       --env SCRIBE_GRAPHQL_ACCESS_TOKEN \
       --env-file=${CONFIG_ENV} \
       --volume="${PWD}:/runner" \
       --volume="${DATA_DIR}:/output" \
       --volume="${CONFIG_DIR}:/config" \
       --gpus device=${GPU_LIST} \
       $TORCH_IMAGE_ID \
       bash ${RUN_SCRIPT}

echo "Benchmark finished successfully. Output data dir is benchmark-results/gh${GITHUB_RUN_ID}."
