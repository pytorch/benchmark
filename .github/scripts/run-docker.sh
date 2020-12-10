#!/bin/sh

set -euo pipefail

# Version of the config
CONFIG_VER=v0

# Run Parameters
RUN_SCRIPT=$1
CONFIG_DIR=${PWD}/score/configs/${CONFIG_VER}
CONFIG_ENV=${CONFIG_DIR}/config-${CONFIG_VER}.env
DATA_DIR=${HOME}/benchmark-results/gh${GITHUB_RUN_ID}
# Use the latest pytorch-benchmark image
TORCH_IMAGE_ID=torchbench/pytorch-benchmark:latest

# Load environment variables
set -a;
source ${CONFIG_ENV}
set +a;

TORCHBENCH_IMAGE=$(cat $DATA_DIR/summary.txt |head -n 2)
echo "Running pytorch-benchmark image ${TORCHBENCH_IMAGE}, config version ${CONFIG_VER}"

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
