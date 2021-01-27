#!/bin/sh

set -eo pipefail

# Version of the config
CONFIG_VER=v0

# Run Parameters
RUN_SCRIPT=$1
CONFIG_DIR=${PWD}/score/configs/${CONFIG_VER}
CONFIG_ENV=${CONFIG_DIR}/config-${CONFIG_VER}.env
# Use the latest pytorch-benchmark image
TORCH_IMAGE_ID=torchbench/pytorch-benchmark:latest
if [ -z "$2" ]; then
    DATA_DIR=${HOME}/benchmark-results-v0.1/gh${GITHUB_RUN_ID}
else
    DATA_DIR=$2
fi

# Load environment variables
set -a;
source ${CONFIG_ENV}
set +a;

mkdir -p ${DATA_DIR}

export CUDA_VISIBLE_DEVICES=${GPU_LIST}

docker run \
       --env GITHUB_RUN_ID \
       --env SCRIBE_GRAPHQL_ACCESS_TOKEN \
       --env-file=${CONFIG_ENV} \
       --volume="${DATA_DIR}:/output" \
       --volume="${CONFIG_DIR}:/config" \
       --gpus device=${GPU_LIST} \
       $TORCH_IMAGE_ID \
       bash ${RUN_SCRIPT}

echo "Benchmark finished successfully. Output data dir is benchmark-results-v0.1/gh${GITHUB_RUN_ID}."
