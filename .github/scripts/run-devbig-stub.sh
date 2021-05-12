#!/bin/sh
# This is the stub of run-devbig.sh
# DO NOT directly run it!
set -eu

DATA_DIR=$1
BENCHMARK_FILTER="$2"
CONDA_ENV_NAME=$3
NUM_ROUNDS=20
DATA_JSON_PREFIX=$(date +"%Y%m%d_%H%M%S")

. ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}

echo "Running benchmark with filter: \"${BENCHMARK_FILTER}\""

# Ignore the machine config, because it is on devbig
pytest test_bench.py -k "${BENCHMARK_FILTER}" \
       --ignore_machine_config \
       --benchmark-min-rounds "${NUM_ROUNDS}" \
       --benchmark-json ${DATA_DIR}/${DATA_JSON_PREFIX}.json
