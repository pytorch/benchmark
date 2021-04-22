#!/bin/sh
# This is the stub of run-v0-big.sh
# Do not directly run it!

DATA_DIR=$1
BENCHMARK_FILTER="$2"
CONDA_ENV_NAME=$3
NUM_ITER=1
NUM_ROUNDS=20
DATA_JSON_PREFIX=$(date +"%Y%m%d_%H%M%S")

bash $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}

echo "Running benchmark with filter: \"${BENCHMARK_FILTER}\""
pytest test_bench.py -k "${BENCHMARK_FILTER}" \
       --ignore_machine_config \
       --benchmark-min-rounds "${NUM_ROUNDS}" \
       --benchmark-json ${DATA_DIR}/${DATA_JSON_PREFIX}_${c}.json
