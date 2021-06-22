#!/bin/sh
# This script runs TorchBench without installing the pytorch and torch dep packages
# It assumes pytorch, torchtext, and torchvision have already been installed
# Usage:
# run.sh RESULT_DIR [BENCHMARK_FILTER]
# The RESULT_DIR is required, BENCHMARK_FILTER is optional

set -xeo pipefail

# Check Github Run ID
if [ -z "$GITHUB_RUN_ID" ]; then
    echo "You must specify the GitHub Run ID"
    exit 1
fi

# Version of the config
if [ -z "$CONFIG_VER" ]; then
    CONFIG_VER=v1
fi
CONFIG_DIR=${PWD}/torchbenchmark/score/configs/${CONFIG_VER}
CONFIG_ENV=${CONFIG_DIR}/config-${CONFIG_VER}.env
DATA_JSON_PREFIX=$(date +"%Y%m%d_%H%M%S")
if [ -z "$1" ]; then
    echo "You must specify output data dir"
    exit 1
fi
DATA_DIR=$1

# Load environment variables
set -a;
source ${CONFIG_ENV}
set +a;
# Must read BENCHMARK_FILTER after loading the config
# Because config has a preset BENCHMARK_FILTER
if [ -n "$2" ]; then
    BENCHMARK_FILTER="$2"
fi

sudo nvidia-smi -ac ${GPU_FREQUENCY}
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export GOMP_CPU_AFFINITY="${CORE_LIST}"

echo "Running benchmark with filter: \"${BENCHMARK_FILTER}\""

# Install benchmark dependencies
python install.py

# Run the benchmark
for c in $(seq 1 $NUM_ITER); do
    taskset -c "${CORE_LIST}" pytest test_bench.py -k "${BENCHMARK_FILTER}" \
            --benchmark-min-rounds "${NUM_ROUNDS}" \
            --benchmark-json ${DATA_DIR}/${DATA_JSON_PREFIX}_${c}.json
done

echo "Benchmark finished successfully. Output data dir is ${DATA_DIR}."
