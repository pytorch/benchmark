#!/bin/sh
# This script runs TorchBench v0 without installing the package
# It assumes pytorch and its dependencies have been installed correctly
# Usage:
# run-v0-devbig.sh RESULT_DIR [BENCHMARK_FILTER]
# The RESULT_DIR is required, BENCHMARK_FILTER is optional

set -eo pipefail

# Big machine config
PYTHON_VERSION=3.7
CUDA_VERSION=cpu
NUM_ITER=1
NUM_ROUNDS=20
CORE_LIST=14-27
DATA_JSON_PREFIX=$(date +"%Y%m%d_%H%M%S")
DATA_DIR=$1

if [ -n "$2" ]; then
    BENCHMARK_FILTER="$2"
fi

export GOMP_CPU_AFFINITY="${CORE_LIST}"

echo "Running benchmark with filter: \"${BENCHMARK_FILTER}\""

# Install benchmark dependencies
python install.py

# Run the benchmark
for c in $(seq 1 $NUM_ITER); do
    taskset -c "${CORE_LIST}" \
         pytest test_bench.py -k "${BENCHMARK_FILTER}" \
                --benchmark-min-rounds "${NUM_ROUNDS}" \
                --benchmark-json ${DATA_DIR}/${DATA_JSON_PREFIX}_${c}.json
done

echo "Benchmark finished successfully. Output data dir is ${DATA_DIR}."
