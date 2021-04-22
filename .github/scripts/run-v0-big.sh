#!/bin/sh
# This script runs TorchBench v0 without installing the package
# It assumes pytorch and its dependencies have been installed correctly
# Usage:
# run-v0-devbig.sh RESULT_DIR BENCHMARK_FILTER CONDA_ENV_NAME
# All three arguments are required

set -eo pipefail

# Big machine config
DATA_DIR=$1
BENCHMARK_FILTER="$2"
CONDA_ENV_NAME=$3
CURRENT_DIR=$(dirname "$(readlink -f "$0")")

# Install benchmark dependencies
python install.py

sudo -E systemd-run --slice=workload.slice --same-dir --wait --collect --service-type=exec --pty --uid=$USER \
     bash $CURRENT_DIR/run-v0-big-stub.sh $DATA_DIR "${BENCHMARK_FILTER}" $CONDA_ENV_NAME

echo "Benchmark finished successfully. Output data dir is ${DATA_DIR}."
