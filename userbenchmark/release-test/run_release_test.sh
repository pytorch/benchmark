#!/bin/bash

set -xeuo pipefail

CUDA_VERSION="$1"
RESULT_DIR="$2"
EXAMPLES_DIR="${RESULT_DIR}/../examples"
# get the directory of the current script
CURRENT_DIR=$(dirname -- "$0")

. switch-cuda.sh "${CUDA_VERSION}"
nvcc --version
# run mnist
mkdir -p "${RESULT_DIR}/mnist"
pushd "${EXAMPLES_DIR}/mnist"
export LOG_FILE=${RESULT_DIR}/mnist/result.log
export MEM_FILE=${RESULT_DIR}/mnist/result_mem.log
bash "${CURRENT_DIR}/monitor_proc.sh" python main.py --epochs 10
