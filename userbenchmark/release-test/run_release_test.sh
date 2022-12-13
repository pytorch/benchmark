#!/bin/bash

set -xeuo pipefail

CUDA_VERSION="$1"
RESULT_DIR="$2"
EXAMPLES_DIR="${RESULT_DIR}/../examples"
# get the directory of the current script
CURRENT_DIR=$(dirname -- "$0")

CORE_LIST="0-11"
export GOMP_CPU_AFFINITY="0-11"

. switch-cuda.sh "${CUDA_VERSION}"
nvcc --version
# run mnist
mkdir -p "${RESULT_DIR}/mnist"
pushd "${EXAMPLES_DIR}/mnist"
export LOG_FILE=${RESULT_DIR}/mnist/result.log
export MEM_FILE=${RESULT_DIR}/mnist/result_mem.log
taskset -c "${CORE_LIST}" bash "${CURRENT_DIR}/monitor_proc.sh" python main.py --epochs 1
# run mnist-hogwild
mkdir -p ${RESULT_DIR}/mnist_hogwild
pushd "${EXAMPLES_DIR}/mnist_hogwild"
export LOG_FILE=${RESULT_DIR}/mnist_hogwild/result.log
export MEM_FILE=${RESULT_DIR}/mnist_hogwild/result_mem.log
taskset -c "${CORE_LIST}" bash "${CURRENT_DIR}/monitor_proc.sh" python main.py --epochs 10
# run CPU WLM LSTM
mkdir -p ${RESULT_DIR}/wlm_cpu_lstm
pushd "${EXAMPLES_DIR}/word_language_model"
export LOG_FILE=${RESULT_DIR}/wlm_cpu_lstm/result.log
export MEM_FILE=${RESULT_DIR}/wlm_cpu_lstm/result_mem.log
taskset -c "${CORE_LIST}" bash "${CURRENT_DIR}/monitor_proc.sh" python main.py --epochs 10 --model LSTM
# run GPU WLM LSTM
mkdir -p ${RESULT_DIR}/wlm_gpu_lstm
pushd "${EXAMPLES_DIR}/word_language_model"
export LOG_FILE=${RESULT_DIR}/wlm_gpu_lstm/result.log
export MEM_FILE=${RESULT_DIR}/wlm_gpu_lstm/result_mem.log
taskset -c "${CORE_LIST}" bash "${CURRENT_DIR}/monitor_proc.sh" python main.py --epochs 10 --model LSTM --cuda
# run CPU WLM Transformer
mkdir -p ${RESULT_DIR}/wlm_cpu_trans
pushd "${EXAMPLES_DIR}/word_language_model"
export LOG_FILE=${RESULT_DIR}/wlm_cpu_trans/result.log
export MEM_FILE=${RESULT_DIR}/wlm_cpu_trans/result_mem.log
taskset -c "${CORE_LIST}" bash "${CURRENT_DIR}/monitor_proc.sh" python main.py --epochs 10 --model Transformer
# run GPU WLM Transformer
mkdir -p ${RESULT_DIR}/wlm_gpu_trans
pushd "${EXAMPLES_DIR}/word_language_model"
export LOG_FILE=${RESULT_DIR}/wlm_gpu_trans/result.log
export MEM_FILE=${RESULT_DIR}/wlm_gpu_trans/result_mem.log
taskset -c "${CORE_LIST}" bash "${CURRENT_DIR}/monitor_proc.sh" python main.py --epochs 10 --model Transformer --cuda
