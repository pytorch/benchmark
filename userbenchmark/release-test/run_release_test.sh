#!/bin/bash

set -xeuo pipefail

CUDA_VERSION="$1"
RESULT_DIR="$2"
EXAMPLES_DIR="${RESULT_DIR}/../examples"
# get the directory of the current script
CURRENT_DIR=$(dirname -- "$0")

CORE_LIST="24-47"
export GOMP_CPU_AFFINITY="24-47"

. switch-cuda.sh "${CUDA_VERSION}"
nvcc --version
#Wei# run mnist
#Weimkdir -p "${RESULT_DIR}/mnist"
#Weipushd "${EXAMPLES_DIR}/mnist"
#Weiexport LOG_FILE=${RESULT_DIR}/mnist/result.log
#Weiexport MEM_FILE=${RESULT_DIR}/mnist/result_mem.log
#Weitaskset -c "${CORE_LIST}" bash "${CURRENT_DIR}/monitor_proc.sh" python main.py --epochs 1
#Wei# run mnist-hogwild
#Weimkdir -p ${RESULT_DIR}/mnist_hogwild
#Weipushd "${EXAMPLES_DIR}/mnist_hogwild"
#Weiexport LOG_FILE=${RESULT_DIR}/mnist_hogwild/result.log
#Weiexport MEM_FILE=${RESULT_DIR}/mnist_hogwild/result_mem.log
#Weitaskset -c "${CORE_LIST}" bash "${CURRENT_DIR}/monitor_proc.sh" python main.py --epochs 10
#Wei# run CPU WLM LSTM
#Weimkdir -p ${RESULT_DIR}/wlm_cpu_lstm
#Weipushd "${EXAMPLES_DIR}/word_language_model"
#Weiexport LOG_FILE=${RESULT_DIR}/wlm_cpu_lstm/result.log
#Weiexport MEM_FILE=${RESULT_DIR}/wlm_cpu_lstm/result_mem.log
#Weitaskset -c "${CORE_LIST}" bash "${CURRENT_DIR}/monitor_proc.sh" python main.py --epochs 10 --model LSTM
#Wei# run GPU WLM LSTM
#Weimkdir -p ${RESULT_DIR}/wlm_gpu_lstm
#Weipushd "${EXAMPLES_DIR}/word_language_model"
#Weiexport LOG_FILE=${RESULT_DIR}/wlm_gpu_lstm/result.log
#Weiexport MEM_FILE=${RESULT_DIR}/wlm_gpu_lstm/result_mem.log
#Weitaskset -c "${CORE_LIST}" bash "${CURRENT_DIR}/monitor_proc.sh" python main.py --epochs 10 --model LSTM --cuda
#Wei# run CPU WLM Transformer
#Weimkdir -p ${RESULT_DIR}/wlm_cpu_trans
#Weipushd "${EXAMPLES_DIR}/word_language_model"
#Weiexport LOG_FILE=${RESULT_DIR}/wlm_cpu_trans/result.log
#Weiexport MEM_FILE=${RESULT_DIR}/wlm_cpu_trans/result_mem.log
#Weitaskset -c "${CORE_LIST}" bash "${CURRENT_DIR}/monitor_proc.sh" python main.py --epochs 10 --model Transformer
#Wei# run GPU WLM Transformer
#Weimkdir -p ${RESULT_DIR}/wlm_gpu_trans
#Weipushd "${EXAMPLES_DIR}/word_language_model"
#Weiexport LOG_FILE=${RESULT_DIR}/wlm_gpu_trans/result.log
#Weiexport MEM_FILE=${RESULT_DIR}/wlm_gpu_trans/result_mem.log
#Weitaskset -c "${CORE_LIST}" bash "${CURRENT_DIR}/monitor_proc.sh" python main.py --epochs 10 --model Transformer --cuda
