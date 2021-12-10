#!/bin/sh
# This script runs TorchBench without installing the pytorch and torch dep packages
# It assumes pytorch, torchtext, and torchvision have already been installed
# Usage:
# run.sh RESULT_DIR [BENCHMARK_FILTER]
# The RESULT_DIR is required, BENCHMARK_FILTER is optional

set -xeo pipefail

# Version of the config
if [ -z "$CONFIG_VER" ]; then
    CONFIG_VER=v1
fi
CONFIG_DIR=${PWD}/torchbenchmark/score/configs/${CONFIG_VER}
CONFIG_ENV=${CONFIG_DIR}/config-${CONFIG_VER}.env
# Load environment variables
set -a;
source ${CONFIG_ENV}
set +a;

DATA_JSON_PREFIX=$(date +"%Y%m%d_%H%M%S")
if [ -z "$1" ]; then
    echo "You must specify output data dir"
    exit 1
fi
DATA_DIR="$1"
mkdir -p "${DATA_DIR}"

# Must read BENCHMARK_FILTER after loading the config
# Because config has a preset BENCHMARK_FILTER
if [ -n "$2" ]; then
    BENCHMARK_FILTER="$2"
fi

sudo nvidia-smi -ac ${GPU_FREQUENCY}
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export GOMP_CPU_AFFINITY="${CORE_LIST}"

# Comment out the ordinary benchmark steps, and replace them with LTC custom ones.
# echo "Running benchmark with filter: \"${BENCHMARK_FILTER}\""

# Run the benchmark
# for c in $(seq 1 $NUM_ITER); do
#     taskset -c "${CORE_LIST}" pytest test_bench.py -k "${BENCHMARK_FILTER}" \
#             --benchmark-min-rounds "${NUM_ROUNDS}" \
#             --benchmark-json ${DATA_DIR}/${DATA_JSON_PREFIX}_${c}.json
# done

echo "Running check_lazy.py"
# The output is a file full of JSON objects but not legit .JSON.
python check_lazy.py --output_file ${DATA_DIR}/sweep.out
# Ignore the error for this command such that we can still have the json file
# where we can then run the command by hands.
python check_lazy.py --json_to_csv ${DATA_DIR}/sweep.out --output_file ${DATA_DIR}/sweep.csv

pushd ../pytorch/lazy_tensor_core/
LTC_TS_CUDA=1 python lazy_bench.py -d cuda --fuser fuser2  --test train -x div -x hard -k resnet18 --repeat 3
LTC_TS_CUDA=1 python lazy_bench.py -d cuda --fuser fuser2  --test eval -k resnet18 --repeat 3
popd

echo "Benchmark finished successfully. Output data dir is ${DATA_DIR}."
