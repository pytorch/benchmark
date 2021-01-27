#!/bin/sh

set -eo pipefail

BENCHMARK_FILTER=$(echo ${BENCHMARK_FILTER} | xargs)
DATA_JSON_PREFIX=$(date +"%Y%m%d_%H%M%S")
export GOMP_CPU_AFFINITY="${CORE_LIST}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"

conda init bash; conda run /bin/bash

pushd /workspace/benchmark

for c in $(seq 1 $NUM_ITER); do
    echo "Run pytorch/benchmark for ${TORCH_VER} iter ${c}"
    taskset -c "${CORE_LIST}" pytest test_bench.py -k "${BENCHMARK_FILTER}" --benchmark-min-rounds "${NUM_ROUNDS}" \
                              --benchmark-json /output/${DATA_JSON_PREFIX}_${c}.json
    # Fill in circle_build_num and circle_project_reponame
    jq --arg run_id "${GITHUB_RUN_ID}" --arg config_version "githubactions-benchmark-${CONFIG_VER}-metal-fullname" \
       '.machine_info.circle_project_name=$config_version | .machine_info.circle_build_num=$run_id' \
       /output/${DATA_JSON_PREFIX}_${c}.json > /output/${DATA_JSON_PREFIX}_${c}.json.tmp
    mv /output/${DATA_JSON_PREFIX}_${c}.json.tmp /output/${DATA_JSON_PREFIX}_${c}.json
done
