#!/bin/sh

set -euo pipefail

CONFIG_VER=v0
CONFIG_DIR=${PWD}/torchbenchmark/score/configs/${CONFIG_VER}
CONFIG_ENV=${CONFIG_DIR}/config-${CONFIG_VER}.env

SWEEP_DIR="${HOME}/nightly-sweep"

# Load environment variables
set -a;
source ${CONFIG_ENV}
set +a;
sudo nvidia-smi -ac ${GPU_FREQUENCY}
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export GOMP_CPU_AFFINITY="${CORE_LIST}"

# Install multi30k data files
cp docker/multi30k.tar.gz torchbenchmark/models/attention_is_all_you_need_pytorch
pushd torchbenchmark/models/attention_is_all_you_need_pytorch
mkdir -p .data/multi30k
tar xzvf multi30k.tar.gz -C .data --strip 1
tar xzvf .data/training.tar.gz -C .data/multi30k
tar xzvf .data/validation.tar.gz -C .data/multi30k
tar xzvf .data/mmt_task1_test2016.tar.gz -C .data/multi30k
popd

for CONFIG in ${SWEEP_DIR}/configs/*.txt; do
    # Create a new conda version from base
    CONFIG_BASE=$(basename ${CONFIG})
    CONFIG_ENV_NAME=gh-$(echo ${CONFIG} | sed 's/.*-\(.*\)\.txt/\1/')
    conda create -y -q --name ${CONFIG_ENV_NAME} python=${PYTHON_VERSION}
    . activate ${CONFIG_ENV_NAME}
    # Workaround of the torchtext dependency bug
    head -n -1 $CONFIG > $CONFIG.head
    pip install -r $CONFIG.head
    rm $CONFIG.head
    pip install --no-deps -r $CONFIG
    python install.py
    # Run the benchmark
    conda init bash; conda run /bin/bash
    for c in $(seq 1 $NUM_ITER); do
        echo "Run pytorch/benchmark for ${TORCH_VER} iter ${c}"
        taskset -c "${CORE_LIST}" pytest test_bench.py -k "${BENCHMARK_FILTER}" --benchmark-min-rounds "${NUM_ROUNDS}" \
                --benchmark-json ${SWEEP_DIR}/${DATA_JSON_PREFIX}_${c}.json
        # Fill in circle_build_num and circle_project_reponame
        jq --arg run_id "${GITHUB_RUN_ID}" --arg config_version "githubactions-benchmark-${CONFIG_VER}-metal-fullname" \
           '.machine_info.circle_project_name=$config_version | .machine_info.circle_build_num=$run_id' \
           ${SWEEP_DIR}/${DATA_JSON_PREFIX}_${c}.json > ${SWEEP_DIR}/${DATA_JSON_PREFIX}_${c}.json.tmp
        mv ${SWEEP_DIR}/${DATA_JSON_PREFIX}_${c}.json.tmp ${SWEEP_DIR}/${DATA_JSON_PREFIX}_${c}.json
    done
    . activate base
    conda env remove --name ${CONFIG_ENV_NAME}
done

