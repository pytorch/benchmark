#!/bin/sh

set -euo pipefail

# Sanity checks
if [[ ! -v GITHUB_RUN_ID ]]; then
    echo "GITHUB_RUN_ID is not set! Please check your environment variable. Stop."
    exit 1
fi

if [[ ! -v SCRIBE_GRAPHQL_ACCESS_TOKEN ]]; then
    echo "SCRIBE_GRAPHQL_ACCESS_TOKEN is not set! Please check your environment variable. Stop."
    exit 1
fi

# Version of the config
CONFIG_VER=v0
CONFIG_DIR=${PWD}/torchbenchmark/score/configs/${CONFIG_VER}
CONFIG_ENV=${CONFIG_DIR}/config-${CONFIG_VER}.env
CONDA_ENV_NAME=gh${GITHUB_RUN_ID}
DATA_JSON_PREFIX=$(date +"%Y%m%d_%H%M%S")
DATA_DIR=${HOME}/benchmark-results-v0.1/gh${GITHUB_RUN_ID}

# Load environment variables
set -a;
source ${CONFIG_ENV}
set +a;
sudo nvidia-smi -ac ${GPU_FREQUENCY}
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export GOMP_CPU_AFFINITY="${CORE_LIST}"

# Check if nightly builds are available
NIGHTLIES=$(python torchbenchmark/util/torch_nightly.py --packages torch torchvision torchtext)
# If failed, the script will generate empty result
if [ -z "$NIGHTLIES" ]; then
    echo "Torch, torchvision, or torchtext nightly build failed. Cancel the workflow."
    exit 1
fi

mkdir -p ${DATA_DIR}
conda create -y -q --name ${CONDA_ENV_NAME} python=${PYTHON_VERSION}
. activate ${CONDA_ENV_NAME}
conda init bash; conda run /bin/bash

cp docker/multi30k.tar.gz torchbenchmark/models/attention_is_all_you_need_pytorch
pushd torchbenchmark/models/attention_is_all_you_need_pytorch
mkdir -p .data/multi30k
tar xzvf multi30k.tar.gz -C .data --strip 1
tar xzvf .data/training.tar.gz -C .data/multi30k
tar xzvf .data/validation.tar.gz -C .data/multi30k
tar xzvf .data/mmt_task1_test2016.tar.gz -C .data/multi30k
popd

# Install nightly from pip
pip install --pre torch torchvision torchtext -f https://download.pytorch.org/whl/nightly/${CUDA_VERSION}/torch_nightly.html
# Install benchmark dependencies
python install.py

# Run the benchmark
for c in $(seq 1 $NUM_ITER); do
    taskset -c "${CORE_LIST}" pytest test_bench.py -k "${BENCHMARK_FILTER}" \
            --benchmark-min-rounds "${NUM_ROUNDS}" \
            --benchmark-json ${DATA_DIR}/${DATA_JSON_PREFIX}_${c}.json
    # Fill in circle_build_num and circle_project_reponame
    jq --arg run_id "${GITHUB_RUN_ID}" --arg config_version "githubactions-benchmark-${CONFIG_VER}-metal-fullname" \
       '.machine_info.circle_project_name=$config_version | .machine_info.circle_build_num=$run_id' \
       ${DATA_DIR}/${DATA_JSON_PREFIX}_${c}.json > ${DATA_DIR}/${DATA_JSON_PREFIX}_${c}.json.tmp
    mv ${DATA_DIR}/${DATA_JSON_PREFIX}_${c}.json.tmp ${DATA_DIR}/${DATA_JSON_PREFIX}_${c}.json
done

# Upload data to Sribe
CONFIG_NORM_FILE=${CONFIG_DIR}/${CONFIG_FILE}
TORCHBENCH_SCORE=$(python compute_score.py --configuration ${CONFIG_NORM_FILE} --benchmark_data_dir ${DATA_DIR} | awk 'NR>2' )

IFS=$'\n'
for line in $TORCHBENCH_SCORE ; do
    JSON_NAME=$(echo $line | tr -s " " | cut -d " " -f 1)
    SCORE=$(echo $line | tr -s " " | cut -d " " -f 2)
    python scripts/upload_scribe.py --pytest_bench_json ${DATA_DIR}/${JSON_NAME} \
            --torchbench_score $SCORE
done

conda deactivate
conda env remove --name ${CONDA_ENV_NAME}

echo "Benchmark finished successfully. Output data dir is benchmark-results-v0.1/gh${GITHUB_RUN_ID}."
