#!/bin/bash
set -e
. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# set credentials for git https pushing
cat > ~/.netrc <<DONE
machine github.com
login pytorchbot
password ${GITHUB_PYTORCHBOT_TOKEN}
DONE

if [ "$CIRCLE_BRANCH" = "master" ]
then
    PYTEST_FILTER=""
else
    PYTEST_FILTER="(not cyclegan) and (not (stargan and train and cpu))"
fi

BENCHMARK_DATA="`pwd`/.data"
mkdir -p ${BENCHMARK_DATA}
BENCHMARK_FILENAME=${CIRCLE_SHA1}_$(date +"%Y%m%d_%H%M%S").json
BENCHMARK_ABS_FILENAME=${BENCHMARK_DATA}/${BENCHMARK_FILENAME}
pytest test_bench.py --ignore_machine_config --setup-show --benchmark-sort=Name --benchmark-json=${BENCHMARK_ABS_FILENAME} -k "$PYTEST_FILTER"

# Compute benchmark score
TORCHBENCH_SCORE=$(python compute_score.py --configuration torchbenchmark/score/configs/v0/config-v0.yaml --benchmark_data_file ${BENCHMARK_ABS_FILENAME})
