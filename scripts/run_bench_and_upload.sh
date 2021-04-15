#!/bin/bash
set -e
. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

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

# configure benchmark invocation to be as fast as possible since this run isn't used for actual timing results,
# while still producing a json output to flush out bugs in compute_score.
# this will disable warmup, and run each benchmark only once to verify correctness
# this will NOT generate the result json file
pytest test_bench.py --ignore_machine_config --setup-show --benchmark-disable --benchmark-sort=Name --benchmark-json=${BENCHMARK_ABS_FILENAME} -k "$PYTEST_FILTER"
