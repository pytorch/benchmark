#!/bin/bash
set -e

PYTEST_FILTER="(not cyclegan) and (not (stargan and train and cpu))"

BENCHMARK_DATA="`pwd`/.data"
mkdir -p ${BENCHMARK_DATA}
BENCHMARK_FILENAME=${CIRCLE_SHA1}_$(date +"%Y%m%d_%H%M%S").json
BENCHMARK_ABS_FILENAME=${BENCHMARK_DATA}/${BENCHMARK_FILENAME}

# configure benchmark invocation to be as fast as possible since this run isn't used for actual timing results,
# but it's still useful to see (imprecise) timing output for debug/diagnosis of adding a new model
# 'real' timing and score computation are handled by pytorch CI github actions runner with performance-tuned machine
# configure to make CI as fast as possible while still producing a json output to flush out bugs in compute_score.
# need to modify defaults in test_bench.py before possible to override runtime cmd line:
pytest test_bench.py --ignore_machine_config --setup-show --benchmark-max-time=0.001 --benchmark-min-rounds=1 --benchmark-warmup=off --benchmark-sort=Name --benchmark-json=${BENCHMARK_ABS_FILENAME} -k "$PYTEST_FILTER"

# Compute benchmark score, just to check that the script is not crashing
# real score computation is handled by pytorch CI using a performance-tuned machine
TORCHBENCH_SCORE=$(python compute_score.py --configuration torchbenchmark/score/torchbench_0.0.yaml --benchmark_data_file ${BENCHMARK_ABS_FILENAME})
