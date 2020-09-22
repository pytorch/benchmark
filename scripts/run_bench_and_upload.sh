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

BENCHMARK_DATA=".data"
mkdir -p ${BENCHMARK_DATA}
pytest test_bench.py --setup-show --benchmark-sort=Name --benchmark-json=${BENCHMARK_DATA}/hub.json -k "$PYTEST_FILTER"

# Token is only present for certain jobs, only upload if present
if [ -z "$SCRIBE_GRAPHQL_ACCESS_TOKEN" ]
then
    echo "Skipping benchmark upload, token is missing."
else
    python scripts/upload_scribe.py --pytest_bench_json ${BENCHMARK_DATA}/hub.json
fi
