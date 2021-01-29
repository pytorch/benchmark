#!/bin/sh

# [FB Internal] Upload every json file to Scribe

CONFIG_FILE=/config/${CONFIG_FILE}

set -eo pipefail

if [[ ! -v SCRIBE_GRAPHQL_ACCESS_TOKEN ]]; then
    echo "SCRIBE_GRAPHQL_ACCESS_TOKEN is not set! Please check your environment variable. Stop."
    exit 1
fi

pushd /workspace/benchmark

TORCHBENCH_SCORE=$(python compute_score.py --configuration ${CONFIG_FILE} --benchmark_data_dir /output | awk 'NR>2' )

IFS=$'\n'
for line in $TORCHBENCH_SCORE ; do
    JSON_NAME=$(echo $line | tr -s " " | cut -d " " -f 1)
    SCORE=$(echo $line | tr -s " " | cut -d " " -f 2)
    python3 scripts/upload_scribe.py --pytest_bench_json /output/${JSON_NAME} \
            --torchbench_score $SCORE
done

popd
