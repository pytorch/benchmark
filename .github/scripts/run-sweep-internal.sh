#!/bin/sh

set -xeo pipefail

for CONFIG in /output/configs/*; do
    # Create a new conda version from base
    CONFIG_BASE=$(basename ${CONFIG})
    CONFIG_ENV_NAME=$(echo ${CONFIG} | sed 's/.*-\(.*\)\.txt/\1/')
    conda create -y -q --name ${CONFIG_ENV_NAME} python=3.7
    . activate ${CONFIG_ENV_NAME}
    pip install -r $CONFIG
    pushd /workspace/benchmark
    python install.py
    bash /workspace/benchmark/.github/scripts/run-benchmark.sh
done
