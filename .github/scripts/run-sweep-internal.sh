#!/bin/sh

set -xeo pipefail

for CONFIG in /output/configs/*.txt; do
    # Create a new conda version from base
    CONFIG_BASE=$(basename ${CONFIG})
    CONFIG_ENV_NAME=$(echo ${CONFIG} | sed 's/.*-\(.*\)\.txt/\1/')
    conda create -y -q --name ${CONFIG_ENV_NAME} python=3.7
    . activate ${CONFIG_ENV_NAME}
    # Workaround of the torchtext dependency bug
    head -n -1 $CONFIG > $CONFIG.head
    pip install -r $CONFIG.head
    rm $CONFIG.head
    pip install --no-deps -r $CONFIG
    pushd /workspace/benchmark
    python install.py
    bash /workspace/benchmark/.github/scripts/run-benchmark.sh
    . activate base
    conda env remove --name ${CONFIG_ENV_NAME}
done
