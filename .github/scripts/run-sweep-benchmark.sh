#!/bin/sh

set -xeo pipefail

for CONFIG in /output/configs/*; do
    # Create a new conda version from base
    CONFIG_BASE=$(basename ${CONFIG})
    CONFIG_VER=$(echo ${CONFIG} | sed 's/.*-\(.*\)\.txt/\1/')
    conda create --name ${CONFIG_VER} python=3.7
    . activate ${CONFIG_VER}
    pip install -r $CONFIG
    pushd /workspace/benchmark
    # workaround the maskrcnn_benchmark undefined symbol problem
    find . -name "*.so" -delete
    python install.py
    bash /workspace/benchmark/.github/scripts/run-benchmark.sh
done

