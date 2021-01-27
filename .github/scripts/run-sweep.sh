#!/bin/sh

set -xeo pipefail

SWEEP_DIR="${HOME}/nightly-sweep"

# Run benchmark
bash ./.github/scripts/run-docker.sh \
    /workspace/benchmark/.github/scripts/run-sweep-benchmark.sh \
    ${SWEEP_DIR}

