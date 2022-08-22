#!/bin/bash
# This script defines the distributed CI which will be fetched and run on the AWS Cluster
# It requires the wrapper runner define the following environment variables:
# - ${BENCHMARK_DIR}: the root of TorchBench;
# - ${CONDA_ENV_DIR}: the directory of conda environment;
# - ${JOB_DIRECTORY}: the directory to save the job output json files.
set -exuo pipefail

if [ -z ${BENCHMARK_DIR} ]; then
    echo "Must set env BENCHMARK_DIR. Exit."
    exit 1
fi
if [ -z ${CONDA_ENV_DIR} ]; then
    echo "Must set env CONDA_ENV_DIR. Exit."
    exit 1
fi
if [ -z ${JOB_DIRECTORY} ]; then
    echo "Must set env JOB_DIRECTORY. Exit."
    exit 1
fi

pushd "${BENCHMARK_DIR}"
BENCHMARK_JOB="python run_benchmark.py distributed --ngpus 8 --partition train --job_dir ${CONDA_ENV_DIR}/.userbenchmark/distributed/logs"
/opt/slurm/bin/srun -p train --gpus-per-node=8 --cpus-per-task=96 --pty ${BENCHMARK_JOB}

echo "moving the output files to ${JOB_DIRECTORY}"
mv ${CONDA_ENV_DIR}/benchmark/.userbenchmark/distributed/*.json ${JOB_DIRECTORY}

# TODO: upload the result to public GitHub repo

popd
