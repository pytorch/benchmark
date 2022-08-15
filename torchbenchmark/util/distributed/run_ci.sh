#!/bin/bash
# This script defines the distributed CI which will be fetched and run on the AWS Cluster
# It requires the wrapper runner define the following environment variables:
# - ${BENCHMARK_DIR}: the root of TorchBench;
# - ${CONDA_ENV_DIR}: the directory of conda environment;
# - ${JOB_DIRECTORY}: the directory to save the job output json files.
set -exuo pipefail

pushd "${BENCHMARK_DIR}"
BENCHMARK_JOB="python run_benchmark.py distributed --ngpus 8 --partition train --job_dir ${CONDA_ENV_DIR}/.userbenchmark/distributed/logs"
/opt/slurm/bin/srun -p train --gpus-per-node=8 --cpus-per-task=96 --pty ${BENCHMARK_JOB}
mv ${CONDA_ENV_DIR}/benchmark/.userbenchmark/distributed/*.json ${JOB_DIRECTORY}

# upload the result to S3

popd
