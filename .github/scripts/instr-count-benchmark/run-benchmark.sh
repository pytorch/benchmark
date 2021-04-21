#!/bin/sh

set -xueo pipefail

BASEDIR=$(dirname $0)
set -a;
source ${BASEDIR}/config.env
set +a;

. activate ${CONDA_ENV_NAME}
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
PYTORCH_GIT_VERSION=$(python -c "import torch; print(torch.version.git_version)" | head -c 7)
echo "Running instruction benchmark for pytorch-${PYTORCH_VERSION}, commit SHA: ${PYTORCH_GIT_VERSION}"

# Run the instruction count benchmark
pushd ${BENCHMARK_ROOT}
python main.py --mode ci --destination ${RESULT_JSON}
popd

PYTHONPATH="${GIT_ROOT}:${PYTHONPATH:-}" BENCHMARK_USE_DEV_SHM=1 python ${BASEDIR}/upload.py --result_json ${RESULT_JSON}
