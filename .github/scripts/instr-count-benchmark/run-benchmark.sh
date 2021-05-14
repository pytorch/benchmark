#!/bin/sh

set -xueo pipefail

BASEDIR=$(dirname $0)
set -a;
source ${BASEDIR}/config.env
set +a;

. activate ${CONDA_ENV_NAME}
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
PYTORCH_GIT_VERSION=$(python -c "import torch; print(torch.version.git_version)" | head -c 7)
PYTORCH_FILE=$(python -c "import torch; print(torch.__file__)")
echo "Running instruction benchmark for pytorch-${PYTORCH_VERSION}, commit SHA: ${PYTORCH_GIT_VERSION}, PATH: ${PYTORCH_FILE}"

python -c "import os; import torch; env_path = os.getenv('CONDA_PREFIX'); assert torch.__file__.startswith(env_path), f'{torch.__file__} not in {env_path}'"

# Run the instruction count benchmark
pushd ${BENCHMARK_ROOT}
BENCHMARK_USE_DEV_SHM=1 python main.py --mode ci --destination ${RESULT_JSON}
popd

PYTHONPATH="${GIT_ROOT}:${PYTHONPATH:-}" python ${BASEDIR}/upload.py --result_json ${RESULT_JSON}
