#!/bin/sh

set -xueo pipefail

BASEDIR=$(dirname $0)
set -a;
source ${BASEDIR}/config.env
set +a;

conda activate ${CONDA_ENV_NAME}
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
PYTORCH_GIT_VERSION=$(python -c "import torch; print(torch.version.git_version)" | head -c 7)
echo "Running instruction benchmark for pytorch-${PYTORCH_VERSION}, commit SHA: ${PYTORCH_GIT_VERSION}"

# Run the instruction count benchmark
