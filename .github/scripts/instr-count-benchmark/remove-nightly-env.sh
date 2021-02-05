#!/bin/sh

set -xueo pipefail

set -a;
source ${BASEDIR}/config.env
set +a;

conda deactivate
conda env remove --name ${CONDA_ENV_NAME}
