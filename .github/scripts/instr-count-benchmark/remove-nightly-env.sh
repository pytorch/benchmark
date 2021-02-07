#!/bin/sh

set -xueo pipefail

BASEDIR=$(dirname $0)
set -a;
source ${BASEDIR}/config.env
set +a;

conda deactivate
conda env remove --name ${CONDA_ENV_NAME}
