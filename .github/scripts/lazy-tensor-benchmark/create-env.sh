#!/bin/sh

set -xueo pipefail

conda create -y -q --name ${CONDA_ENV} python=${PYTHON_VERSION}
. activate ${CONDA_ENV_NAME}