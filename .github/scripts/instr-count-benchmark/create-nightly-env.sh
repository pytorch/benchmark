#!/bin/sh

set -xueo pipefail

BASEDIR=$(dirname $0)
set -a;
source ${BASEDIR}/config.env
set +a;

conda create -y -q --name ${CONDA_ENV_NAME} python=${PYTHON_VERSION}
. activate ${CONDA_ENV_NAME}

# We need Valgrind to collect instructions.
conda install -y valgrind -c conda-forge

# Install PyTorch nightly from pip
pip install --pre torch \
    -f https://download.pytorch.org/whl/nightly/${CUDA_VERSION}/torch_nightly.html

git clone https://github.com/pytorch/pytorch.git ${REPO_CHECKOUT}
