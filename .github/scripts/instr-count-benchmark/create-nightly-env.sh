#!/bin/sh

set -xueo pipefail

BASEDIR=$(dirname $0)
set -a;
source ${BASEDIR}/config.env
set +a;

mkdir -p ${INSTRUCTION_COUNT_ROOT}

conda create -y -q --name ${CONDA_ENV_NAME} python=${PYTHON_VERSION}
. activate ${CONDA_ENV_NAME}

# For torch_nightly.py
pip install -r requirements.txt

# For benchmarks. We need CMake and Ninja for JIT-ing C++.
conda install -y numpy ninja cmake cffi typing_extensions dataclasses

# We need Valgrind to collect instructions.
conda install -y valgrind -c conda-forge

# Check if nightly builds are available
NIGHTLIES=$(python torchbenchmark/util/torch_nightly.py --packages torch)

# If failed, the script will generate empty result
# if [ -z $NIGHTLIES ]; then
#     echo "Torch nightly build failed. Cancel the workflow."
#     exit 1
# fi

# Install PyTorch nightly from pip
pip install --pre torch \
    -f https://download.pytorch.org/whl/nightly/${CUDA_VERSION}/torch_nightly.html

rm -rf ${REPO_CHECKOUT}
# git clone --depth 1 https://github.com/pytorch/pytorch.git ${REPO_CHECKOUT}
git clone https://github.com/pytorch/pytorch.git ${REPO_CHECKOUT}
cd ${REPO_CHECKOUT}
git checkout gh/taylorrobie/callgrind_scribe2

# Monkey patch Timer for BENCHMARK_USE_DEV_SHM
TORCH_ROOT=$(python -c "import os;import torch;print(os.path.dirname(torch.__file__))")
rm -rf "${TORCH_ROOT}/utils/benchmark"
cp -r "${REPO_CHECKOUT}/utils/benchmark" "${TORCH_ROOT}/utils/"
