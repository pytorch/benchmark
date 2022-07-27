#!/bin/bash

set -xeuo pipefail

CUDA_VERSION="$1"
MAGMA_VERSION="$2"
PYTORCH_VERSION="$3"
PYTORCH_CHANNEL="$4"
WORK_DIR="$5"

GPU_FREQUENCY="5001,900"
# get the directory of the current script
CURRENT_DIR=$(dirname -- "$0")

. switch-cuda.sh ${CUDA_VERSION}
conda uninstall -y pytorch torchvision torchtext
# install magma
conda install -y -c pytorch ${MAGMA_VERSION}
# install pytorch
conda install -y -c ${PYTORCH_CHANNEL} pytorch=${PYTORCH_VERSION} torchvision torchtext
python -c 'import torch; print(torch.__version__); print(torch.version.git_version)'
sudo nvidia-smi -ac "${GPU_FREQUENCY}"

pip install -U py-cpuinfo psutil distro
# Make sure the machine is tuned
python "${CURRENT_DIR}/torchbenchmark/util/machine_config.py"

# clone the examples
if [ -e "${WORK_DIR}/examples" ]; then
    rm -rf "${WORK_DIR}/examples"
fi
git clone https://github.com/pytorch/examples.git "${WORK_DIR}/examples"
