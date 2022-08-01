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
# re-setup the cuda soft link
if [ -e "/usr/local/cuda" ]; then
    sudo rm /usr/local/cuda
fi
sudo ln -sf /usr/local/cuda-${CUDA_VERSION} /usr/local/cuda
conda uninstall -y pytorch torchvision torchtext cudatoolkit
# make sure we have a clean environment without pytorch
pip uninstlal -y torch torchvision
pip uninstall -y torch torchvision
pip uninstlal -y torch torchvision
# install cuda toolkit and dependencies
conda install -y cudatoolkit=${CUDA_VERSION}
# install magma
conda install -y -c pytorch ${MAGMA_VERSION}
# install pytorch
conda install -y -c ${PYTORCH_CHANNEL} pytorch=${PYTORCH_VERSION} torchvision torchtext
python -c 'import torch; print(torch.__version__); print(torch.version.git_version)'

# tune the machine
sudo nvidia-smi -ac "${GPU_FREQUENCY}"

pip install -U py-cpuinfo psutil distro
# check machine tuned
python "${CURRENT_DIR}/torchbenchmark/util/machine_config.py"
