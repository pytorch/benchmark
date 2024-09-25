#!/bin/bash

set -xeuo pipefail

CUDA_VERSION="$1"
MAGMA_VERSION="$2"
PYTORCH_VERSION="$3"
PYTORCH_CHANNEL="$4"
WORK_DIR="$5"

GPU_FREQUENCY="1215,1410"
if [[ ${PLATFORM_NAME} == "aws_t4_metal" ]]; then
 GPU_FREQUENCY="5001,900"
fi

# get the directory of the current script
CURRENT_DIR=$(dirname -- "$0")

wget https://raw.githubusercontent.com/phohenecker/switch-cuda/master/switch-cuda.sh
. switch-cuda.sh ${CUDA_VERSION}
# re-setup the cuda soft link
if [ -e "/usr/local/cuda" ]; then
    sudo rm /usr/local/cuda
fi

conda clean --all -y
conda update --all -y


sudo ln -sf /usr/local/cuda-${CUDA_VERSION} /usr/local/cuda
conda uninstall -y pytorch torchvision pytorch-cuda
conda uninstall -y pytorch torchvision
# make sure we have a clean environment without pytorch
pip uninstall -y torch torchvision

# install magma
conda install -y -c pytorch ${MAGMA_VERSION}

# install pip version of pytorch and torchvision
if [[ ${PYTORCH_CHANNEL} == "pytorch-test" ]]; then
    pip3 install torch==${PYTORCH_VERSION} torchvision --index-url https://download.pytorch.org/whl/test/cu${CUDA_VERSION//./}
else
    pip3 install torch torchvision
fi

python -c 'import torch; print(torch.__version__); print(torch.version.git_version)'

# If torchvision is not yet available, uncomment the following and
# find a good torchvision commit to test
#temp workaround to buid torchvision before vision rc binary is available
#pushd  /tmp
#git clone https://github.com/pytorch/vision.git
#cd vision
## checkout 2022-10-05 nightly as the checkmarks shows green
#git checkout nightly && git checkout 64b14dcda9e4d283819ae69f9a60a41409aee92a
#python setup.py install
#rm -rf /tmp/vision
#popd

# tune the machine
sudo nvidia-smi -ac "${GPU_FREQUENCY}"

# Add bc utility for memory monitor_proc.sh
# sudo apt install -y bc

pip install -U py-cpuinfo psutil distro
