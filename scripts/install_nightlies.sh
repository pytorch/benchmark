#!/bin/bash
set -e

. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# pycocotools requires numpy 1.17 https://github.com/cocodataset/cocoapi/issues/356
conda install -y numpy=1.17 requests=2.22 ninja pyyaml setuptools gitpython

# conda install -y pytorch torchvision -c pytorch-nightly
# Changing to pip to work around https://github.com/pytorch/pytorch/issues/49375

pip install --pre torch torchvision torchtext \
    --progress-bar off \
    -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html

conda install -y expecttest -c conda-forge

# Log final configuration
pip freeze
