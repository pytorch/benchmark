#!/bin/bash
set -e

. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

conda install -y numpy requests ninja pyyaml setuptools gitpython
conda install -y -c pytorch magma-cuda113

pip install --pre torch torchvision torchtext \
    --progress-bar off \
    -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html

conda install -y expecttest -c conda-forge

# Log final configuration
pip freeze
