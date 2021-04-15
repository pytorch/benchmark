#!/bin/bash
set -e

. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# conda install -y pytorch torchvision -c pytorch-nightly
# Changing to pip to work around https://github.com/pytorch/pytorch/issues/49375
pip install -q numpy
pip install -q --pre torch torchvision torchtext \
    -f https://download.pytorch.org/whl/nightly/cu112/torch_nightly.html
