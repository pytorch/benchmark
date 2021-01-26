#!/bin/bash
set -e

. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# conda install -y pytorch torchvision -c pytorch-nightly
# Changing to pip to work around https://github.com/pytorch/pytorch/issues/49375
pip install numpy
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html

# separating to debug issue where when installing all 3 this error printed
# 
# UnsatisfiableError: The following specifications were found
# to be incompatible with the existing python installation in your environment:
#
# Specifications:
#
#   - torchtext -> python[version='>=2.7,<2.8.0a0|>=3.5,<3.6.0a0']

# conda install -y torchtext -c pytorch-nightly
pip install --pre torchtext -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
