#!/bin/bash
set -e

. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

conda install -y numpy requests ninja pyyaml setuptools gitpython beautifulsoup4 regex
conda install -y -c pytorch magma-cuda116

# install the most recent successfully built pytorch packages
python torchbenchmark/util/torch_nightly.py --install-nightlies --packages torch torchvision torchtext

conda install -y expecttest -c conda-forge

# Log final configuration
pip freeze
