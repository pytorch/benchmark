#!/bin/bash
set -e
NAME=torchbenchmark
PYTHON_VERSION=${PYTHON_VERSION:-3.7}

source $(conda info --base)/etc/profile.d/conda.sh

# delete the old one if needed
(conda info --envs | grep -q $NAME) && conda remove --name $NAME --all -y

conda create -y -n $NAME python=$PYTHON_VERSION
conda activate $NAME
conda install -y pytorch torchtext torchvision -c pytorch-nightly
python install.py

echo
echo "To switch to the new environment, run:"
echo "    conda activate $NAME"
echo
