#!/bin/bash

set -x
set -e

pushd $HOME
apt-get update
apt-get install -y wget
apt-get install -y bzip2
apt-get install -y gcc
apt-get install -y g++
apt-get install -y git

CONDA_FILE="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"

pushd /tmp
wget -q "${CONDA_FILE}"
chmod u+x ./Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -u -b
popd

source $HOME/miniconda3/bin/activate

conda update -y -n base conda

conda install -q -y numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -y -c mingfeima mkldnn
conda install -y cython

pip install --upgrade pip
pip install future
pip install hypothesis
pip install protobuf
pip install pytest
pip install pyyaml
pip install ninja
pip install scipy
pip install pillow
pip install typing
popd
