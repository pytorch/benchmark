#!/bin/bash

set -x
set -e

# Install common dependencies
apt-get update
# TODO: Some of these may not be necessary
# TODO: libiomp also gets installed by conda, aka there's a conflict
apt-get install -y --no-install-recommends \
  gfortran \
  cmake \
  apt-transport-https \
  autoconf \
  automake \
  build-essential \
  ca-certificates \
  curl \
  git \
  libatlas-base-dev \
  libiomp-dev \
  libyaml-dev \
  libz-dev \
  libjpeg-dev \
  python \
  python-dev \
  python-setuptools \
  python-wheel \
  software-properties-common \
  sudo \
  wget \
  valgrind \
  vim

# Cleanup package manager
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

mkdir -p /opt/conda

CONDA_FILE="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"

pushd /tmp
wget -q "${CONDA_FILE}"
chmod u+x ./Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -u -b -p "$HOME/miniconda"
popd

export PATH="$HOME/miniconda/bin:$PATH"

conda update -y -n base conda

conda install -q -y numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -y -c mingfeima mkldnn
conda install -y -c intel mkl-devel
conda install -y cython

# Need the official toolchain repo to get alternate packages
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get update -y
apt-get install -y g++-5

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 50

# Cleanup package manager
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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
