#!/bin/bash

if [ -z "${CONDA_ENV}" ]; then
  echo "ERROR: CONDA_ENV is not set"
  exit 1
fi

mkdir workspace
cd workspace
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh && \
bash ./Miniconda3-latest-Linux-x86_64.sh -b -u
cd ..

. "${HOME}"/miniconda3/etc/profile.d/conda.sh
conda activate base
conda init

python utils/python_utils.py --create-conda-env "${CONDA_ENV}"

conda activate "${CONDA_ENV}"

python utils/cuda_utils.py --install-torch-deps
python utils/cuda_utils.py --install-torch-nightly
python utils/cuda_utils.py --install-torchbench-deps

# use the same numpy version as the build environment
pip install -r utils/build_requirements.txt
