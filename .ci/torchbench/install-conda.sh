#!/bin/bash

if [ -z "${CONDA_ENV}" ]; then
  echo "ERROR: CONDA_ENV is not set"
  exit 1
fi

python utils/python_utils.py --create-conda-env ${CONDA_ENV}

conda activate ${CONDA_ENV}

python utils/cuda_utils.py --install-torch-deps
python utils/cuda_utils.py --install-torch-nightly
python utils/cuda_utils.py --install-torchbench-deps