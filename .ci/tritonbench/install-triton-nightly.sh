#!/bin/bash
if [ -z "${BASE_CONDA_ENV}" ]; then
  echo "ERROR: BASE_CONDA_ENV is not set"
  exit 1
fi

if [ -z "${CONDA_ENV}" ]; then
  echo "ERROR: CONDA_ENV is not set"
  exit 1
fi

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

CONDA_ENV=${BASE_CONDA_ENV} . "${SETUP_SCRIPT}"
conda activate "${BASE_CONDA_ENV}"
# Remove the conda env if exists
conda remove --name "${CONDA_ENV}" -y --all || true
conda create --name "${CONDA_ENV}" -y --clone "${BASE_CONDA_ENV}"
conda activate "${CONDA_ENV}"

. "${SETUP_SCRIPT}"
# Install the nightly openai/triton
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
