#!/bin/bash
set -ex

. ~/miniconda3/etc/profile.d/conda.sh

if [[ -z "${CONDA_ENV}" ]]; then
    conda activate base
else
    conda activate "${CONDA_ENV}"
fi

conda install -y numpy requests ninja pyyaml setuptools gitpython beautifulsoup4 regex
conda install -y -c pytorch magma-cuda116

# install the most recent successfully built pytorch packages
# torchaudio is required by fairseq/fambench_xlmr
pip install --pre torch torchvision torchtext torchaudio -f https://download.pytorch.org/whl/nightly/cu116/torch_nightly.html

conda install -y expecttest -c conda-forge

# Log final configuration
pip freeze
