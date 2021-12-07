#!/bin/sh
# Script that run the PR bisection
# Requirements:
# 1. Conda environment created using the following command:
#     conda create -y -n bisection python=3.7
#     # numpy=1.17 is from yolov3 requirements.txt, requests=2.22 is from demucs
#     conda install numpy=1.17 requests=2.22 ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six dataclasses
# 2. PyTorch git repo (specified by PYTORCH_SRC_DIR) and TorchBench git repo (specified by TORCHBENCH_SRC_DIR)
#    in clean state with the latest code.
# 3. Bisection config, in the YAML format.
#    An example of bisection configuration in YAML can be found in bisection-config.sample.yaml

set -xeo pipefail

SCRIPTPATH=$(realpath $0)
BASEDIR=$(dirname $SCRIPTPATH)

if [ -z ${PYTORCH_SRC_DIR} ]; then
    PYTORCH_SRC_DIR=${HOME}/pytorch
fi

if [ -z ${TORCHBENCH_SRC_DIR} ]; then
    TORCHBENCH_SRC_DIR=${HOME}/benchmark
fi

if [ -z ${BISECT_CONDA_ENV} ]; then
    BISECT_CONDA_ENV=bisection
fi

# Allows user to specify github issue name
if [ -z ${BISECT_ISSUE} ]; then
    BISECT_ISSUE=example-issue
fi

if [ -z ${BISECT_BASE} ]; then
    echo "You must set the env BISECT_BASE to run the bisector script."
fi
# create the work directory
mkdir -p ${BISECT_BASE}/gh${GITHUB_RUN_ID}

. activate ${BISECT_CONDA_ENV}

# specify --debug to allow restart from the last failed point
python bisection.py --work-dir ${BISECT_BASE}/gh${GITHUB_RUN_ID} \
       --pytorch-src ${PYTORCH_SRC_DIR} \
       --torchbench-src ${TORCHBENCH_SRC_DIR} \
       --config ${BISECT_BASE}/config.yaml \
       --output ${BISECT_BASE}/gh${GITHUB_RUN_ID}/result.json \
       --build_lazy true
