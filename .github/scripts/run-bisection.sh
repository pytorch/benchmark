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
#    An example of bisection configuration in YAML looks like this:
################ Sample YAML config ###############
# # Start and end commits
# start: a87a1c1
# end: 0ead9d5
# # 10 percent regression
# threshold: 10
# # Support increase, decrease, or both
# # increase means performance regression, decrease means performance optimization
# direction: increase
# # Test timeout in minutes
# timeout: 60
# # Only the tests specified are executed. If not specified, use the tests in the TorchBench v0 config
# tests:
#  - test_eval[yolov3-cpu-eager]

set -xeo pipefail

SCRIPTPATH=$(realpath $0)
BASEDIR=$(dirname $SCRIPTPATH)

PYTORCH_SRC_DIR=${HOME}/pytorch
TORCHBENCH_SRC_DIR=${HOME}/benchmark-main

BISECT_CONDA_ENV=bisection
BISECT_BASE=${HOME}/bisection

# get torch_nightly.html
curl -O https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html 

. activate ${BISECT_CONDA_ENV} &> /dev/null

python bisection.py --work-dir ${BISECT_BASE}/gh${GITHUB_RUN_ID} \
       --pytorch-src ${PYTORCH_SRC_DIR} \
       --torchbench-src ${TORCHBENCH_SRC_DIR} \
       --config ${BISECT_BASE}/config.yaml \
       --output ${BISECT_BASE}/gh${GITHUB_RUN_ID}/output.json \
       --debug
