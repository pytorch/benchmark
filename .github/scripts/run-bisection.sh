#!/bin/sh
# Script that run the PR bisection

set -xeo pipefail

SCRIPTPATH=$(realpath $0)
BASEDIR=$(dirname $SCRIPTPATH)

if [ -z ${PYTORCH_SRC_DIR} ]; then
    PYTORCH_SRC_DIR=${HOME}/pytorch
fi

if [ -z ${TORCHBENCH_SRC_DIR} ]; then
    echo "User must set the env TORCHBENCH_SRC_DIR to run the bisector script."
    exit 1
fi

if [ -z ${BISECT_CONDA_ENV} ]; then
    BISECT_CONDA_ENV=bisection
fi

# Allows user to specify github issue name
if [ -z ${BISECT_ISSUE} ]; then
    BISECT_ISSUE=example-issue
fi

if [ -z ${BISECT_BASE} ]; then
    echo "User must set the env BISECT_BASE to run the bisector script."
    exit 1
fi

# create the work directory
mkdir -p ${BISECT_BASE}/gh${GITHUB_RUN_ID}

. activate ${BISECT_CONDA_ENV}

# Install pytorch nightly
conda install -y -c pytorch-nightly torchtext torchvision
# Install torchbench deps
python install.py

# specify --debug to allow restart from the last failed point
python bisection.py --work-dir ${BISECT_BASE}/gh${GITHUB_RUN_ID} \
       --pytorch-src ${PYTORCH_SRC_DIR} \
       --torchbench-src ${TORCHBENCH_SRC_DIR} \
       --config ${BISECT_BASE}/config.yaml \
       --output ${BISECT_BASE}/gh${GITHUB_RUN_ID}/result.json
