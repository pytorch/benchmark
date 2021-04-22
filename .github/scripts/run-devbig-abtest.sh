#!/bin/sh
# Script that run the local abtest on devbig

set -eo pipefail

SCRIPTPATH=$(realpath $0)
BASEDIR=$(dirname $SCRIPTPATH)

if [ -z ${PYTORCH_SRC_DIR} ]; then
    PYTORCH_SRC_DIR=${HOME}/pytorch
fi

if [ -z ${TORCHBENCH_SRC_DIR} ]; then
    TORCHBENCH_SRC_DIR=${HOME}/benchmark
fi

if [ -z ${CONDA_ENV_NAME} ]; then
    CONDA_ENV_NAME=abtest
fi

# Allows user to specify github issue name
if [ -z ${ABTEST_ISSUE} ]; then
    ABTEST_ISSUE=example-issue
fi

ABTEST_BASE=${HOME}/.torchbench/abtest/${ABTEST_ISSUE}

. activate ${CONDA_ENV_NAME}
# create the work directory
mkdir -p ${ABTEST_BASE}

# specify --debug to allow restart from the last failed point
python bisection.py --work-dir ${ABTEST_BASE} \
       --pytorch-src ${PYTORCH_SRC_DIR} \
       --torchbench-src ${TORCHBENCH_SRC_DIR} \
       --config ${ABTEST_BASE}/config.yaml \
       --output ${ABTEST_BASE}/result.json \
       --devbig ${CONDA_ENV_NAME} \
       --debug
