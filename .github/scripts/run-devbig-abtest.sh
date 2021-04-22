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

if [ -z ${BISECT_CONDA_ENV} ]; then
    BISECT_CONDA_ENV=bisection
fi

# Allows user to specify github issue name
if [ -z ${BISECT_ISSUE} ]; then
    BISECT_ISSUE=example-issue
fi

BISECT_BASE=${HOME}/.torchbench/bisection/${BISECT_ISSUE}

. activate ${BISECT_CONDA_ENV}
# create the work directory
mkdir -p ${BISECT_BASE}/gh${GITHUB_RUN_ID}

# specify --debug to allow restart from the last failed point
python bisection.py --work-dir ${BISECT_BASE}/gh${GITHUB_RUN_ID} \
       --pytorch-src ${PYTORCH_SRC_DIR} \
       --torchbench-src ${TORCHBENCH_SRC_DIR} \
       --config ${BISECT_BASE}/config.yaml \
       --output ${BISECT_BASE}/gh${GITHUB_RUN_ID}/result.json \
       --devbig ${CONDA_ENV_NAME} \
       --debug
