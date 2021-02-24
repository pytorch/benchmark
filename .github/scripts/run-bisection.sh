#!/bin/sh

set -xeo pipefail

SCRIPTPATH=$(realpath $0)
BASEDIR=$(dirname $SCRIPTPATH)

PYTORCH_SRC_DIR=${HOME}/pytorch
TORCHBENCH_SRC_DIR=${HOME}/benchmark-main
BISECT_CONDA_ENV=bisection
BISECT_BASE=${HOME}/bisection/gh${GITHUB_RUN_ID}

# get torch_nightly.html
curl -O https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html 

. activate ${BISECT_CONDA_ENV} &> /dev/null

python bisection.py --work-dir ${BISECT_BASE} \
       --pytorch-src ${PYTORCH_SRC_DIR} \
       --torchbench-src ${TORCHBENCH_SRC_DIR} \
       --config ${BISECT_BASE}/config.yaml \
       --output ${BISECT_BASE}/output.json
