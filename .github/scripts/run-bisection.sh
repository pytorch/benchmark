#!/bin/sh

set -xeuo pipefail

SCRIPTPATH=$(realpath $0)
BASEDIR=$(dirname $SCRIPTPATH)
BISECTION_BASE=${HOME}/bisection/gh${GITHUB_RUN_ID}

BISECTION_CONFIG=${BISECTION_BASE}/config.env
set -a
source $BISECTION_CONFIG
set +a

# TODO: update the code using `git pull`

# get torch_nightly.html
curl -O https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html 

python bisection.py --work-dir ${BISECTION_BASE} \
       --pytorch-src ${PYTORCH_SRC_DIR} \
       --torchbench-src ${TORCHBENCH_SRC_DIR} \
       --start ${BISECT_START} \
       --end ${BISECT_END} \
       --threshold ${BISECT_THRESHOLD} \
       --timeout 60 \
       --output ${HOME}/bisection/output.json \
       --conda-env ${BISECT_CONDA_ENV}
