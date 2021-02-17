#!/bin/sh

set -xeo pipefail

SCRIPTPATH=$(realpath $0)
BASEDIR=$(dirname $SCRIPTPATH)
BISECT_BASE=${HOME}/bisection/gh${GITHUB_RUN_ID}

BISECT_CONFIG=${BISECT_BASE}/config.env
set -a
source $BISECT_CONFIG
set +a

# TODO: update the code using `git pull`

# get torch_nightly.html
curl -O https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html 

. activate ${BISECT_CONDA_ENV} &> /dev/null

python bisection.py --work-dir ${BISECT_BASE} \
       --pytorch-src ${PYTORCH_SRC_DIR} \
       --torchbench-src ${TORCHBENCH_SRC_DIR} \
       --start ${BISECT_START} \
       --end ${BISECT_END} \
       --threshold ${BISECT_THRESHOLD} \
       --timeout 60 \
       --output ${BISECT_BASE}/output.json
