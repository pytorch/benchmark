#!/bin/sh

set -xeuo pipefail

SCRIPTPATH=$(realpath $0)
BASEDIR=$(dirname $SCRIPTPATH)
BISECTION_BASE=${HOME}/bisection

BISECTION_CONFIG=${BISECTION_BASE}/config.env
set -a
source $BISECTION_CONFIG
set +a

pushd $PYTORCH_SRC_DIR
git pull origin master
popd

python $BASEDIR/bisection.py --pytorch-src ${PYTORCH_SRC_DIR} \
       --start ${BISECT_START} \
       --end ${BISECT_END} \
       --threshold ${BISECT_THRESHOLD}
