#!/bin/bash

set -x
set -e

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 COMMIT" >&2
  exit 1
fi

cd $HOME
export PATH="$HOME/miniconda3/bin:$PATH"

git clone --recursive --quiet https://github.com/pytorch/pytorch
pushd pytorch
git fetch --quiet --tags https://github.com/pytorch/pytorch.git +refs/heads/*:refs/remotes/origin/* +refs/pull/*:refs/remotes/origin/pr/* --depth=50
git checkout "$1"
git submodule --quiet update --init
NO_TEST=1 BUILD_CAFFE2_OPS=0 python setup.py install
popd

git clone --quiet https://github.com/pytorch/vision
pushd vision
python setup.py install
popd
