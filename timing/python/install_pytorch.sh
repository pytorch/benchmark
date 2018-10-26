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
git checkout "$1"
git submodule --quiet update --init
git clean -xffd
NO_TEST=1 BUILD_CAFFE2_OPS=0 python setup.py install
popd

git clone --quiet https://github.com/pytorch/vision
pushd vision
python setup.py install
popd

rm -rf pytorch vision
