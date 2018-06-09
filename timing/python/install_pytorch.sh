#!/bin/bash

set -x
set -e

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 COMMIT" >&2
  exit 1
fi

cd $HOME
export PATH="$HOME/miniconda3/bin:$PATH"

git clone --quiet https://github.com/pytorch/pytorch
pushd pytorch
git checkout "$1"
git submodule update --init
python setup.py install
popd

git clone --quiet https://github.com/pytorch/vision
pushd vision
python setup.py install
popd
