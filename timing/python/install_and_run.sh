#!/bin/bash

set -x
set -e

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 COMMIT" >&2
  exit 1
fi

export PATH="$HOME/miniconda/bin:$PATH"

git clone --quiet https://github.com/pytorch/pytorch
pushd pytorch

# Fetching upstream changes from https://github.com/pytorch/pytorch.git
git fetch --tags --quiet https://github.com/pytorch/pytorch.git +refs/heads/*:refs/remotes/origin/* +refs/pull/*:refs/remotes/origin/pr/* --depth=50
git checkout -f "$1"

git submodule --quiet update --init
python setup.py --quiet install
popd

git clone https://github.com/pytorch/vision
pushd vision
python setup.py install
popd

git clone https://github.com/pytorch/benchmark
pushd benchmark
pushd timing
pushd python
python run.py
