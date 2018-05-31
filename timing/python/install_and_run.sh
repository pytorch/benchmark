#!/bin/bash

set -x
set -e


sudo mkdir /home/perf
sudo chmod 777 /home/perf
pushd /home/perf
git clone https://github.com/pytorch/pytorch
pushd pytorch
git submodule update --init
conda install -y -c intel mkl-dnn
python setup.py install
popd
git clone https://github.com/pytorch/benchmark
pushd benchmark
pushd python
pushd ubenchmarks
python run_benchmarks.py
