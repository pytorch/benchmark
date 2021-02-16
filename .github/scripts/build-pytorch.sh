#!/bin/sh

export USE_CUDA=1
export BUILD_CAFFE2_OPS=0
export USE_XNNPACK=0
export USE_MKLDNN=1
export USE_MKL=1
export USE_CUDNN=1
export CMAKE_PREFIX_PATH=$CONDA_PREFIX

python setup.py install
unset CMAKE_PREFIX_PATH
