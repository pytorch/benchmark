#!/bin/bash

INSTALL_DIR=$1

# APEX warned that non-gcc compiler wouldn't work with pytorch:w

export CC=gcc
export CXX=g++

pushd $INSTALL_DIR

echo "----------Installing pycocotools--------------"
git clone https://github.com/cocodataset/cocoapi.git
pushd cocoapi/PythonAPI
python setup.py build_ext install
popd

echo "----------Installing cityscapesScripts--------------"
git clone https://github.com/mcordts/cityscapesScripts.git
pushd cityscapesScripts
python setup.py build_ext install
popd

echo "----------Installing apex--------------"
git clone https://github.com/NVIDIA/apex.git
pushd apex
python setup.py install --cuda_ext --cpp_ext
popd

popd #INSTALL_DIR

echo "----------Installing maskrcnn-benchmark--------------"
python setup.py build develop