#!/bin/bash

if [ "$1" = "-h" -o "$1" = "--help" ]; then
  echo "Usage: prog BRANCH_NAME REPO"
  exit 1
fi

if [ "$#" -ne 2 ]; then
    echo echo "Usage: prog BRANCH_NAME REPO"
    exit 1
fi

. ${CONDA_EXE%bin/conda}/etc/profile.d/conda.sh
ORIG_BRANCH_NAME=$1
REPO=$2
BRANCH_NAME=$(python -c 'import sys; print (sys.argv[1].replace("/","_"))' $ORIG_BRANCH_NAME)
echo "Your branch will be built into $BRANCH_NAME and torchbenchmark will be installed in ${BRANCH_NAME}/benchmark"
conda create -y -n $BRANCH_NAME python=3.7 2>&1 > /dev/null
test $? -eq 0 || { echo "failed to create env ${BRANCH_NAME}!"; exit; }
conda activate $BRANCH_NAME
echo "Installing PyTorch dependencies"
conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses 2>&1 > /dev/null
conda install -y -c pytorch magma-cuda100 2>&1 > /dev/null
echo "Checking out $"
git clone $REPO $BRANCH_NAME
cd $BRANCH_NAME
git checkout $ORIG_BRANCH_NAME
git submodule sync 2>&1 > /dev/null
git submodule update --init --recursive 2>&1 > /dev/null
echo "Builing PyTorch"
python setup.py install 2>&1 > /dev/null
test $? -eq 0 || { echo "PyTorch build failed!"; exit; }

git clone --recursive  https://github.com/pytorch/vision.git torchvision
cd torchvision
python setup.py install 2>&1 > /dev/null
test $? -eq 0 || { echo "torchvision build failed!"; exit; }
cd .. 

git clone --recursive https://github.com/pytorch/audio.git torchaudio
cd torchaudio
BUILD_SOX=1 python setup.py install 2>&1 > /dev/null
test $? -eq 0 || { echo "torchaudio build failed!"; exit; }
cd ..

git clone https://github.com/pytorch/text torchtext
cd torchtext
git submodule update --init --recursive
python setup.py clean install 2>&1 > /dev/null
test $? -eq 0 || { echo "torchtext build failed!"; exit; }
cd ..

git clone --recursive https://github.com/pytorch/benchmark.git
cd benchmark
python install.py

