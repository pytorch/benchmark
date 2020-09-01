#!/bin/bash

CONDA_LIB="$(which python)"
CONDA_LIB=`dirname $CONDA_LIB`/../lib
g++ \
    -I"$PYTORCH_HOME"/torch/lib/tmp_install/include \
    -std=c++11 \
    -Wl,-rpath-link,"$CONDA_LIB" \
    -pthread \
    -fopenmp \
    -lgomp \
    -g \
    -O0 \
    test.cpp \
    "$PYTORCH_HOME"/torch/lib/tmp_install/lib/libcaffe2.so

LD_LIBRARY_PATH="$PYTORCH_HOME"/torch/lib/tmp_install/lib 
OMP_NUM_THREADS=10 
numactl --cpunodebind=1 --membind=1 taskset -c 20-29 perf stat ./a.out
