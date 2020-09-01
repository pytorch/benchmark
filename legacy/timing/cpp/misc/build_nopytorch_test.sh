#!/bin/bash

g++ \
    -I../../third_party/benchmark/include \
    -std=c++11 \
    -Wl,-rpath-link,"$CONDA_LIB" \
    -pthread \
    -fopenmp \
    -lgomp \
    test_nopytorch.cpp \
    build/benchmark/src/libbenchmark.a \
    -o ./a.out2


OMP_NUM_THREADS=10 
numactl --cpunodebind=1 --membind=1 taskset -c 20-29 perf stat ./a.out2
