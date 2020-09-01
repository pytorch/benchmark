#!/bin/bash

echo 'Alpha stage setup script. Read and modify before executing'
exit 1
cmake -DCMAKE_INSTALL_DIR=timing/cpp/build/eigen_headers third_party/eigen
cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON -DCMAKE_BUILD_TYPE=Release -G"Unix Makefiles" third_party/benchmark/ -DCMAKE_INSTALL_PREFIX=timing/cpp/build/benchmark_install
cmake ../.. -DPYTORCH_HOME=${PYTORCH_HOME}
