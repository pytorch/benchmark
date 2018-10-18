#!/bin/bash

set -x
set -e

mkdir -p build
pushd build

# Google benchmark
mkdir gbenchmark
mkdir gbenchmark_install
pushd gbenchmark
cmake -G "Unix Makefiles" ../../../../third_party/benchmark/ -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../gbenchmark_install
make -j $(nproc)
make install
popd

