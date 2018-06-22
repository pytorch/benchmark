#!/bin/bash

set -x
set -e

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 COMMIT" >&2
  exit 1
fi

rm -rf /root/benchmark
git clone https://github.com/pytorch/benchmark /root/benchmark
cd /root/benchmark
git fetch --quiet --tags https://github.com/pytorch/pytorch.git +refs/heads/*:refs/remotes/origin/* +refs/pull/*:refs/remotes/origin/pr/* --depth=50
git checkout -f "$1"
