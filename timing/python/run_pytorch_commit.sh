#!/bin/bash

set -e

if [ "$#" -le 1 ]; then
  echo "Usage: $0 COMMIT" >&2
  exit 1
fi

COMMIT="$1"
LOCAL_NAME="pytorch_benchmark_cpu_""$1"

docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v `pwd`:/mnt/localdrive -w /mnt/localdrive -t "$LOCAL_NAME" /root/miniconda3/bin/python "${@:2}"
