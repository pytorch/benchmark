#!/bin/bash

set -e

if [ "$#" -le 1 ]; then
  echo "Usage: $0 COMMIT" >&2
  exit 1
fi

COMMIT="$1"
LOCAL_NAME="pytorch_benchmark_cpu_""$1"

if [[ "$(docker images -q "$LOCAL_NAME" 2> /dev/null)" == "" ]]; then
  sudo docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -i -v `pwd`:/mnt/localdrive -w /mnt/localdrive -t "$LOCAL_NAME" /root/miniconda/bin/python "${@:2}"
else
  echo "local pytorch commit wasn't build. build it first."
fi


