#!/bin/bash

set -x
set -e

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 COMMIT" >&2
  exit 1
fi

COMMIT="$1"
LOCAL_NAME="pytorch_benchmark_cpu_""$1"

docker run -v `pwd`:/mnt/localdrive --restart=on-failure:3 --name "$LOCAL_NAME" -t cpuhrsch/pytorch_benchmark_cpu /bin/bash /mnt/localdrive/install_pytorch.sh "$COMMIT"
docker commit "$LOCAL_NAME" "$LOCAL_NAME"
echo "Created local pytorch install based on commit ""$COMMIT"
