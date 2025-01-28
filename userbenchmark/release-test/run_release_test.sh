#!/bin/bash

set -euo pipefail

python -c "import torch; import time; a = torch.randn([4096, 4096]).cuda(); time.sleep(60); print('done!')"  > log.txt 2>&1 &

for i in {1..120}; do
    nvidia-smi dmon -s m -c 1 -o T -i 0
    curr=$(nvidia-smi dmon -s m -c 1 -o T -i 0 | tail -n +3 | awk '{print $3}' | sort -n | tail -1 | grep -o "[0-9.]*")
    sleep 0.5
done
