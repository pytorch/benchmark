#!/bin/bash

set -euo pipefail

python -c "import torch; import time; a = torch.randn([4096, 4096]).cuda(); time.sleep(60); print('done!')"  > log.txt 2>&1 &

for i in {1..120}; do
    nvidia-smi pmon -s m -c 1 -o T 
    sleep 0.5
done