#!/bin/bash

set -eux

sleep 7200

python3 -c "import torch; print(torch.__version__); print(torch.version.git_version)"
python3 install.py
