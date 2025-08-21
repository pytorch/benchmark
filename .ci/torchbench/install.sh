#!/bin/bash

set -eux

python3 --version
python3 -m pip --version

python3 -c "import torch; print(torch.__version__); print(torch.version.git_version)"
python3 install.py $@
