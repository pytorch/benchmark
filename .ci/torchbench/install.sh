#!/bin/bash

set -eux

python3 --version
python3 -m pip --version

whoami
ls -lah /usr/local/lib/python3.12/
ls -lah /usr/local/lib/python3.12/dist-packages

sleep 3600

python3 -c "import torch; print(torch.__version__); print(torch.version.git_version)"
python3 install.py
