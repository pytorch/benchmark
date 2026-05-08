#!/bin/bash

set -eux

python3 -c "import torch; print(torch.__version__); print(torch.version.git_version)"
# TODO (huydhn): Look for a more permanent solution once https://github.com/pytorch/pytorch/issues/167895
# is resolved
python3 install.py --skip stable_diffusion_text_encoder stable_diffusion_unet
