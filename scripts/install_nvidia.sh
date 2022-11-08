#!/bin/bash
# Python3.8 is the latest tested working version with PyTorch and NumPy. 
conda create -yq -n torchbenchmark python=3.8 git-lfs
conda activate torchbenchmark
# If you don't need nightly build, switch to "-c pytorch" channel.
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch-nightly -c nvidia

# conda-forge channel version of these tools have been tested working.
conda install -yq -c conda-forge spacy sentencepiece transformers pillow psutil

# this installs all datasets and does some system setup as well
python install.py —continue_on_fail
