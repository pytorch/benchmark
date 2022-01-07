#!/bin/bash
set -e

# Install basics
sudo apt-get install vim

# Install miniconda
CONDA=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
filename=$(basename "$CONDA")
wget "$CONDA"
chmod +x "$filename"
./"$filename" -b -u

. ~/miniconda3/etc/profile.d/conda.sh
conda activate base
# Use python3.8 by default
conda install -y python=3.8


