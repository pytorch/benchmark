#!/usr/bin/env bash
set -ex -o pipefail

# Setup NVIDIA Driver
if ! lsmod | grep -q nvidia; then
  DRIVER_FN="NVIDIA-Linux-x86_64-515.76.run"
  wget "https://s3.amazonaws.com/ossci-linux/nvidia_driver/$DRIVER_FN"
  sudo /bin/bash "$DRIVER_FN" -s --no-drm || (sudo cat /var/log/nvidia-installer.log && false)
  nvidia-smi
fi

# Setup CUDA 11.7 and cuDNN 8.5.0.96
wget -q https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run -O cuda_11.7.0_515.43.04_linux.run
sudo bash ./cuda_11.7.0_515.43.04_linux.run --toolkit --silent
wget -q https://ossci-linux.s3.amazonaws.com/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz -O cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
tar xJf cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
cd cudnn-linux-x86_64-8.5.0.96_cuda11-archive && \
sudo cp include/* /usr/local/cuda-11.7/include && \
sudo cp lib/* /usr/local/cuda-11.7/lib64 && \
sudo ldconfig
