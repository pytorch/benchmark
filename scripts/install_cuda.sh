#!/bin/bash

set -ex

echo "Installing nvidia kernel driver"
DRIVER_FN="NVIDIA-Linux-x86_64-450.51.06.run"
wget "https://s3.amazonaws.com/ossci-linux/nvidia_driver/$DRIVER_FN"
sudo /bin/bash "$DRIVER_FN" -s --no-drm || (sudo cat /var/log/nvidia-installer.log && false)
nvidia-smi

echo "Installing CUDA 11.1 and CuDNN"
rm -rf /usr/local/cuda-11.1 /usr/local/cuda
# install CUDA 11.1 in the same container
# CUDA download archive: https://developer.nvidia.com/cuda-toolkit-archive
wget -q https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
chmod +x cuda_11.1.0_455.23.05_linux.run
./cuda_11.1.0_455.23.05_linux.run --toolkit --silent
rm -f ./cuda_11.1.0_455.23.05_linux.run
rm -f /usr/local/cuda && ln -s /usr/local/cuda-11.1 /usr/local/cuda

# install CUDA 11.1 CuDNN
# cuDNN download archive: https://developer.nvidia.com/rdp/cudnn-archive
# cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
mkdir tmp_cudnn && cd tmp_cudnn
wget -q https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/11.1_20201106/cudnn-11.1-linux-x64-v8.0.5.39.tgz -O cudnn-8.0.tgz
tar xf cudnn-8.0.tgz
cp -a cuda/include/* /usr/local/cuda/include/
cp -a cuda/lib64/* /usr/local/cuda/lib64/
cd ..
rm -rf tmp_cudnn
ldconfig
