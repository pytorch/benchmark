#!/bin/bash

set -ex

echo "Installing nvidia kernel driver"
DRIVER_FN="NVIDIA-Linux-x86_64-460.39.run"
wget "https://s3.amazonaws.com/ossci-linux/nvidia_driver/$DRIVER_FN"
sudo /bin/bash "$DRIVER_FN" -s --no-drm || (sudo cat /var/log/nvidia-installer.log && false)
nvidia-smi

echo "Installing CUDA 11.2 and CuDNN"
rm -rf /usr/local/cuda-11.2 /usr/local/cuda
# install CUDA 11.1 in the same container
# CUDA download archive: https://developer.nvidia.com/cuda-toolkit-archive
CUDA_INSTALLER=cuda_11.2.2_460.32.03_linux.run
wget -q https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/$CUDA_INSTALLER
chmod +x $CUDA_INSTALLER
./$CUDA_INSTALLER --toolkit --silent
rm -f ${CUDA_INSTALLER}
rm -f /usr/local/cuda && ln -s /usr/local/cuda-11.2 /usr/local/cuda

# install CUDA 11.2 CuDNN
# cuDNN download archive: https://developer.nvidia.com/rdp/cudnn-archive
# cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
mkdir tmp_cudnn && cd tmp_cudnn
wget -q https://developer.download.nvidia.com/compute/redist/cudnn/v8.1.0/cudnn-11.2-linux-x64-v8.1.0.77.tgz -O cudnn-8.0.tgz
tar xf cudnn-8.0.tgz
cp -a cuda/include/* /usr/local/cuda/include/
cp -a cuda/lib64/* /usr/local/cuda/lib64/
cd ..
rm -rf tmp_cudnn
ldconfig
