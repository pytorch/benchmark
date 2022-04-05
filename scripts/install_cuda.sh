#!/bin/bash

set -ex

echo "Installing nvidia kernel driver"
DRIVER_FN="NVIDIA-Linux-x86_64-470.86.run"
wget "https://s3.amazonaws.com/ossci-linux/nvidia_driver/$DRIVER_FN"
sudo /bin/bash "$DRIVER_FN" -s --no-drm || (sudo cat /var/log/nvidia-installer.log && false)
nvidia-smi

echo "Installing CUDA 11.3 and CuDNN"
rm -rf /usr/local/cuda-11.3 /usr/local/cuda
# install CUDA 11.3 in the same container
CUDA_INSTALLER=cuda_11.3.1_465.19.01_linux.run
wget -q https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/"${CUDA_INSTALLER}"
chmod +x "${CUDA_INSTALLER}"
./"${CUDA_INSTALLER}" --toolkit --silent
rm -f "${CUDA_INSTALLER}"
rm -f /usr/local/cuda && ln -s /usr/local/cuda-11.3 /usr/local/cuda

# install CUDA 11.3 CuDNN 8.3.2
# cuDNN download archive: https://developer.nvidia.com/rdp/cudnn-archive
# cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
mkdir tmp_cudnn && cd tmp_cudnn
wget -q https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.2/local_installers/11.5/cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz -O cudnn-8.3.2.tar.xz
tar xJf cudnn-8.3.2.tar.xz
cp -a cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive/include/* /usr/local/cuda/include/
cp -a cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive/lib/* /usr/local/cuda/lib64/
cd ..
rm -rf tmp_cudnn
ldconfig
