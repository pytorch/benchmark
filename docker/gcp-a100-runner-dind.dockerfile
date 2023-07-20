# default base image: summerwind/actions-runner-dind:latest
# base image: Ubuntu 20.04
ARG BASE_IMAGE=summerwind/actions-runner-dind:latest
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# NVIDIA Driver version: 525.105.17, GKE version: 1.26.5-gke.1200
# GKE release notes: https://cloud.google.com/kubernetes-engine/docs/release-notes#current_versions
ENV NVIDIA_VERSION="525.105.17"

RUN sudo apt-get -y update && sudo apt -y update
RUN sudo apt-get install -y git jq \
                            vim wget curl ninja-build cmake \
                            libgl1-mesa-glx libsndfile1-dev kmod libxml2-dev libxslt1-dev

# get switch-cuda utility
RUN sudo wget -q https://raw.githubusercontent.com/phohenecker/switch-cuda/master/switch-cuda.sh -O /usr/bin/switch-cuda.sh
RUN sudo chmod +x /usr/bin/switch-cuda.sh

RUN sudo mkdir -p /workspace; sudo chown runner:runner /workspace

# Download and the NVIDIA Driver files, NVIDIA Driver version is bundled together with GKE
# Install runtime libraries only, do not compile the kernel modules
# The kernel modules are already provided by the host GKE environment
RUN cd /workspace && mkdir tmp_nvidia && cd tmp_nvidia && \
    wget -q https://storage.googleapis.com/nvidia-drivers-us-public/tesla/${NVIDIA_VERSION}/NVIDIA-Linux-x86_64-${NVIDIA_VERSION}.run && \
    sudo bash ./NVIDIA-Linux-x86_64-${NVIDIA_VERSION}.run --no-kernel-modules -s --no-systemd --no-kernel-module-source --no-nvidia-modprobe

# Source of the CUDA installation scripts:
# https://github.com/pytorch/builder/blob/main/common/install_cuda.sh
# Install CUDA 11.7 and cudnn 8.5.0.96
RUN cd /workspace && wget -q https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run -O cuda_11.7.0_515.43.04_linux.run && \
    sudo bash ./cuda_11.7.0_515.43.04_linux.run --toolkit --silent && \
    rm -f cuda_11.7.0_515.43.04_linux.run
RUN cd /workspace && wget -q https://ossci-linux.s3.amazonaws.com/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz -O cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz && \
    tar xJf cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz && \
    cd cudnn-linux-x86_64-8.5.0.96_cuda11-archive && \
    sudo cp include/* /usr/local/cuda-11.7/include && \
    sudo cp lib/* /usr/local/cuda-11.7/lib64 && \
    sudo ldconfig && \
    cd .. && rm -rf cudnn-linux-x86_64-8.5.0.96_cuda11-archive && rm -f cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz

# Install CUDA 11.8 and cudnn 8.7 and NCCL 2.15
RUN cd /workspace && wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run -O cuda_11.8.0_520.61.05_linux.run && \
    sudo bash ./cuda_11.8.0_520.61.05_linux.run --toolkit --silent && \
    rm -f cuda_11.8.0_520.61.05_linux.run
RUN cd /workspace && wget -q https://developer.download.nvidia.com/compute/redist/cudnn/v8.7.0/local_installers/11.8/cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz -O cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz && \
    tar xJf cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz && \
    cd cudnn-linux-x86_64-8.7.0.84_cuda11-archive && \
    sudo cp include/* /usr/local/cuda-11.8/include && \
    sudo cp lib/* /usr/local/cuda-11.8/lib64 && \
    sudo ldconfig && \
    cd .. && rm -rf cudnn-linux-x86_64-8.7.0.84_cuda11-archive && rm -f cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
RUN cd /workspace && mkdir tmp_nccl && cd tmp_nccl && \
    wget -q https://developer.download.nvidia.com/compute/redist/nccl/v2.15.5/nccl_2.15.5-1+cuda11.8_x86_64.txz && \
    tar xf nccl_2.15.5-1+cuda11.8_x86_64.txz && \
    sudo cp -a nccl_2.15.5-1+cuda11.8_x86_64/include/* /usr/local/cuda-11.8/include/ && \
    sudo cp -a nccl_2.15.5-1+cuda11.8_x86_64/lib/* /usr/local/cuda-11.8/lib64/ && \
    cd .. && \
    rm -rf tmp_nccl && \
    sudo ldconfig

# Install CUDA CUDA 12.1 and cuDNN 8.8 and NCCL 2.17.1
RUN cd /workspace && mkdir tmp_cuda && cd tmp_cuda && \
    wget -q https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run && \
    chmod +x cuda_12.1.0_530.30.02_linux.run && \
    sudo ./cuda_12.1.0_530.30.02_linux.run --toolkit --silent && \
    cd .. && \
    rm -rf tmp_cuda && \
    sudo ldconfig
# cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
RUN cd /workspace && mkdir tmp_cudnn && cd tmp_cudnn && \
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.8.1.3_cuda12-archive.tar.xz -O cudnn-linux-x86_64-8.8.1.3_cuda12-archive.tar.xz && \
    tar xf cudnn-linux-x86_64-8.8.1.3_cuda12-archive.tar.xz && \
    sudo cp -a cudnn-linux-x86_64-8.8.1.3_cuda12-archive/include/* /usr/local/cuda-12.1/include/ && \
    sudo cp -a cudnn-linux-x86_64-8.8.1.3_cuda12-archive/lib/* /usr/local/cuda-12.1/lib64/ && \
    cd .. && \
    rm -rf tmp_cudnn && \
    sudo ldconfig
# NCCL license: https://docs.nvidia.com/deeplearning/nccl/#licenses
RUN cd /workspace && mkdir tmp_nccl && cd tmp_nccl && \
    wget -q https://developer.download.nvidia.com/compute/redist/nccl/v2.17.1/nccl_2.17.1-1+cuda12.1_x86_64.txz && \
    tar xf nccl_2.17.1-1+cuda12.1_x86_64.txz && \
    sudo cp -a nccl_2.17.1-1+cuda12.1_x86_64/include/* /usr/local/cuda-12.1/include/ && \
    sudo cp -a nccl_2.17.1-1+cuda12.1_x86_64/lib/* /usr/local/cuda-12.1/lib64/ && \
    cd .. && \
    rm -rf tmp_nccl && \
    sudo ldconfig

# Install miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/Miniconda3-latest-Linux-x86_64.sh && \
    cd /workspace && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -u

# Use Python 3.10 as default
RUN . ${HOME}/miniconda3/etc/profile.d/conda.sh && \
    conda activate base && \
    conda init && \
    conda install -y python=3.10

RUN echo "\
. \${HOME}/miniconda3/etc/profile.d/conda.sh\n\
conda activate base\n\
export CONDA_HOME=\${HOME}/miniconda3\n\
export CUDA_HOME=/usr/local/cuda\n\
export PATH=\${CUDA_HOME}/bin\${PATH:+:\${PATH}}\n\
export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\n\
export LIBRARY_PATH=\${CUDA_HOME}/lib64\${LIBRARY_PATHPATH:+:\${LIBRARY_PATHPATH}}\n" >> /workspace/setup_instance.sh

RUN echo ". /workspace/setup_instance.sh\n" >> ${HOME}/.bashrc
