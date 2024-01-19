# default base image: ghcr.io/actions/actions-runner:latest
# base image: Ubuntu 22.04 jammy
ARG BASE_IMAGE=ghcr.io/actions/actions-runner:latest
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# GKE version: 1.28.5-gke.1217000
# NVIDIA driver version: 535.104.05
# NVIDIA drivers list available at gs://ubuntu_nvidia_packages/
# We assume that the host NVIDIA driver binaries and libraries are mapped to the docker filesystem

RUN sudo apt-get -y update && sudo apt -y update
# fontconfig: needed by model doctr_det_predictor
RUN sudo apt-get install -y git jq gcc g++ \
                            vim wget curl ninja-build cmake \
                            libgl1-mesa-glx libsndfile1-dev kmod libxml2-dev libxslt1-dev \
                            fontconfig libfontconfig1-dev \
                            libpango-1.0-0 libpangoft2-1.0-0 \
                            libsdl2-dev libsdl2-2.0-0

# get switch-cuda utility
RUN sudo wget -q https://raw.githubusercontent.com/phohenecker/switch-cuda/master/switch-cuda.sh -O /usr/bin/switch-cuda.sh
RUN sudo chmod +x /usr/bin/switch-cuda.sh

RUN sudo mkdir -p /workspace; sudo chown runner:runner /workspace

# Source of the CUDA installation scripts:
# https://github.com/pytorch/builder/blob/main/common/install_cuda.sh

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

# Test activate miniconda
RUN . ${HOME}/miniconda3/etc/profile.d/conda.sh && \
    conda activate base && \
    conda init

RUN echo "\
. \${HOME}/miniconda3/etc/profile.d/conda.sh\n\
conda activate base\n\
export CONDA_HOME=\${HOME}/miniconda3\n\
export CUDA_HOME=/usr/local/cuda\n\
export PATH=\${CUDA_HOME}/bin\${PATH:+:\${PATH}}\n\
export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\n\
export LIBRARY_PATH=\${CUDA_HOME}/lib64\${LIBRARY_PATHPATH:+:\${LIBRARY_PATHPATH}}\n" >> /workspace/setup_instance.sh

RUN echo ". /workspace/setup_instance.sh\n" >> ${HOME}/.bashrc
