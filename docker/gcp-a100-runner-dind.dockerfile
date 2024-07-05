# default base image: ghcr.io/actions/actions-runner:latest
# base image: Ubuntu 22.04 jammy
# Prune CUDA to only keep gencode >= A100
ARG BASE_IMAGE=ghcr.io/actions/actions-runner:latest
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ARG OVERRIDE_GENCODE="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90a,code=sm_90a"
ARG OVERRIDE_GENCODE_CUDNN="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90a,code=sm_90a"

RUN sudo apt-get -y update && sudo apt -y update
# fontconfig: required by model doctr_det_predictor
# libjpeg and libpng: optionally required by torchvision (vision#8342)
RUN sudo apt-get install -y git jq gcc g++ \
                            vim wget curl ninja-build cmake \
                            libgl1-mesa-glx libsndfile1-dev kmod libxml2-dev libxslt1-dev \
                            fontconfig libfontconfig1-dev \
                            libpango-1.0-0 libpangoft2-1.0-0 \
                            libsdl2-dev libsdl2-2.0-0 \
                            libjpeg-dev libpng-dev zlib1g-dev

# get switch-cuda utility
RUN sudo wget -q https://raw.githubusercontent.com/phohenecker/switch-cuda/master/switch-cuda.sh -O /usr/bin/switch-cuda.sh
RUN sudo chmod +x /usr/bin/switch-cuda.sh

RUN sudo mkdir -p /workspace; sudo chown runner:runner /workspace

# GKE version: 1.28.5-gke.1217000
# NVIDIA driver version: 535.104.05
# NVIDIA drivers list available at gs://ubuntu_nvidia_packages/
# We assume that the host NVIDIA driver binaries and libraries are mapped to the docker filesystem

# Use the CUDA installation scripts from pytorch/builder
# Install CUDA 12.4 only to reduce docker size
RUN cd /workspace; git clone https://github.com/pytorch/builder.git
RUN sudo bash -c "set -x; source /workspace/builder/common/install_cuda.sh; install_124; export OVERRIDE_GENCODE=\"${OVERRIDE_GENCODE}\" OVERRIDE_GENCODE_CUDNN=\"${OVERRIDE_GENCODE_CUDNN}\"; prune_124"

# Install miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/Miniconda3-latest-Linux-x86_64.sh
RUN cd /workspace && \
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
