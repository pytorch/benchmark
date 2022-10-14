# default base image: summerwind/actions-runner-dind:latest
ARG BASE_IMAGE=summerwind/actions-runner-dind:latest
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get -y update
RUN apt-get install -y git git-lfs jq \
                       vim wget curl \
                       libgl1-mesa-glx libsndfile1-dev

# Install CUDA 11.6 and cudnn
RUN wget -q https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run \
-O cuda_11.6.2_510.47.03_linux.run
RUN ./cuda_11.6.2_510.47.03_linux.run --toolkit --silent && \
    sudo rm -f cuda_11.6.2_510.47.03_linux.run
# Install Nvidia cuDNN
RUN wget -q https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.2/local_installers/11.5/cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz \
     -O cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz
RUN tar xJf cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz && \
    pushd cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive && \
    cp include/* /usr/local/cuda/include && \
    cp lib/* /usr/local/cuda/lib64 && \
    ldconfig

# Install miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    sudo su runner && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -u

# Use Python 3.8 as default
RUN sudo su runner && \
    . ${HOME}/miniconda3/etc/profile.d/conda.sh && \
    conda activate base \
    conda install -y python=3.8 \
    pip install unittest-xml-reporting
