# default base image: summerwind/actions-runner-dind:latest
ARG BASE_IMAGE=summerwind/actions-runner-dind:latest
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN sudo apt-get -y update
RUN sudo apt-get install -y git git-lfs jq \
                            vim wget curl ninja-build cmake \
                            libgl1-mesa-glx libsndfile1-dev

RUN sudo mkdir -p /workspace; sudo chown runner:runner /workspace

# Install CUDA 11.6 and cudnn 8.3.2.44
RUN cd /workspace && wget -q https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run -O cuda_11.6.2_510.47.03_linux.run && \
    sudo bash ./cuda_11.6.2_510.47.03_linux.run --toolkit --silent && \
    rm -f cuda_11.6.2_510.47.03_linux.run
RUN cd /workspace && wget -q https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.2/local_installers/11.5/cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz \
    -O /workspace/cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz && \
    tar xJf cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz && \
    cd cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive && \
    sudo cp include/* /usr/local/cuda/include && \
    sudo cp lib/* /usr/local/cuda/lib64 && \
    sudo ldconfig && \
    cd .. && rm -rf cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive && rm -rf cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz

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

# Setup the default CUDA version to 11.6
RUN sudo rm -f /usr/local/cuda && sudo ln -s /usr/local/cuda-11.6 /usr/local/cuda

# Install miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/Miniconda3-latest-Linux-x86_64.sh && \
    cd /workspace && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -u

# Use Python 3.10 as default
RUN . ${HOME}/miniconda3/etc/profile.d/conda.sh && \
    conda activate base && \
    conda init && \
    conda install -y python=3.10 && \
    pip install unittest-xml-reporting pyyaml

RUN echo "\
export CONDA_HOME=\${HOME}/miniconda3\n\
export NVIDIA_HOME=/usr/local/nvidia\n\
export CUDA_HOME=/usr/local/cuda\n\
export PATH=\${NVIDIA_HOME}/bin:\${CUDA_HOME}/bin\${PATH:+:\${PATH}}\n\
export LD_LIBRARY_PATH=\${NVIDIA_HOME}/lib64:\${CUDA_HOME}/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\n" >> ${HOME}/.bashrc

RUN echo "\
. \${HOME}/miniconda3/etc/profile.d/conda.sh\n\
conda activate base\n\
export CONDA_HOME=\${HOME}/miniconda3\n\
export NVIDIA_HOME=/usr/local/nvidia\n\
export CUDA_HOME=/usr/local/cuda\n\
export PATH=\${NVIDIA_HOME}/bin:\${CUDA_HOME}/bin\${PATH:+:\${PATH}}\n\
export LD_LIBRARY_PATH=\${NVIDIA_HOME}/lib64:\${CUDA_HOME}/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\n" >> /workspace/setup_instance.sh
