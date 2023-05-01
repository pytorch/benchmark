# default base image: xzhao9/gcp-a100-runner-dind:latest
ARG BASE_IMAGE=xzhao9/gcp-a100-runner-dind:latest
FROM ${BASE_IMAGE}

ENV CONDA_ENV_NAME=torchbench

# Setup Conda env and CUDA
RUN git clone https://github.com/pytorch/benchmark /workspace/benchmark
RUN cd /workspace/benchmark && \
    . /workspace/setup_instance.sh && \
    python ./utils/python_utils.py --create-conda-env ${CONDA_ENV_NAME} && \
    echo "conda activate ${CONDA_ENV_NAME}" >> ${HOME}/.bashrc && \
    echo "conda activate ${CONDA_ENV_NAME}" >> /workspace/setup_instance.sh && \
    conda activate ${CONDA_ENV_NAME} && \
    sudo python ./utils/cuda_utils.py --setup-cuda-softlink

# Install PyTorch nightly and verify the date is correct
RUN cd /workspace/benchmark && \
    . /workspace/setup_instance.sh && \
    conda activate ${CONDA_ENV_NAME} && \
    python utils/cuda_utils.py --install-torch-deps && \
    python utils/cuda_utils.py --install-torch-nightly && \
    TODAY_STR=$(date +'%Y%m%d') && \
    python utils/cuda_utils.py --check-torch-nightly-version --force-date ${TODAY_STR}

# Install TorchBench
RUN cd /workspace/benchmark && \
    . /workspace/setup_instance.sh && \
    conda activate ${CONDA_ENV_NAME} && \
    python install.py
