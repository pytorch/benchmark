# default base image: xzhao9/gcp-a100-runner-dind:latest
ARG BASE_IMAGE=xzhao9/gcp-a100-runner-dind:latest
FROM ${BASE_IMAGE}

ENV CONDA_ENV=torchbench

# Setup Conda env and CUDA
RUN git clone https://github.com/pytorch/benchmark /workspace/benchmark
RUN cd /workspace/benchmark && \
    . /workspace/setup_instance.sh && \
    python ./utils/python_utils.py --create-conda-env ${CONDA_ENV} && \
    echo "if [[ -z \${CONDA_ENV} ]]; then echo \"CONDA_ENV is not set\" && exit 1; else conda activate \${CONDA_ENV} fi" >> /workspace/setup_instance.sh && \
    conda activate ${CONDA_ENV} && \
    sudo python ./utils/cuda_utils.py --setup-cuda-softlink

# Install PyTorch nightly and verify the date is correct
RUN cd /workspace/benchmark && \
    . /workspace/setup_instance.sh && \
    python utils/cuda_utils.py --install-torch-deps && \
    python utils/cuda_utils.py --install-torch-nightly && \
    TODAY_STR=$(date +'%Y%m%d') && \
    python utils/cuda_utils.py --check-torch-nightly-version --force-date ${TODAY_STR}

# Install TorchBench conda and python dependencies
RUN cd /workspace/benchmark && \
    . /workspace/setup_instance.sh && \
    python utils/cuda_utils.py --install-torchbench-deps && \
    python install.py
