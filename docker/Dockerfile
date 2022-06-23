# default base image: PyTorch nightly docker
# ghcr.io/pytorch:pytorch-nightly
ARG BASE_IMAGE=ghcr.io/pytorch/pytorch-nightly:latest
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# setup conda by default
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ${HOME}/.bashrc

RUN apt-get update
RUN apt-get install -y git git-lfs jq

RUN git clone https://github.com/pytorch/functorch /workspace/functorch
RUN git clone https://github.com/pytorch/torchdynamo /workspace/torchdynamo
RUN git clone https://github.com/pytorch/benchmark /workspace/benchmark

# Clone conda env
RUN conda create --name torchbench --clone base && \
    echo "conda activate torchbench" >> ${HOME}/.bashrc

# Uninstall domain packages if they pre-exist
RUN pip uninstall -y functorch torchdynamo
# Run the setup
RUN cd /workspace/benchmark; python install.py
RUN cd /workspace/functorch; python setup.py develop
RUN cd /workspace/torchdynamo; python setup.py develop
