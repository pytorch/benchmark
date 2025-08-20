ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

FROM ${BASE_IMAGE} AS base

ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG TORCHBENCH_BRANCH=${TORCHBENCH_BRANCH:-main}

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies from deadsnakes/ppa repo
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl wget sudo vim python3-pip python3-distutils \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS ${GET_PIP_URL} | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

RUN --mount=type=cache,target=/root/.cache/uv \
    python3 -m pip install uv

# Checkout Torchbench and submodules
RUN git clone --recurse-submodules -b "${TORCHBENCH_BRANCH}" --single-branch \
    https://github.com/pytorch/benchmark /workspace/benchmark

WORKDIR /workspace/benchmark

# Install nightly
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \

# Install python dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Install Torchbench models
RUN python3 install.py
