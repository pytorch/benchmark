ARG CUDA_VERSION=13.0.2
ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

FROM ${BASE_IMAGE} AS base

ARG CUDA_VERSION
ARG PYTHON_VERSION
# The logic is copied from https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile
ARG GET_PIP_URL="https://bootstrap.pypa.io/get-pip.py"

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies from deadsnakes/ppa repo. There are few
# other dependencies like libgl1-mesa-dev used by various Python module like cv2
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl wget sudo vim libgl1-mesa-dev \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS ${GET_PIP_URL} | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

RUN python3 -m pip install uv

RUN mkdir -p /workspace/benchmark
ADD . /workspace/benchmark

WORKDIR /workspace/benchmark

# Create a venv and use it instead of using --system
RUN uv venv --seed .venv
ENV PATH="/workspace/benchmark/.venv/bin:$PATH"

# Install nightly. Remember to include torchao nightly here because the
# current stable torchao 0.12.0 here doesn't work with transformers yet
RUN uv pip install --pre torch torchvision torchaudio torchao \
  --index-url https://download.pytorch.org/whl/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

# TODO (huydhn): Find a home for this requirements file. This is here to ensure
# that we are using the same numpy version as PyTorch CI
RUN uv pip install -r utils/build_requirements.txt

# Install python dependencies
RUN uv pip install -r requirements.txt

# Install TorchBench models
RUN python3 install.py

# Check the dependency
RUN uv pip list

# CI tags the nightly with the dev<date> of the torch build it installed. On an
# OSDC runner there is no docker daemon to `docker run` the image and ask it, so
# record the version in the image and let buildx export just that one file:
#   docker buildx build --target version-export --output type=local,dest=<dir>
RUN python3 -c "import torch; print(torch.__version__)" > /torch_version

FROM scratch AS version-export
COPY --from=base /torch_version /torch_version

# Keeps the default build target the real image: with no --target, docker builds
# the *last* stage, which would otherwise be the scratch one above.
FROM base AS image
