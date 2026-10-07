ARG PYTORCH_VERSION=

FROM nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}-py3

WORKDIR /workspace

RUN git clone <REPO> /workspace/maestro

# Maestro dependencies come from maestro/pyproject.toml
RUN pip install hatchling
# The base image constrains cuda-bindings/cuda-python via PIP_CONSTRAINT; cupti-python
# needs newer versions than that constraint allows, so it's overridden for this install.
RUN PIP_CONSTRAINT= pip install --no-build-isolation /workspace/maestro