# default base image: xzhao9/gcp-a100-runner-dind:latest
ARG BASE_IMAGE=xzhao9/gcp-a100-runner-dind:latest
FROM ${BASE_IMAGE}

ENV CONDA_ENV=torchbench
ENV SETUP_SCRIPT=/workspace/setup_instance.sh

# Setup Conda env and CUDA
RUN git clone https://github.com/pytorch/benchmark /workspace/benchmark
RUN cd /workspace/benchmark && \
    . ${SETUP_SCRIPT} && \
    python ./utils/python_utils.py --create-conda-env ${CONDA_ENV} && \
    echo "if [ -z \${CONDA_ENV} ]; then export CONDA_ENV=${CONDA_ENV}; fi" >> /workspace/setup_instance.sh && \
    echo "conda activate \${CONDA_ENV}" >> /workspace/setup_instance.sh && \
    echo "\
export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\n\
export LIBRARY_PATH=\${CONDA_PREFIX}/lib\${LIBRARY_PATHPATH:+:\${LIBRARY_PATHPATH}}\n" >> /workspace/setup_instance.sh

RUN cd /workspace/benchmark && \
    . ${SETUP_SCRIPT} && \
    sudo python ./utils/cuda_utils.py --setup-cuda-softlink

# Install PyTorch nightly and verify the date is correct
RUN cd /workspace/benchmark && \
    . ${SETUP_SCRIPT} && \
    python utils/cuda_utils.py --install-torch-deps && \
    python utils/cuda_utils.py --install-torch-nightly && \
    TODAY_STR=$(date +'%Y%m%d') && \
    python utils/cuda_utils.py --check-torch-nightly-version --force-date ${TODAY_STR}

# Install TorchBench conda and python dependencies
RUN cd /workspace/benchmark && \
    . ${SETUP_SCRIPT} && \
    python utils/cuda_utils.py --install-torchbench-deps && \
    cc_path=$(conda run -n "${CONDA_ENV}" printenv CC) && \
    cxx_path=$(conda run -n "${CONDA_ENV}" printenv CXX) && \
    ln -s "${cc_path}" "$(dirname "$cc_path")/cc" && \
    ln -s "${cc_path}" "$(dirname "$cc_path")/gcc" && \
    ln -s "${cxx_path}" "$(dirname "$cxx_path")/c++" && \
    ln -s "${cxx_path}" "$(dirname "$cxx_path")/g++"

RUN cd /workspace/benchmark && \
    . ${SETUP_SCRIPT} && \
    python install.py
