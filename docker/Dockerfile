FROM torchbench/pytorch:base

ARG PYTHON_VERSION=3.7
ARG CUDA_VERSION=10.2
ARG INSTALL_CHANNEL=pytorch-nightly

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV BENCHMARK_REPO https://github.com/pytorch/benchmark

RUN /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Need to add as per https://stackoverflow.com/questions/55313610
RUN apt-get update
RUN apt-get install git jq ffmpeg libsm6 libxext6 -y

# Install pytorch, torchvision, and torchtext nightly with conda
RUN /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -y python=${PYTHON_VERSION} \
    pytorch torchvision torchtext "cudatoolkit=${CUDA_VERSION}" && \
    /opt/conda/bin/conda clean -ya
RUN /opt/conda/bin/pip install torchelastic

RUN git clone ${BENCHMARK_REPO} /workspace/benchmark
RUN cd /workspace/benchmark; python install.py
