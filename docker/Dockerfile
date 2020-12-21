FROM torchbench/pytorch:base

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV BENCHMARK_REPO https://github.com/pytorch/benchmark

RUN /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN /opt/conda/bin/conda uninstall --force pytorch torchvision torchtext
RUN /opt/conda/bin/conda install -c pytorch-nightly pytorch torchvision torchtext

# Need to add as per https://stackoverflow.com/questions/55313610
RUN apt-get update
RUN apt-get install git jq ffmpeg libsm6 libxext6 -y

RUN git clone -b 0.1 --single-branch ${BENCHMARK_REPO} /workspace/benchmark

# use local multi30k training data to workaround the connection failure
COPY multi30k.tar.gz /workspace/benchmark/torchbenchmark/models/attention_is_all_you_need_pytorch/
RUN cd /workspace/benchmark/torchbenchmark/models/attention_is_all_you_need_pytorch/ && \
    mkdir -p .data && \
    tar xzvf multi30k.tar.gz -C .data --strip 1

RUN cd /workspace/benchmark; python install.py
