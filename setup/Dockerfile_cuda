FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive

ADD ./install_dependencies.sh install_dependencies.sh
RUN bash ./install_dependencies.sh && rm ./install_dependencies.sh

ENV PATH="/root/miniconda3/bin:${PATH}"

USER root
CMD ["bash"]
