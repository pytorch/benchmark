FROM ubuntu:14.04

ENV DEBIAN_FRONTEND noninteractive

# Install common dependencies (so that this step can be cached separately)
ADD ./third_party/pytorch-dockerfiles/common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install user
ADD ./third_party/pytorch-dockerfiles/common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

# Install conda
ENV PATH /opt/conda/bin:$PATH
ENV ANACONDA_VERSION 2
ADD ./third_party/pytorch-dockerfiles/common/install_conda.sh install_conda.sh
RUN bash ./install_conda.sh && rm install_conda.sh

# Install gcc
ENV GCC_VERSION 5
ADD ./third_party/pytorch-dockerfiles/common/install_gcc.sh install_gcc.sh
RUN bash ./install_gcc.sh && rm install_gcc.sh

# Install non-standard Python versions (via Travis binaries)
ENV TRAVIS_PYTHON_VERSION 2.7.9
ENV PATH /opt/python/$TRAVIS_PYTHON_VERSION/bin:$PATH
ADD ./third_party/pytorch-dockerfiles/common/install_travis_python.sh install_travis_python.sh
RUN bash ./install_travis_python.sh && rm install_travis_python.sh

USER jenkins
CMD ["bash"]
