FROM nvcr.io/nvidia/pytorch:21.06-py3

RUN rm -rf /workspace/*
WORKDIR /workspace/unet

ADD requirements.txt .
RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install --no-cache-dir -r requirements.txt
ADD . .
