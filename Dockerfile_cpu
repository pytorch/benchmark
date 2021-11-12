FROM python:3.7.12-slim-buster

RUN apt-get update -y && apt-get install -y git build-essential

RUN pip install \
    torch==1.10.0+cpu \
    torchvision==0.11.1+cpu \
    torchtext \
    torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

RUN git clone https://github.com/pytorch/benchmark.git && \
    cd benchmark && \
    python install.py

RUN pip install typer
    
COPY . .

RUN chmod +x ./run_test.sh ./run_test_bench.sh

ENTRYPOINT ["python", "mlcube.py"]