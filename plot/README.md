# Running

Run those in from the root repo:

```
# Create base image for benchmark builds
docker build -t pytorch_bench setup/Dockerfile_cuda
# Configure the system for benchmarking (disable HT and turbo, isolate CPUs)
python setup/bench_conf.py --setup
# Build last X commits and benchmark them
python plot/main.py
```
