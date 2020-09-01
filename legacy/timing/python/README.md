There are two helper scripts

```
build_pytorch_commit.sh
run_pytorch_commit.sh
```

The first can be used to build pytorch on top of a canonical remote benchmarking docker image based on a given commit.
The second can be used to run the python interpreter of a particular commit with the arguments given after the first.

To find information on the build of the docker image, look at setup/Dockerfile under this repo's root.

You can also visit [the docker hub](https://hub.docker.com/u/cpuhrsch/) where it is stored.

For example

```
usage: run.py [-h]
              [--include {CPUConvnets,CPULSTMBench,CPUNNBench,CPUUnaryBench,CUDALSTMBench,NumpyReduceComparison,NumpyUnaryComparison}]
              [--list] [--verbose {0,10,20,30,40,50}] [--dry-run]
              [--benchmark-format {console,csv}]
              [--benchmark-filter BENCHMARK_FILTER] [--benchmark-shuffle]
              [--benchmark-min-time BENCHMARK_MIN_TIME]
              [--benchmark-warmup-repetitions BENCHMARK_WARMUP_REPETITIONS]
              [--benchmark-repetitions BENCHMARK_REPETITIONS]
              [--benchmark-out BENCHMARK_OUT]

Run benchmarks.

optional arguments:
  -h, --help            show this help message and exit
  --include {CPUConvnets,CPULSTMBench,CPUNNBench,CPUUnaryBench,CUDALSTMBench,NumpyReduceComparison,NumpyUnaryComparison}
                        Run only the specified benchmark class (default: all)
  --list                List all benchmarks and their arguments (default:
                        False)
  --verbose {0,10,20,30,40,50}
                        Threshold on logging module events to ignore.Lower
                        values lead to more verbose output (default: 30)
  --dry-run             Do a dry run without collecting information (default:
                        False)
  --benchmark-format {console,csv}
                        Choose output format for stdout (default: console)
  --benchmark-filter BENCHMARK_FILTER
                        Run benchmarks which match specified regex (default:
                        None)
  --benchmark-shuffle   Shuffle all benchmark jobs before executing (default:
                        False)
  --benchmark-min-time BENCHMARK_MIN_TIME
                        Min time per benchmark (default: 5)
  --benchmark-warmup-repetitions BENCHMARK_WARMUP_REPETITIONS
                        Number of reptitions to ignore to warmup (default: 0)
  --benchmark-repetitions BENCHMARK_REPETITIONS
                        Repeat benchmark (default: 1)
  --benchmark-out BENCHMARK_OUT
                        Write benchmark results to file (default: None)
```

Detailed example docker workflow

```
sudo docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -i -v `pwd`:/mnt/localdrive --name pytorch_benchmark_cpu_b2cdd08252a089874e4f19aae3feafc7d5b98eb4 -t cpuhrsch/pytorch_benchmark_cpu /bin/bash /mnt/localdrive/install_and_run.sh b2cdd08252a089874e4f19aae3feafc7d5b98eb4
docker commit pytorch_benchmark_cpu_b2cdd08252a089874e4f19aae3feafc7d5b98eb4 pytorch_benchmark_cpu_b2cdd08252a089874e4f19aae3feafc7d5b98eb4
sudo docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -i -v `pwd`:/mnt/localdrive --cpuset-mems 0 -w /mnt/localdrive -t pytorch_benchmark_cpu_b2cdd08252a089874e4f19aae3feafc7d5b98eb4 /root/miniconda/bin/python run.py --include NumpyComparison --benchmark-max-time 1 --benchmark-repetitions 1 --benchmark-warmup-repetitions 2
```
