# CPU Userbenchmark

The `cpu` userbenchmark is target for typical user scenarios for cpu benchmark
of PyTorch. It can be used to track regression, prove performance benefit of cpu
optimizations and reproduce performance easily. 

To support this target, the `cpu` userbenchmark has implemented core binding
option, multi-instances test, gomp/iomp option and memory allocator option. And
it able to benchmark all enabled cpu features based on torchbench model, e.g.
channels-last / fx_int8 / jit with fusers and so on.

## Usage

Samilar with other userbenchmarks, `cpu` userbenchmark can be launched by
`run_benchmark.py` with below command,
```shell
python run_benchmark.py cpu <cpu userbenchmark specific parameters> <extra args supportted by torchbench model>
```

All parameters of `cpu` userbenchmark as below,
- `--device, -d` devices to run, default value is `cpu`.
- `--test, -t` tests to run, splited by comma. Default value is `eval`.
- `--model, -m` only run the specifice models, split by comma. Default value is
  `None`, means run all models.
- `--batch-size, -b` run with the specifice batch size. Default value is `None`,
  means run eith mdoel predifined default batch size.
- `--config, -c` YAML config to specify tests to run.
- `--metrics` benchmark metrics, split by comma. Current support metrics
  including `latencies`, `throughputs` and `cpu_peak_mem`, default value is
  `latencies`.
- `--output, -o` output dir. By default will create folder under
  `.userbenchmark/cpu`.
- `--timeout` limit single model test run time. Default `None` means no
  limitation.
- `--launcher` whether to use `torch.backends.xeon.run_cpu` to get the peak
  performance on Intel(R) Xeon(R) Scalable Processors.
- `--launcher-args` work with `--launcher` enabled, to provide the args of
  torch.backends.xeon.run_cpu. The detail usage of the launcher can be found by
  executing `python -m torch.backends.xeon.run_cpu --help`, or check the source
  code in
  [here](https://github.com/pytorch/pytorch/blob/main/torch/backends/xeon/run_cpu.py).
  Default value is `--throughput-mode`.
- `--dryrun` whether dryrun the userbenchmark command.

Besides those parameters provided by the `cpu` userbenchmark directly, user also
can add all supported extra args defined in
[`extra_args.py`](../../torchbenchmark/util/extra_args.py) and specific args
defined in each [`backend`](../../torchbenchmark/util/backends). For example, if
the extra arg `--precision fx_int8` with `-t eval` have been added into the test
command, it will do fx int8 inference benchmark for specifice models. And if
`torchdynamo` backend related args have been added, it will do torchdynamo
related benchmarks accordingly.

## Examples

Below command tested 2 models fx_int8 inference with batch size 8 on CLX socket
0 and 4 instances at the same time.
```shell
python run_benchmark.py cpu --model resnet50,alexnet --test eval -b 8 --precision fx_int8 --launcher --launcher-args "--node-id 0 --ninstances 4"
```
Test results can be found in `.userbenchmark/cpu/cpu-20230420004336` and
`.userbenchmark/cpu/metrics-20230420004336.json`. The `cpu` userbenchmark will
create a folder `cpu-YYmmddHHMMSS` for the test, and aggregate all test results
into `metrics-YYmmddHHMMSS.json`. The `YYmmddHHMMSS` is the time to start the
test. And for each single model test, it will create a subfolder under folder
`cpu-YYmmddHHMMSS`. For each subfolder, it contains instances logs named with
instance PID for that model test.
```shell
$ ls .userbenchmark/cpu/cpu-20230420004336
alexnet-eval/  resnet50-eval/  
$ ls .userbenchmark/cpu/cpu-20230420004336/alexnet-eval-eager/
metrics-3347653.json  metrics-3347654.json  metrics-3347655.json  metrics-3347656.json
$ cat .userbenchmark/cpu/metrics-20230420004336.json 
{
    "name": "cpu",
    "environ": {
        "pytorch_git_version": "de1114554c38322273c066c091d455519d45472d"
    },
    "metrics": {
        "alexnet-eval_latency": 58.309660750000006,
        "alexnet-eval_cmem": 0.416259765625,
        "resnet50-eval_latency": 335.04970325,
        "resnet50-eval_cmem": 0.90673828125
    }
}
```

Test with config YAML file,
```shell
python run_benchmark.py cpu -c cpu_test.yaml
```

Other typical cpu benchmark test examples.

Benchmark torchdynamo inducotr backend on cpu device can use below command,
```shell
python run_benchmark.py cpu --model resnet50 --test eval --torchdynamo inductor
```

Benchmark amp_bf16 on cpu device can use below command,
```shell
python run_benchmark.py cpu --model resnet50 --test eval --precision amp_bf16
```

Benchmark jit (`--backend torchscript`) with oneDNN fuser can use below command,
```shell
python run_benchmark.py cpu --model resnet50 --test eval --backend torchscript --fuser fuser3
```

Benchmark `float32` eager inference with `channels-last` command as below,
```shell
python run_benchmark.py cpu --model resnet50 --test eval --channels-last
```
