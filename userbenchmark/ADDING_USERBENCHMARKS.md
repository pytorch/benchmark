# How to add a new TorchBench userbenchmark

## Overview
One design principle of TorchBench is to de-couple the models with the benchmark experiments.
Users will share the same set of models and develop their own benchmarks.
In TorchBench, we use the [userbenchmark](https://github.com/pytorch/benchmark/tree/main/userbenchmark) directory
to manage user-customized benchmarks.

## TorchBench Userbenchmark Interface

To create a new userbenchmark, you need to create a new sub-directory in
the [userbenchmark](https://github.com/pytorch/benchmark/tree/main/userbenchmark) folder.
It is your userbenchmark's home directory, and you need to create an `__init__.py` file under it.
Let's suppose the name of your benchmark is `test-userbenchmark`.

We only require users to implement one function, `def run(args: List[str]): ...` in `__init__.py`.
We expect this function should output a *TorchBench userbenchmark metrics json* file in the `$REPO/.userbenchmark/test-userbenchmark/metrics-yyyyMMddHHmmss.json`.
Where `yyyyMMddHHmmss` is the time when the metrics file is generated.

Here is an example of the metrics json file:

```
{
    "name": <benchmark-name>,
    "environ": {
        "metrics_version": "v0.1",
        "pytorch_git_version": <torch.version.git_version>
    },
    "metrics": {
        "metric1": metric-value-1,
        "metric2": metric-value-2,
        ...
    }
}
```

where `metric-value-1` and `metric-value-2` are floats that indicate the performance metrics. 

## (Optional) Integrate your userbenchmark with CI service

After implementing your TorchBench userbenchmark, you can easily integrate it into a series of TorchBench services.

### PR-level CI
If you would like to run A/B testing of your userbenchmark on a pytorch/pytorch GitHub PR, create the PR and add the
following "magic word" to your PR body:

```
RUN_TORCHBENCH: <userbenchmark-name>
```

For example, to run the userbenchmark `nvfuser`, add the following to your PR:

```
RUN_TORCHBENCH: nvfuser
```

You will see the "TorchBench CI" starts to build PyTorch and run the userbenchmark you specified.
When it succeeds, it will generate a link that points to the PyTorch HUD which shows the A/B test result dashboard.

See an example of such A/B testing result below:
```
https://hud.pytorch.org/userbenchmark_view?url=https:%2F%2Fossci-metrics.s3.amazonaws.com%2Ftorchbench-pr-test%2Fpr84626%2Fresult.csv
```

### Nightly CI

To enroll your userbenchmark in nightly CI, create a `ci.yaml` file in your userbenchmark home directory.
For example, here is the [ci.yaml](https://github.com/pytorch/benchmark/blob/main/userbenchmark/nvfuser/ci.yaml) of the nvfuser userbenchmark:

```
platform:   "gcp_a100"
schedule:   "nightly"
```

It specifies the CI platform owner wants to run (`gcp_a100`), and the frequency (`nightly`).

Currently, we only provide two platforms:

- `gcp_a100`, Google Cloud 1xNVIDIA A100 instance, featuring GPU-only workloads.
- `aws_t4_metal`, AWS g4dn.metal instance, featuring CPU-only workloads or GPU-and-CPU workloads.
   It has lower performance noise because we can do more performance tuning on metal instances.

Currently, the bisector (bisecting which PyTorch commit caused the performance metrics change) does not work for the userbenchmark. We are still working on it. 
