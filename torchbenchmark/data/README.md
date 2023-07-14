# TorchBench input data set

This directory manages a set of minimal input data and models used by the [core model set](https://github.com/pytorch/benchmark/tree/main/torchbenchmark/models)
in the pytorch/benchmark repository.

## How to download the dataset

The full list is stored in `index.yaml`. It contains two categories of inputs: `INPUT_TARBALLS` and `MODEL_PKLS`.

To download an input tarball, use the following URL:

```
https://ossci-datasets.s3.amazonaws.com/torchbench/data/<TARBALL_NAME>
```

For example, the URL to download `multi30k.tar.gz` is `https://ossci-datasets.s3.amazonaws.com/torchbench/data/multi30k.tar.gz`.

To download a model pkl, use the following URL:

```
https://ossci-datasets.s3.amazonaws.com/torchbench/models/<MODEL_PKL_NAME>
```

For example, the URL to download `drq/obs.pkl` is `https://ossci-datasets.s3.amazonaws.com/torchbench/models/drq/obs.pkl`.

## How to contribute

TorchBench core model set runs on a single batch of data repeatedly to collect performance metrics.
Therefore, we only accept minimal datasets which only contains less than 10 batches of input data.
The total input data size should be smaller than 20 MB.

If you would like to contribute new dataset, please submit a GitHub issue.
