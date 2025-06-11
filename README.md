# PyTorch Benchmarks
This is a collection of open source benchmarks used to evaluate PyTorch performance.

`torchbenchmark/models` contains copies of popular or exemplary workloads which have been modified to:
*(a)* expose a standardized API for benchmark drivers, *(b)* optionally, enable backends such as torchinductor/torchscript,
 *(c)* contain a miniature version of train/test data and a dependency install script.

## Installation
The benchmark suite should be self contained in terms of dependencies,
except for the torch products which are intended to be installed separately so
different torch versions can be benchmarked.

### Using Pre-built Packages
We support Python 3.8+, and 3.11 is recommended. Conda is optional but suggested. To start with Python 3.11 in conda:
```
# Using your current conda environment:
conda install -y python=3.11

# Or, using a new conda environment:
conda create -n torchbenchmark python=3.11
conda activate torchbenchmark
```

If you are running NVIDIA GPU tests, we support both CUDA 11.8 and 12.1, and use CUDA 12.1 as default:
```
conda install -y -c pytorch magma-cuda121
```

Then install pytorch, torchvision, and torchaudio using conda:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
```
Or use pip:
(but don't mix and match pip and conda for the torch family of libs! - [see notes below](#notes))
```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

Install the benchmark suite, which will recursively install dependencies for all the models.  Currently, the repo is intended to be installed from the source tree.
```
git clone https://github.com/pytorch/benchmark
cd benchmark
python3 install.py
```

### Install torchbench as a library

if you're interested in running torchbench as a library you can

```bash
python3 install.py
pip install git+https://www.github.com/pytorch/benchmark.git
```

or 

```bash
python3 install.py
pip install . # add -e for an editable installation
```

The above

```python
import torchbenchmark.models.densenet121
model, example_inputs = torchbenchmark.models.densenet121.Model(test="eval", device="cuda", batch_size=1).get_module()
model(*example_inputs)
```

### Building From Source
Note that when building PyTorch from source, torchvision and torchaudio must also be built from source to make sure the C APIs match.

See detailed instructions to install torchvision [here](https://github.com/pytorch/vision/blob/main/CONTRIBUTING.md) and torchaudio [here](https://github.com/pytorch/audio/blob/main/CONTRIBUTING.md).
Make sure to enable CUDA (by `USE_CUDA=1`) if using CUDA.
Then,
```
git clone https://github.com/pytorch/benchmark
cd benchmark
python3 install.py
```

### Notes
- Setup steps require network connectivity - make sure to enable a proxy if needed.
- We suggest using the latest PyTorch nightly releases to run the benchmark. Stable versions are NOT tested or maintained.
- torch, torchvision, and torchaudio must all be installed from the same build process.  This means it isn't possible to mix conda torchvision
  with pip torch, or mix built-from-source torch with pip torchvision.  It's important to match even the conda channel (nightly vs regular).
  This is due to the differences in the compilation process used by different packaging systems producing incompatible Python binary extensions.

## Using a low-noise machine
Various sources of noise, such as interrupts, context switches, clock frequency scaling, etc. can all conspire to make benchmark results variable.  It's important to understand the level of noise in your setup before drawing conclusions from benchmark data.  While any machine can in principle be tuned up, the steps and end-results vary with OS, kernel, drivers, and hardware.  To this end, torchbenchmark picks a favorite machine type it can support well, and provides utilities for automated tuning on that machine.  In the future, we may support more machine types and would be happy for contributions here.

The currently supported machine type is an AWS g4dn.metal instance using Amazon Linux.  This is one of the subsets of AWS instance types that supports [processor state control](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/processor_state_control.html), with documented tuning guides for Amazon Linux.  Most if not all of these steps should be possible on Ubuntu but haven't been automated yet.

To tune your g4dn.metal Amazon Linux machine, run
```
sudo `which python` torchbenchmark/util/machine_config.py --configure
```

When running pytest (see below), the machine_config script is invoked to assert a proper configuration and log config info into the output json.  It is possible to ```--ignore_machine_config``` if running pytest without tuning is desired.


## Running Model Benchmarks
There are multiple ways for running the model benchmarks.

`test.py` offers the simplest wrapper around the infrastructure for iterating through each model and installing and executing it.

`test_bench.py` is a pytest-benchmark script that leverages the same infrastructure but collects benchmark statistics and supports pytest filtering.

`userbenchmark` allows to develop and run customized benchmarks.

In each model repo, the assumption is that the user would already have all of the torch family of packages installed (torch, torchvision, torchaudio...) but it installs the rest of the dependencies for the model.

### Using `test.py`
`python3 test.py` will execute the APIs for each model, as a sanity check.  For benchmarking, use `test_bench.py`.  It is based on unittest, and supports filtering via CLI.

For instance, to run the BERT model on CPU for the train execution mode:
```
python3 test.py -k "test_BERT_pytorch_train_cpu"
```

The test name follows the following pattern:

```
"test_" + <model_name> + "_" + {"train" | "eval" } + "_" + {"cpu" | "cuda"}
```

### Using pytest-benchmark driver
`pytest test_bench.py` invokes the benchmark driver.  See `--help` for a complete list of options.

Some useful options include:
- `--benchmark-autosave` (or other save related flags) to get .json output
- `-k <filter expression>` standard pytest filtering
- `--collect-only` only show what tests would run, useful to see what models there are or debug your filter expression
- `--cpu_only` if running on a local CPU machine and ignoring machine configuration checks

#### Examples of Benchmark Filters
- `-k "test_train[NAME-cuda]"` for a particular flavor of a particular model
- `-k "(BERT and (not cuda))"` for a more flexible approach to filtering

Note that `test_bench.py` will eventually be deprecated as the `userbenchmark` work evolve. Users are encouraged to explore and consider using [userbenchmark](#using-userbenchmark).

### Using userbenchmark

The `userbenchmark` allows you to develop your customized benchmarks with TorchBench models. Refer to the [userbenchmark instructions](https://github.com/pytorch/benchmark/blob/main/userbenchmark/ADDING_USERBENCHMARKS.md) to learn more on how you can create a new `userbenchmark`. You can then use the `run_benchmark.py` driver to drive the benchmark. e.g. `python run_benchmark.py <benchmark_name>`. Run `python run_benchmark.py —help` to find out available options.

### Using `run.py` for simple debugging or profiling
Sometimes you may want to just run train or eval on a particular model, e.g. for debugging or profiling.  Rather than relying on __main__ implementations inside each model, `run.py` provides a lightweight CLI for this purpose, building on top of the standard BenchmarkModel API.

```
python3 run.py <model> [-d {cpu,cuda}] [-t {eval,train}] [--profile]
```
Note: `<model>` can be a full, exact name, or a partial string match.

### Using torchbench models as a library

If you're interested in using torchbench as a suite of models you can test, the easiest way to integrate it into your code/ci/tests would be something like

```python
import torch
import importlib 
import sys

# If your directory looks like this_file.py, benchmark/
sys.path.append("benchmark")
model_name = "torchbenchmark.models.stable_diffusion_text_encoder" # replace this by the name of the model you're working on
module = importlib.import_module(model_name)

benchmark_cls = getattr(module, "Model", None)
benchmark = benchmark_cls(test="eval", device = "cuda") # test = train or eval device = cuda or cpu

model, example = benchmark.get_module()
model(*example)
```

## Nightly CI runs

Currently, the models run on nightly pytorch builds and push data to Meta's internal database.
The [Nightly CI](https://github.com/pytorch/benchmark/actions) publishes both
[V1](torchbenchmark/score/configs/v1/config-v1.md) and
[V0](torchbenchmark/score/configs/v0/config-v0.md) performance scores.


See [Unidash](https://www.internalfb.com/intern/unidash/dashboard/pytorch_benchmarks/torchbenchmark_v0/) (Meta-internal only)

## Adding new models

See [Adding Models](torchbenchmark/models/ADDING_MODELS.md).
