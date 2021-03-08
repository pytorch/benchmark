# Pytorch Benchmarks
This is a collection of open source benchmarks used to evaluate pytorch performance.

`torchbenchmark/models` contains copies of popular or exemplary workloads which have been modified to
(a) expose a standardized API for benchmark drivers, (b) optionally, be JITable,
 (c) contain a miniature version of train/test data and a dependency install script.

## Installation
The benchmark suite should be self contained in terms of dependencies,
except for the torch products which are intended to be installed separately so
different torch versions can be benchmarked.

Use python 3.7 as currently there are compatibility issues with 3.8+.  Conda is optional but suggested.  To switch to python 3.7 in conda:
```
# using your current conda enviroment:
conda install -y python=3.7

# or, using a new conda environment
conda create -n torchbenchmark python=3.7
conda activate torchbenchmark
```

Install pytorch, torchvision and torchtext using conda:
```
conda install -y pytorch torchtext torchvision -c pytorch-nightly
```
or use pip:
(but don't mix and match pip and conda for the torch family of libs! - see note below)
```
pip install numpy
pip install --pre torch torchvision torchtext -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
```

Install the benchmark suite, which will recursively install dependencies for all the models.  Currently, the repo is intended to be installed from the source tree.
```
git clone <benchmark>
cd <benchmark>
python install.py
```


### Notes
- Setup steps require connectivity, make sure to enable a proxy if needed.
- See the [CI scripts](scripts/) and their orchestration in [config.yml](.circleci/config.yml) 
for hints about how to replicate the CI environment if you have issues
- PyTorch versions before 1.6 are not compatible with all the models in torchbenchmark.  See branch [wconstab/compare_torch_versions](https://github.com/pytorch/benchmark/tree/wconstab/compare_torch_versions) for a set of models that worked back to torch 1.4.0.
- torch, torchvision, torchtext must all be installed from the same build process.  This means it isn't possible to mix conda torchtext
  with pip torch, or mix built-from-source torch with pip torchtext.  It's important to match even the conda channel (nightly vs regular).
  This is due to differences in the compilation process used by different packaging systems producing incompatible python binary extensions.

## Using a low-noise machine
Various sources of noise, such as interrupts, context switches, clock frequency scaling, etc. can all conspire to make benchmark results variable.  It's important to understand the level of noise in your setup before drawing conclusions from benchmark data.  While any machine can in principle be tuned up, the steps and end-results vary with OS, kernel, drivers, and hardware.  To this end, torchbenchmark picks a favorite machine type it can support well, and provides utilities for automating tuning on that machine.  In the future, we may support more machine types and would be happy for contributions here.

The currently supported machine type is an AWS g4dn.metal instance using Amazon Linux.  This is one of the subset of AWS instance types that supports [processor state control](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/processor_state_control.html), with documented tuning guides for Amazon Linux.  Most or all of these steps should be possible on Ubuntu but haven't been automated yet.

To tune your g4dn.metal Amazon Linux machine, run
```
sudo `which python` torchbenchmark/util/machine_config.py --configure
```

When running pytest (see below), the machine_config script is invoked to assert a proper configuration and log config info into the output json.  It is possible to ```--ignore_machine_config``` if running pytest without tuning is desired.


## Running Model Benchmarks
There are currently two top-level scripts for running the models.

`test.py` offers the simplest wrapper around the infrastructure for iterating through each model and installing and executing it.

test_bench.py is a pytest-benchmark script that leverages the same infrastructure but collects benchmark statistics and supports filtering ala pytest.  

In each model repo, the assumption is that the user would already have all of the torch family of packages installed (torch, torchtext, torchvision, ...) but it installs the rest of the dependencies for the model.

### Using `test.py`
`python test.py` will execute the APIs for each model, as a sanity check.  For benchmarking, use test_bench.py.  It is based on unittest, and supports filtering via CLI.

For instance, to run the BERT model on CPU for the example execution mode:
```
python test.py -k "test_BERT_pytorch_example_cpu"
```

The test name follows the following pattern:

```
"test_" + <model_name> + "_" + {"example" | "train" | "eval" } + "_" + {"cpu" | "cuda"}
```

### Using pytest-benchmark driver
`pytest test_bench.py` invokes the benchmark driver.  See `--help` for a complete list of options.  

Some useful options include
- `--benchmark-autosave` (or other save related flags) to get .json output
- `-k <filter expression>` (standard pytest filtering)
- `--collect-only` only show what tests would run, useful to see what models there are or debug your filter expression

## Nightly CI runs
Currently, models run on nightly pytorch builds and push data to scuba.

See [Unidash](https://www.internalfb.com/intern/unidash/dashboard/pytorch_benchmarks/torchbenchmark_v0/) (internal only)

## Adding new models

See [Adding Models](torchbenchmark/models/ADDING_MODELS.md).

## Legacy
See `legacy` for rnn benchmarks and related scripts that were previously at the top level of this repo.
