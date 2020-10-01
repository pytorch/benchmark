# Pytorch Benchmarks
This is a collection of open source benchmarks used to evaluate pytorch performance.

`models` contains copies of popular or exemplary workloads which have been modified to
(a) expose a standardized API for benchmark drivers, (b) optionally, be JITable,
 (c) contain a miniature version of train/test data and a dependency install script.

## Installation
The benchmark suite should be self contained in terms of dependencies,
except for the torch products which are intended to be installed separately so
different torch versions can be benchmarked.

Use python 3.7 as currently there are compatibility issues with 3.8+.  Conda is optional but suggested.
`conda install -y python=3.7`

Install pytorch, torchvision and torchtext
`conda install -y pytorch torchtext torchvision -c pytorch-nightly`

Install the benchmark suite, which will recursively install dependencies for all the models
`python install.py`


### Notes
- Setup steps require connectivity, make sure to enable a proxy if needed.
- See the [CI scripts](scripts/) and their orchestration in [config.yml](.circleci/config.yml) 
for hints about how to replicate the CI environment if you have issues
- PyTorch versions before 1.6 are not compatible with all the models in torchbenchmark.  See branch [wconstab/compare_torch_versions](https://github.com/pytorch/benchmark/tree/wconstab/compare_torch_versions) for a set of models that worked back to torch 1.4.0.
- torch, torchvision, torchtext must all be installed from the same build process.  This means it isn't possible to mix conda torchtext
  with pip torch, or mix built-from-source torch with pip torchtext.  It's important to match even the conda channel (nightly vs regular).
  This is due to differences in the compilation process used by different packaging systems producing incompatible python binary extensions.


## Running Model Benchmarks
There are currently two top-level scripts for running the models.

`test.py` offers the simplest wrapper around the infrastructure for iterating through each model and installing and executing it.

test_bench.py is a pytest-benchmark script that leverages the same infrastructure but collects benchmark statistics and supports filtering ala pytest.  

In each model repo, the assumption is that the user would already have all of the torch family of packages installed (torch, torchtext, torchvision, ...) but it installs the rest of the dependencies for the model.

### Using `test.py`
`python test.py` will execute the APIs for each model, as a sanity check.  For benchmarking, use test_bench.py.  It is based on unittest, and supports filtering via CLI.

### Using pytest-benchmark driver
`pytest test_bench.py` invokes the benchmark driver.  See `--help` for a complete list of options.  

Some useful options include
- `--benchmark-autosave` (or other save related flags) to get .json output
- `-k <filter expression>` (standard pytest filtering)
- `--collect-only` only show what tests would run, useful to see what models there are or debug your filter expression

## Nightly CI runs
Currently, models run on nightly pytorch builds and push data to scuba.

See [Unidash](https://www.internalfb.com/intern/unidash/dashboard/pytorch_benchmarks/hub_detail/) (internal only)

## Adding new models

See [Adding Models](torchbenchmark/models/ADDING_MODELS.md).

## Legacy
See `legacy` for rnn benchmarks and related scripts that were previously at the top level of this repo.
