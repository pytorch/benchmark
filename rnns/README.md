# LSTM Benchmarking

## Setup environment
Make sure you're on a machine with CUDA, torchvision, and pytorch installed. Install in the following order:
```
# Install torchvision. It comes with the pytorch stable release binary
conda install pytorch torchvision -c pytorch

# Install the latest pytorch master from source.
# It should supercede the installation from the release binary.
cd $PYTORCH_HOME
python setup.py build develop

# Check the pytorch installation version
python -c "import torch; print(torch.__version__)"
```

Test the fastrnns benchmarking scripts with the following:
`python -m fastrnns.test --rnns jit`

For most stable results, do the following:
- Set CPU Governor to performance mode (as opposed to energy save)
- Turn off turbo for all CPUs (assuming Intel CPUs)
- Shield cpus via `cset shield` when running benchmarks.

## Run benchmarks
`python -m fastrnns.bench --rnns cudnn aten jit` should give a good comparision.

## Run nvprof
`python -m fastrnns.profile --rnns aten jit` should output an nvprof file somewhere.

# OLD: RNN benchmarks

To run all the benchmarks, and get a summary view, use `python runner.py`

To run a specific benchmark, run it as a python script:
`python benchmarks/sru.py` or `python benchmarks/sequence_labeler.py`
They come with a lot of command line options for fine-tuning.

### Caveats

Use Linux for the most accurate timing. A lot of these tests only run
on CUDA.
