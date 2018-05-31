How to run:

1. Get these modules:

```
(yes | module clear 2> /dev/null) && module purge
module load git
module load cuda
module load cudnn
module load llvm
module load vim
module load gcc/5.3.0
```

2. Install a Conda environment

3. Install the libstdc++ ATen was built with: `conda install libstdcxx-ng`

4. Install ATen: `conda install -c ezyang aten`

5. `make run_lstm`

Look in the Makefile if you want to try running things manually.
All parameters are hard coded so you'll have to edit the C++ to use it.

To build `lstm_variable.exe`, you need an env var saying where PyTorch
build tree is:

```
PYTORCH_HOME=$PATH_TO_PYTORCH make run_lstm_variable
```
