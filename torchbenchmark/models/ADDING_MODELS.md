# How to add a new model

## Overview
- copy the model code into a subdirectory of models/
- prepare a mini dataset for fast benchmark iterations
- add scripts conform to our API that install the dependencies and run the model
- (optional) modify the model to support JIT compilation

## Detailed steps

### Adding the model code
The intent is to preserve the original user code as much as possible while 
adding support for a standardized interface to the benchmark suite.

In many case it is fine to simply copy the entire original repo into a subdirectory
as a starting point, paying attention to avoid the .git folder, and not to add any 
large unnecessary data files unintentionally.  

Create a new file 'origin' that contains the url to the git repo you're copying, 
so it's easy to trace the code back to where it came from.

### Preparing install.py and dependencies
Simply put, install.py should be a one stop shop to install all the dependencies
for your model, __except torch, torchvision, torchtext__ which should be assumed to 
have been installed by an outsider (the benchmark CI).

- avoid pinning packages to specific versions with == without good reason, as the
dependencies for all models get installed into the same environment currently
- usually, leverage a requirements.txt and/or the existing setup.py

If the model depends on a C or cuda extension, it may still be possible to invoke
the build steps from install.py.  Avoid getting to fancy trying to be cross-platform
compatible (e.g. windows/mac/etc., or using package managers like yum/dnf) - if it's
not easy to build, there may be easier models to target.

[Example install.py](attention-is-all-you-need-pytorch/install.py)

### Mini-dataset
By the time install.py script runs, a miniature version of the dataset is expected to be 
staged and ready for use.  It's fine to use install.py to download and prepare the data
if the download is quick.  Otherwise, prepare the dataset manually, checking in the required
artifacts and modifying the hubconf.py script as needed to use them.

- the easiest approach may be to run the dataloader an iteration, pickle its output, and check
that file in
- it's also fine to manually modify the raw data files to include only a small fraction, but make sure not to run the dataloader during train/eval measurements

TODO
- list examples of different styles
- provide table of existing datasets that can be reused 

### Creating hubconf.py
This file should define two things:
- `class Model`, with the API described below
- `__main__` function, which exercises the model APIs for local testing

Important: be deliberate about support for cpu/gpu and jit/no-jit.  In the case that
your model is instantiated in an unsupported configuration, the convention is to return
a model object from __init__ but raise NotImplementedError() from all its methods.

    class Model:
        def __init__(self, device='cpu', jit=False):
            """ Do all setup here, including JIT compilation, dataloader invocation, to device
            """
            self.device = device
            self.jit = jit
            self.model = ...
            self.example_inputs = ...

        def get_module(self):
            """ Returns model, example_inputs such that `model(*example_inputs)` works
            (optional api)
            """
            raise NotImplementedError()

        def train(self, niterations=1):
            """ Run training on model for `niterations` times.
            Leave warmup to the caller (e.g. don't do it inside)
            (optional api)
            """
            if self.jit:
                raise NotImplementedError()
            else:
                ...

        def eval(self, niterations=1):
            """ Run evaluation on model for `niterations` inputs. One iteration should be sufficient
            Leave warmup to the caller (e.g. don't do it inside)
            (optional api)
            """
            model, example_inputs = self.get_module()
            for i in range(niterations):
                model(*example_inputs)

[example hubconf.py](attention-is-all-you-need-pytorch/hubconf.py)

### JIT
As an optional step, make whatever modifications necessary to the model code to enable it to script or trace.  If doing this,
make sure to verify correctness by capturing the output of the non-jit model for a given seed/input and replicating it with the JIT
version.

[PyTorch Docs: Creating Torchscript Code](https://pytorch.org/docs/1.1.0/jit.html#creating-torchscript-code)

### Unidash
Update [this unidash page](https://www.internalfb.com/intern/unidash/dashboard/pytorch_benchmarks/hub_detail/) to include a view of your new model. 

