import json
import os
import pandas as pd
from  collections.abc import Iterable
import torch
from contextlib import contextmanager
import warnings
import inspect
import os
from typing import Optional, List, Tuple
from torchbenchmark.util.extra_args import parse_args, apply_args

class PostInitProcessor(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post__init__()
        return obj

@contextmanager
def no_grad(val):
    """Some meta-learning models (e.g. maml) may need to train a target(another) model
    in inference runs
    """
    old_state = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(not val)
        yield
    finally:
        torch.set_grad_enabled(old_state)

class BenchmarkModel(metaclass=PostInitProcessor):
    DEFAULT_TRAIN_BSIZE: Optional[int] = None
    DEFAULT_EVAL_BSIZE: Optional[int] = None

    test: str
    device: str
    jit: bool
    batch_size: int
    extra_args: List[str]

    """
    A base class for adding models to torch benchmark.
    See [Adding Models](#../models/ADDING_MODELS.md)
    """
    def __init__(self, test: str, device: str, jit: bool=False, batch_size: Optional[int]=None, extra_args: List[str]=[]):
        self.test = test
        assert self.test == "train" or self.test == "eval", f"Test must be 'train' or 'eval', but get {self.test}. Please submit a bug report."
        self.device = device
        self.jit = jit
        self.batch_size = batch_size
        if not self.batch_size:
            self.batch_size = self.DEFAULT_TRAIN_BSIZE if test == "train" else self.DEFAULT_EVAL_BSIZE
            # If the model doesn't implement test or eval test
            # its DEFAULT_TRAIN_BSIZE or DEFAULT_EVAL_BSIZE will still be None
            if not self.batch_size:
                raise NotImplementedError(f"Test {test} is not implemented.")
        # Check if customizing batch size is supported
        if hasattr(self, "ALLOW_CUSTOMIZE_BSIZE") and (not getattr(self, "ALLOW_CUSTOMIZE_BSIZE")):
            if test == "train" and (not self.batch_size == self.DEFAULT_TRAIN_BSIZE):
                raise NotImplementedError("Model doesn't support customizing batch size.")
            elif test == "eval" and (not self.batch_size == self.DEFAULT_EVAL_BSIZE):
                raise NotImplementedError("Model doesn't support customizing batch size.")
        self.extra_args = extra_args

    # Run the post processing for model acceleration
    def __post__init__(self):
        # sanity checks of the options
        assert self.test == "train" or self.test == "eval", f"Test must be 'train' or 'eval', but provided {self.test}."
        self.extra_args = parse_args(self, self.extra_args)
        apply_args(self, self.extra_args)

    # Default implementation for replacing the model
    def set_module(self, new_model):
        if hasattr(self, 'model') and isinstance(self.model, torch.nn.Module):
            self.model = new_model
        else:
            raise NotImplementedError("The instance variable 'model' does not exist or is not type 'torch.nn.Module', implement your own `set_module()` function.")

    def train(self):
        raise NotImplementedError("Base type doesn't have train implementation.")

    def eval(self) -> Tuple[torch.Tensor]:
        raise NotImplementedError("Base type doesn't have eval implementation.")

    def set_eval(self):
        self._set_mode(False)

    def set_train(self):
        self._set_mode(True)

    def eval_in_nograd(self):
        return True

    def _set_mode(self, train):
        (model, _) = self.get_module()
        model.train(train)

    def check_opt_vs_noopt_jit(self):
        if not self.jit:
            return

        model_name = inspect.getfile(self.__class__).split(os.sep)[-2]
        print(f"model_name={model_name} , {inspect.getfile(self.__class__)}")
        model_blacklist = [
            'demucs', # set up issue
            'yolov3', # set up issue
            'BERT_pytorch', # set up issue
            'moco', # set up issue
            'Super_SloMo', # results don't match, might be due to the way TE CUDA handles rand?
            'attention_is_all_you_need_pytorch', # results don't match, might be due to the way TE CUDA handles rand?
        ]

        if model_name in model_blacklist:
            warnings.warn(UserWarning(f"{model_name}.get_module() doesn't support `check_results` yet!"))
            return

        # if a model doesn't support `get_module`
        # we should let it throw and then
        # override `check_results` for that model
        try:
            model, inputs = self.get_module()
        except NotImplementedError:
            warnings.warn(UserWarning(f"{model_name}.get_module() doesn't support `check_results` yet!"))
            return

        def bench_allclose(a, b):
            if isinstance(a, torch.Tensor):
                assert(isinstance(b, torch.Tensor))
                assert(a.allclose(b))
            elif isinstance(a, tuple) or isinstance (b, list):
                assert(type(a) == type(b))
                assert(len(a) == len(b))
                for i in range(len(a)):
                    bench_allclose(a[i], b[i])
            else:
                raise RuntimeError("Encountered an supported type.\n" +
                    "Please add the type or override `bench_allclose`")

       
        try:
            opt = model(*inputs)
        except Exception as e:
            print(e)
            warnings.warn(UserWarning(f"{model_name}.eval() doesn't support `check_results` yet!"))
            return
        
        # disable optimizations and force a recompilation
        # to a baseline version
        fwd = model._c._get_method("forward")
        fwd._debug_flush_compilation_cache()
        torch._C._set_graph_executor_optimize(False)
        base = model(*inputs)
        torch._C._set_graph_executor_optimize(True)

        bench_allclose(base, opt)
