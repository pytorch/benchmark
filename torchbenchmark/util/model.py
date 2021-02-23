import json
import os
import pandas as pd
import typing
from  collections.abc import Iterable
import torch
from contextlib import contextmanager
import deepcopy


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

class BenchmarkModel():
    """
    A base class for adding models to torch benchmark.
    See [Adding Models](#../models/ADDING_MODELS.md)
    """
    def __init__(self, *args, **kwargs): 
        pass

    def train(self):
        raise NotImplementedError()

    def set_eval(self):
        self._set_mode(False)

    def set_train(self):
        self._set_mode(True)

    def eval_in_nograd(self):
        return True

    def freeze_model(self, model):
        cp_model = deepcopy.copy(model)
        cp_model.eval()
        return torch.jit.freeze(cp_model, True)

    def eval_freeze(self, model, inputs):
        for i in range(niter):
            model(inputs)

    def eval(self, model, inputs):
        for i in range(niter):
            model(inputs)

    def _set_mode(self, train):
        (model, _) = self.get_module()
        model.train(train)


