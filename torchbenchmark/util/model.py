import json
import os
import pandas as pd
import typing
from  collections.abc import Iterable
import torch

class BenchmarkModel():
    def __init__(self, *args, **kwargs): 
        pass

    def set_eval(self):
        self._set_mode(False)

    def set_train(self):
        self._set_mode(True)

    def _set_mode(self, train):
        (model, _) = self.get_module()
        model.train(train)


