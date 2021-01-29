import json
import os
import pandas as pd
import typing
from  collections.abc import Iterable
import torch

def check_results(m, config):
    if not config.getoption("check_results"):
        return

    if isinstance(m, BenchmarkModel):
        m.check_results()

class BenchmarkModel():
    def __init__(self):
        self.eager_results = {}
        self.jit_results = {}

    def check_results(self):

        def bench_allclose(a, b):
            if isinstance(a, torch.Tensor):
                assert(isinstance(b, torch.Tensor))
                assert(a.allclose(b))
            elif isinstance(a, tuple) or isinstance (b, list):
                assert(type(a) == type(b))
                assert(len(a) == len(b))
                for i in range(len(a)):
                    bench_allclose(a[i], b[i])

        if len(self.jit_results) == 0 or len(self.eager_results) == 0:
            return

        assert(len(self.jit_results) == len(self.eager_results))

        for k, r in self.jit_results.items():
            assert(k in self.eager_results)
            bench_allclose(self.eager_results[k], r)
            


    def save_results(self, results, train: bool = False):
        if train:
            # TODO: NYI
            return
        if self.jit:
            self.jit_results[(train, self.device)] = results
        else:
            self.eager_results[(train, self.device)] = results