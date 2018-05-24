import time
import sys
from itertools import product

__all__ = ['over', 'make_params', 'Benchmark']

PY2 = sys.version_info[0] == 2


class over(object):
    def __init__(self, *args):
        self.values = args


class AttrDict(dict):
    def __repr__(self):
        return ', '.join(k + '=' + str(v) for k, v in self.items())

    def __getattr__(self, name):
        return self[name]


def make_params(**kwargs):
    keys = list(kwargs.keys())
    iterables = [kwargs[k].values if isinstance(kwargs[k], over) else (kwargs[k],) for k in keys]
    all_values = list(product(*iterables))
    param_dicts = [AttrDict({k: v for k, v in zip(keys, values)}) for values in all_values]
    return [param_dicts]


class Benchmark(object):
    timer = time.time if PY2 else time.perf_counter
    default_params = []
    params = make_params()
    param_names = ['config']

    # NOTE: subclasses should call prepare instead of setup
    def setup(self, params):
        for k, v in self.default_params.items():
            params.setdefault(k, v)
        self.prepare(params)
