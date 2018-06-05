from __future__ import print_function

import torch
import math
import numpy as np

from framework import GridBenchmark


def make_size(dim, size_):
    if dim == 1:
        size = size_
    else:
        size = [0] * dim
        for i in range(dim):
            size[i] = int(math.pow(size_, 1.0 / float(dim)))
        size = tuple(size)
    return size


def make_tensor(size_, dtype, cont, dim, trans):
    size = make_size(dim, size_)
    if cont:
        tv = torch.rand(size).type(dtype)
    else:
        size = [size[0]] + list(size)
        size[dim] = 18
        size = tuple(size)
        tv = torch.rand(size).type(dtype)
        tv = tv.select(dim, 0)
    if trans:
        # tv = tv.transpose(dim -2, dim -1)
        tv = tv.transpose(0, 1)
    return tv


# Left column is torch, right column is NumPy
# NB: Numpy doesn't support rsqrt
ALL_UNARY_FUNCTIONS = [
    ("cos", "cos"),
    ("sin", "sin"),
    ("acos", "arccos"),
    ("asin", "arcsin"),
    ("atan", "arctan"),
    ("exp", "exp"),
    ("log", "log"),
    ("expm1", "expm1"),
    ("cosh", "cosh"),
    ("tan", "tan"),
    ("sinh", "sinh"),
    ("tanh", "tanh"),
    ("abs", "abs"),
    ("ceil", "ceil"),
    ("floor", "floor"),
    ("round", "round"),
    ("sqrt", "sqrt"),
    ("trunc", "trunc"),
    ("erf", "erf"),
    ("log10", "log10"),
    ("log1p", "log1p"),
    ("log2", "log2"),
]


# TODO: Make args list to allow specification of order
class NumpyComparison(GridBenchmark):

    args = {
        "mag": (4, 3),
        "dim": (3, 5),
        "cont": (True, False),
        "trans": (False, True),
        "dtype": (torch.float, torch.double),
        "framework": ("Torch", "NumPy"),
        "function": ALL_UNARY_FUNCTIONS,
    }

    def setup(self, state, arg):
        size_ = int(1024 * math.pow(10, arg.mag))
        tv = make_tensor(size_, arg.dtype, arg.cont, 3, arg.trans)
        state["torch_tensor"] = tv
        state["numpy_tensor"] = state["torch_tensor"].clone().numpy()

    def teardown(self, state, arg):
        pass

    def benchmark(self, state, arg):
        if arg.framework == "Torch":
            getattr(state["torch_tensor"], arg["function"][0])()
        else:
            getattr(np, arg["function"][1])(state["numpy_tensor"])
