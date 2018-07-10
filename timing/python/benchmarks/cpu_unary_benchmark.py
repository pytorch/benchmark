from __future__ import print_function

import torch
import math
import numpy as np

from framework import Benchmark
from framework import utils


# TODO: Split into additional benchmarks specific to torch only (sigmoid etc.)


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
    ("abs", "abs"),
    ("acos", "arccos"),
    ("asin", "arcsin"),
    ("atan", "arctan"),
    ("ceil", "ceil"),
    ("cos", "cos"),
    ("cosh", "cosh"),
    ("erf", "erf"),  # Needs Intel Numpy (conda)
    ("exp", "exp"),
    ("expm1", "expm1"),
    ("floor", "floor"),
    ("log", "log"),
    ("log10", "log10"),
    ("log1p", "log1p"),
    ("log2", "log2"),
    ("round", "round"),
    ("sin", "sin"),
    ("sinh", "sinh"),
    ("sqrt", "sqrt"),
    ("tan", "tan"),
    ("tanh", "tanh"),
    ("trunc", "trunc"),
]

TORCH_ONLY_FUNCTIONS = [("sigmoid", None), ("rsqrt", None)]


def _setupRun(self, state, arg):
    size_ = int(math.pow(10, arg.mag))
    tv = make_tensor(size_, arg.dtype, arg.cont, arg.dim, arg.trans)
    state.torch_tensor = tv
    state.output = tv.clone()
    state.sizes = tv.size()
    state.strides = tv.stride()
    if arg.framework == "NumPy":
        if arg.dtype == torch.float:
            state.numpy_tensor = state.torch_tensor.numpy()
            assert state.numpy_tensor.dtype == np.float32
        if arg.dtype == torch.double:
            state.numpy_tensor = state.torch_tensor.numpy()
            assert state.numpy_tensor.dtype == np.float64
        state.output = state.numpy_tensor.copy()


def _benchmark(self, state, arg):
    if arg.framework == "Torch":
        getattr(torch, arg.function[0])(state.torch_tensor, out=state.output)
    else:
        getattr(np, arg.function[1])(state.numpy_tensor, out=state.output)


def _common_arg(d):
    d.update(
        {
            "dim": (3,),
            "mag": (1, 3, 6, 7),
            "cont": (True, False),
            "trans": (False, True),
            "dtype": (torch.float, torch.double),
        }
    )
    return d


class NumpyUnaryComparison(Benchmark):

    args = utils.grid(
        _common_arg(
            {"function": ALL_UNARY_FUNCTIONS, "framework": ("Torch", "NumPy")}
        )
    )

    user_counters = {"sizes": 30 * " ", "strides": 30 * " "}

    def setupRun(self, state, arg):
        _setupRun(self, state, arg)

    def benchmark(self, state, arg):
        _benchmark(self, state, arg)


class CPUUnaryBench(Benchmark):
    args = utils.grid(
        _common_arg(
            {"function": TORCH_ONLY_FUNCTIONS, "framework": ("Torch",)}
        )
    )

    user_counters = {"sizes": 30 * " ", "strides": 30 * " "}

    def setupRun(self, state, arg):
        _setupRun(self, state, arg)

    def benchmark(self, state, arg):
        _benchmark(self, state, arg)
