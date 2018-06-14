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
    ("exp", "exp"),
    ("log", "log"),
    ("cos", "cos"),
    ("sin", "sin"),
]
ALL_UNUSED_UNARY_FUNCTIONS = [
    ("abs", "abs"),
    ("acos", "arccos"),
    ("asin", "arcsin"),
    ("atan", "arctan"),
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
    ("rsqrt", "rsqrt"),
    ("trunc", "trunc"),
    #    ("erf", "erf"), Needs Intel Numpy (conda)
    ("log10", "log10"),
    ("log1p", "log1p"),
    ("log2", "log2"),
]


class NumpyComparison(GridBenchmark):

    args = {
        "dim": (3,),
        "mag": (1, 3, 6, 7),
        "cont": (False, True),
        "trans": (False, True),
        "dtype": (torch.float,),
        "function": ALL_UNARY_FUNCTIONS,
        "framework": ("Torch", "NumPy"),
    }

    def setupRun(self, state, arg):
        size_ = int(math.pow(10, arg.mag))
        tv = make_tensor(size_, arg.dtype, arg.cont, arg.dim, arg.trans)
        state.torch_tensor = tv
        state.output = tv.clone()
        if arg.framework == "NumPy":
            if arg.dtype == torch.float:
                state.numpy_tensor = state.torch_tensor.numpy()
                assert state.numpy_tensor.dtype == np.float32
            if arg.dtype == torch.double:
                state.numpy_tensor = state.torch_tensor.numpy()
                assert state.numpy_tensor.dtype == np.float64
            state.output = state.numpy_tensor.copy()

    def benchmark(self, state, arg):
        if arg.framework == "Torch":
            getattr(torch, arg.function[0])(
                state.torch_tensor, out=state.output
            )
        else:
            getattr(np, arg.function[1])(state.numpy_tensor, out=state.output)
