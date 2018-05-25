from __future__ import print_function
import torch
import time
import gc
import argparse
import math
import numpy as np
from torch import functional as F
import sys
PY2 = sys.version_info[0] == 2
timer = time.time if PY2 else time.perf_counter



def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

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

def start_stats(common_name, framework_name, fname, mag, count, tv):
    status = ""
    status += "tag: {:<15}".format(common_name)
    status += "fname: {:<15}".format(framework_name)
    status += "{:<15}".format(fname)
    status += " memory: {:<10}".format("O(10^" + str(mag) + ")KB")
    status += " count: {:<6}".format(count)
    status += " size: {:<20}".format(list(tv.size()))
    status += " stride: {:<60}".format(list(map(lambda x: "{:>7}".format(x), list(tv.stride()))))
    status += " numel: {:<9}".format(tv.numel())
    return status

def finish_stats(dtype, dim, elapsed):
    status = ""
    status += " type: {:<18}".format(dtype)
    status += " dim: {:<5}".format(dim)
    status += " elapsed: {:8.4f}".format(elapsed)
    return status

def lambda_benchmark(common_name, types, fun, name, framework_name, cast):
    goal_size = 1000
    onek = 1000
    goal = onek * 1000 * goal_size
    for dtype in types:
        for cont in [True, False]:
            for trans in [True, False]:
                for mag in [4, 5]:
                    for dim in [4]:
                        size_ = int(onek * math.pow(10, mag))
                        count = goal / size_
                        tv = make_tensor(size_, dtype, cont, 3, trans)
                        status = start_stats(common_name, framework_name, name, mag, count, tv)
                        gc.collect()
                        gc.collect()
                        fun(tv)
                        gc.collect()
                        gc.collect()
                        tstart = timer()
                        for _ in range(count):
                            fun(tv)
                        elapsed = timer() - tstart
                        print(status + finish_stats(dtype, dim, elapsed))
                        gc.collect()
                        gc.collect()

def numpy_comparison():
    all_fns = [
        "cos",
        "sin",
        ("acos", "arccos"),
        ("asin", "arcsin"),
        ("atan", "arctan"),
        "exp",
        "log",
        "expm1",
        "cosh",
        "tan",
        "sinh",
        "tanh",
        "abs",
        "ceil",
        "floor",
        "round",
        "sqrt",
        "trunc",
        "erf",
        "log10",
        "log1p",
        "log2",
        "rsqrt",
    ]
    for fn in all_fns:
        if isinstance(fn, tuple):
            torch_fn = fn[0]
            numpy_fn = fn[1]
        else:
            torch_fn = fn
            numpy_fn = fn

	try:
            lambda_benchmark(torch_fn, [torch.float], lambda x: getattr(x, torch_fn)(), torch_fn, "torch", lambda x: x)
        except AttributeError:
            eprint(torch_fn + " not supported by torch.")

    for fn in all_fns:
        if isinstance(fn, tuple):
            torch_fn = fn[0]
            numpy_fn = fn[1]
        else:
            torch_fn = fn
            numpy_fn = fn
        try:
            lambda_benchmark(torch_fn, float_types, lambda x: getattr(np, numpy_fn)(x), numpy_fn, "numpy", lambda x: x.numpy())
        except AttributeError:
            eprint(numpy_fn + " not supported by numpy.")


# TODO: Output csv for further processing - Table to show results directly on github
# TODO: Output time per op
# TODO: Output amount of data in machine readable form
# TODO: Shuffle operations to remove more bias
# TODO: Write general setup script to check for environment setup

if __name__ == "__main__":
    numpy_comparison()
