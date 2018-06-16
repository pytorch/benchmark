import torch
import math
import numpy as np

from framework import Benchmark
from framework import utils


def _make_args():
    layers = ["Linear"]
    layer_init_args = [(2000, 1000)]
    layer_input_sizes = [(1024, 2000)]
    args = []
    for i in range(len(layers)):
        arg = {}
        arg["layer"] = layers[i]
        arg["init_args"] = layer_init_args[i]
        arg["input_size"] = layer_input_sizes[i]
        args.append(arg)
    return args


class CPUNNBench(Benchmark):
    args = _make_args()

    def setupRun(self, state, arg):
        state.layer_obj = getattr(torch.nn, arg.layer)(*arg.init_args)
        state.input = torch.randn(*arg.input_size).type("torch.FloatTensor")

    def benchmark(self, state, arg):
        state.output = self.layer_job(state.input)
