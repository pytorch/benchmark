import torch

from framework import Benchmark


def _make_args():
    layers = ["Linear", "LogSoftmax"]
    layer_init_args = [(2000, 1000), (1,)]
    layer_input_sizes = [(1024, 2000), (1024, 2000)]
    args = []
    for i in range(len(layers)):
        arg = {}
        arg["layer"] = layers[i]
        arg["init_args"] = layer_init_args[i]
        arg["input_size"] = layer_input_sizes[i]
        arg["train"] = False
        args.append(arg)
    layers = ["Tanh", "Sigmoid", "ReLU"]
    for i in range(len(layers)):
        arg = {}
        arg["layer"] = layers[i]
        arg["init_args"] = ()
        arg["input_size"] = (1000, 1000)
        arg["train"] = False
        args.append(arg)
    return args


class CPUNNBench(Benchmark):
    args = _make_args()

    def setupRun(self, state, arg):
        state.layer_obj = getattr(torch.nn, arg.layer)(*arg.init_args)
        state.input = torch.randn(*arg.input_size).type("torch.FloatTensor")

    def benchmark(self, state, arg):
        state.output = state.layer_obj(state.input)
