import torch
from . import eos_pytorch
from torchbenchmark.tasks import OTHER
from ...util.model import BenchmarkModel

def _generate_inputs(size):
    import math
    import numpy as np
    np.random.seed(17)

    shape = (
        math.ceil(2 * size ** (1/3)),
        math.ceil(2 * size ** (1/3)),
        math.ceil(0.25 * size ** (1/3)),
    )

    s = np.random.uniform(1e-2, 10, size=shape)
    t = np.random.uniform(-12, 20, size=shape)
    p = np.random.uniform(0, 1000, size=(1, 1, shape[-1]))
    return s, t, p

class EquationOfState(torch.nn.Module):
    def __init__(self):
        super(EquationOfState, self).__init__()

    def forward(self, s, t, p):
        return eos_pytorch.gsw_dHdT(s, t, p)

class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS
    # Original size: [2 ** i for i in range(12, 23, 2)
    # Source: https://github.com/dionhaefner/pyhpc-benchmarks/blob/650ecc650e394df829944ffcf09e9d646ec69691/run.py#L25
    # Pick data point: i = 20, size = 1048576
    DEFAULT_EVAL_BSIZE = 1048576

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        self.model = EquationOfState().to(device=self.device)
        input_size = self.batch_size
        raw_inputs = _generate_inputs(input_size)
        if hasattr(eos_pytorch, "prepare_inputs"):
            inputs = eos_pytorch.prepare_inputs(*raw_inputs, device=device)
        self.example_inputs = inputs

    def get_module(self):
        return self.model, self.example_inputs

    def train(self, niter=1):
        raise NotImplementedError("Training not supported")

    def eval(self, niter=1):
        model, example_inputs = self.get_module()
        with torch.no_grad():
            for i in range(niter):
                model(*example_inputs)
