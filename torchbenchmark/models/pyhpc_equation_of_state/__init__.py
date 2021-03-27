import torch
from . import eos_pytorch
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
    def __init__(self, device=None, jit=False):
        super().__init__()
        self.device = device
        self.jit = jit
        self.model = EquationOfState().to(self.device)
        if self.jit:
            self.model = torch.jit.script(self.model)
        self.example_inputs = tuple(
            torch.from_numpy(x).to(self.device)
            for x in _generate_inputs(2 ** 22)
        )

    def get_module(self):
        return self.model, self.example_inputs

    def train(self, niter=1):
        raise NotImplementedError("Training not supported")

    def eval(self, niter=1):
        model, example_inputs = self.get_module()
        for i in range(niter):
            model(*example_inputs)

if __name__ == "__main__":
    m = Model(device="cuda", jit=True)
    module, example_inputs = m.get_module()
    module(*example_inputs)
    m.eval(niter=1)
