import torch
from . import isoneutral_pytorch
from torchbenchmark.tasks import OTHER
from ...util.model import BenchmarkModel

def _generate_inputs(size):
    import math
    import numpy as np

    np.random.seed(17)

    shape = (
        math.ceil(2 * size ** (1 / 3)),
        math.ceil(2 * size ** (1 / 3)),
        math.ceil(0.25 * size ** (1 / 3)),
    )

    # masks
    maskT, maskU, maskV, maskW = (
        (np.random.rand(*shape) < 0.8).astype("float64") for _ in range(4)
    )

    # 1d arrays
    dxt, dxu = (np.random.randn(shape[0]) for _ in range(2))
    dyt, dyu = (np.random.randn(shape[1]) for _ in range(2))
    dzt, dzw, zt = (np.random.randn(shape[2]) for _ in range(3))
    cost, cosu = (np.random.randn(shape[1]) for _ in range(2))

    # 3d arrays
    K_iso, K_iso_steep, K_11, K_22, K_33 = (np.random.randn(*shape) for _ in range(5))

    # 4d arrays
    salt, temp = (np.random.randn(*shape, 3) for _ in range(2))

    # 5d arrays
    Ai_ez, Ai_nz, Ai_bx, Ai_by = (np.zeros((*shape, 2, 2)) for _ in range(4))

    return (
        maskT,
        maskU,
        maskV,
        maskW,
        dxt,
        dxu,
        dyt,
        dyu,
        dzt,
        dzw,
        cost,
        cosu,
        salt,
        temp,
        zt,
        K_iso,
        K_11,
        K_22,
        K_33,
        Ai_ez,
        Ai_nz,
        Ai_bx,
        Ai_by,
    )


class IsoneutralMixing(torch.nn.Module):
    def __init__(self):
        super(IsoneutralMixing, self).__init__()

    def forward(
        self,
        maskT,
        maskU,
        maskV,
        maskW,
        dxt,
        dxu,
        dyt,
        dyu,
        dzt,
        dzw,
        cost,
        cosu,
        salt,
        temp,
        zt,
        K_iso,
        K_11,
        K_22,
        K_33,
        Ai_ez,
        Ai_nz,
        Ai_bx,
        Ai_by,
    ):
        return isoneutral_pytorch.isoneutral_diffusion_pre(
            maskT,
            maskU,
            maskV,
            maskW,
            dxt,
            dxu,
            dyt,
            dyu,
            dzt,
            dzw,
            cost,
            cosu,
            salt,
            temp,
            zt,
            K_iso,
            K_11,
            K_22,
            K_33,
            Ai_ez,
            Ai_nz,
            Ai_bx,
            Ai_by,
        )

class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS

    # Original input size: [2 ** i for i in range(12, 23, 2)]
    # Source: https://github.com/dionhaefner/pyhpc-benchmarks/blob/650ecc650e394df829944ffcf09e9d646ec69691/run.py#L25
    # Pick data-point when i = 20, size = 1048576
    DEFAULT_EVAL_BSIZE = 1048576

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        self.model = IsoneutralMixing().to(device=device)
        input_size = self.batch_size
        raw_inputs = _generate_inputs(input_size)
        if hasattr(isoneutral_pytorch, "prepare_inputs"):
            inputs = isoneutral_pytorch.prepare_inputs(*raw_inputs, device=device)
        self.example_inputs = inputs
        if self.jit:
            if hasattr(torch.jit, '_script_pdt'):
                self.model = torch.jit._script_pdt(self.model, example_inputs=[self.example_inputs, ])
            else:
                self.model = torch.jit.script(self.model, example_inputs = [self.example_inputs, ])

    def get_module(self):
        return self.model, self.example_inputs

    def train(self, niter=1):
        raise NotImplementedError("Training not supported")

    def eval(self, niter=1):
        model, example_inputs = self.get_module()
        with torch.no_grad():
            for i in range(niter):
                model(*example_inputs)
