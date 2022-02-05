import torch
from . import tke_pytorch
from torchbenchmark.tasks import OTHER
from ...util.model import BenchmarkModel


def _generate_inputs(size):
    import numpy as np
    import math

    np.random.seed(17)

    shape = (
        math.ceil(2 * size ** (1 / 3)),
        math.ceil(2 * size ** (1 / 3)),
        math.ceil(0.25 * size ** (1 / 3)),
    )

    # masks
    maskU, maskV, maskW = (
        (np.random.rand(*shape) < 0.8).astype("float64") for _ in range(3)
    )

    # 1d arrays
    dxt, dxu = (np.random.randn(shape[0]) for _ in range(2))
    dyt, dyu = (np.random.randn(shape[1]) for _ in range(2))
    dzt, dzw = (np.random.randn(shape[2]) for _ in range(2))
    cost, cosu = (np.random.randn(shape[1]) for _ in range(2))

    # 2d arrays
    kbot = np.random.randint(0, shape[2], size=shape[:2])
    forc_tke_surface = np.random.randn(*shape[:2])

    # 3d arrays
    kappaM, mxl, forc = (np.random.randn(*shape) for _ in range(3))

    # 4d arrays
    u, v, w, tke, dtke = (np.random.randn(*shape, 3) for _ in range(5))

    return (
        u,
        v,
        w,
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
        kbot,
        kappaM,
        mxl,
        forc,
        forc_tke_surface,
        tke,
        dtke,
    )

class TurbulentKineticEnergy(torch.nn.Module):
    def __init__(self, device):
        super(TurbulentKineticEnergy, self).__init__()
        self.device = device

    def forward(
        self,
        u,
        v,
        w,
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
        kbot,
        kappaM,
        mxl,
        forc,
        forc_tke_surface,
        tke,
        dtke,
    ):
        return tke_pytorch.integrate_tke(
            u,
            v,
            w,
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
            kbot,
            kappaM,
            mxl,
            forc,
            forc_tke_surface,
            tke,
            dtke,
        )


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS

    # Original input size: [2 ** i for i in range(12, 23, 2)]
    # Source: https://github.com/dionhaefner/pyhpc-benchmarks/blob/650ecc650e394df829944ffcf09e9d646ec69691/run.py#L25
    # Pick data-point when i = 20, size = 1048576
    def __init__(self, test, device, jit=False, eval_bs=1048576, extra_args=[]):
        super().__init__()
        self.device = device
        self.jit = jit
        self.test = test
        self.eval_bs = eval_bs
        self.extra_args = extra_args
        self.model = TurbulentKineticEnergy(self.device).to(device=self.device)
        input_size = eval_bs
        self.eval_example_inputs = tuple(
            torch.from_numpy(x).to(self.device) for x in _generate_inputs(input_size)
        )
        if self.jit:
            if hasattr(torch.jit, '_script_pdt'):
                self.model = torch.jit._script_pdt(self.model, example_inputs=[self.eval_example_inputs, ])
            else:
                self.model = torch.jit.script(self.model, example_inputs = [self.eval_example_inputs, ])

    def get_module(self):
        return self.model, self.eval_example_inputs

    def train(self, niter=1):
        raise NotImplementedError("Training not supported")

    def eval(self, niter=1):
        model, example_inputs = self.get_module()
        with torch.no_grad():
            for i in range(niter):
                model(*example_inputs)
