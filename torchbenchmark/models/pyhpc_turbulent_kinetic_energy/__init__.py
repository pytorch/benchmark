import torch
from . import tke_pytorch
from typing import Tuple
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
        # tke and dtke will be modified in integrate_tke and generate inconsistent results
        # so clone them before passing them in
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
            torch.clone(tke),
            torch.clone(dtke),
        )


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS

    # Original input size: [2 ** i for i in range(12, 23, 2)]
    # Source: https://github.com/dionhaefner/pyhpc-benchmarks/blob/650ecc650e394df829944ffcf09e9d646ec69691/run.py#L25
    # Pick data-point when i = 20, size = 1048576
    DEFAULT_EVAL_BSIZE = 1048576
    ALLOW_CUSTOMIZE_BSIZE = False
    CANNOT_SET_CUSTOM_OPTIMIZER = True

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)

        self.model = TurbulentKineticEnergy(self.device).to(device=self.device)
        input_size = self.batch_size
        self.example_inputs = tuple(
            torch.from_numpy(x).to(self.device) for x in _generate_inputs(input_size)
        )

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        raise NotImplementedError("Training not supported")

    def eval(self) -> Tuple[torch.Tensor]:
        model, example_inputs = self.get_module()
        out = model(*example_inputs)
        return out
