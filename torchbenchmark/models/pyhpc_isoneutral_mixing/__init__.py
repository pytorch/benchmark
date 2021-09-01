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
    def __init__(self, device):
        super(IsoneutralMixing, self).__init__()
        self.device = device

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
            self.device,
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

    def __init__(self, device=None, jit=False):
        super().__init__()
        self.device = device
        self.jit = jit
        self.model = IsoneutralMixing(self.device).to(device=self.device)
        self.example_inputs = tuple(
            torch.from_numpy(x).to(self.device) for x in _generate_inputs(2 ** 22)
        )
        if self.jit:
            self.model = torch.jit.script(self.model, example_inputs = [self.example_inputs, ])

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
