import torch
import torch.nn as nn
from functorch import vmap, jacfwd, jacrev
from .util import BenchmarkCase

# batched hessians of fully connected layers is a popular quantity
# in physics-related models.
# This test case is from https://github.com/pytorch/functorch/issues/989
# We haven't been able to get the full model yet, so, this test case
# is going into the functorch userbenchmark instead of torchbenchmark.
class VmapHessianFC(BenchmarkCase):
    def __init__(self):
        device = 'cuda'
        D1 = 2  # x, y
        D2 = 3  # u, v, p
        B = 10000
        x = torch.randn(B, D1).to(device)

        model = nn.Sequential(
            nn.Linear(D1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, D2),
        ).to(device)

        self.model = model
        self.x = x

    def name(self):
        return 'vmap_hessian_fc_cuda'

    def run(self):
        def predict(x):
            out = self.model(x)
            return out, out

        hessian, pred = vmap(
            jacfwd(jacrev(predict, argnums=0, has_aux=True), argnums=0, has_aux=True),
            in_dims=0,
        )(
            self.x
        )
