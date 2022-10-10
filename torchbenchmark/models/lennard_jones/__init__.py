# This example was adapated from https://github.com/muhrin/milad
# It is licensed under the GLPv3 license. You can find a copy of it
# here: https://www.gnu.org/licenses/gpl-3.0.en.html .

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from functorch import vmap, jacrev
from typing import Tuple

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER


sigma = 0.5
epsilon = 4.


def lennard_jones(r):
    return epsilon * ((sigma / r)**12 - (sigma / r)**6)


def lennard_jones_force(r):
    """Get magnitude of LJ force"""
    return -epsilon * ((-12 * sigma**12 / r**13) + (6 * sigma**6 / r**7))


def make_prediction(model, drs):
    norms = torch.norm(drs, dim=1).reshape(-1, 1)
    energies = model(norms)

    network_derivs = vmap(jacrev(model))(norms).squeeze(-1)
    forces = -network_derivs * drs / norms
    return energies, forces


def loss_fn(energies, forces, predicted_energies, predicted_forces):
    return F.mse_loss(energies, predicted_energies) + 0.01 * F.mse_loss(forces, predicted_forces) / 3


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS
    DEFAULT_TRAIN_BSIZE = 1000
    DEFAULT_EVAL_BSIZE = 1000

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        self.model = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        self.model = self.model.to(device)

        r = torch.linspace(0.5, 2 * sigma, steps=self.batch_size)

        # Create a bunch of vectors that point along positive-x.
        # These are the dummy inputs to the model.
        self.drs = torch.outer(r, torch.tensor([1.0, 0, 0])).to(device=device)

        # Generate some dummy targets based off of some interpretation of the lennard_jones force.
        norms = torch.norm(self.drs, dim=1).reshape(-1, 1)
        self.norms = norms
        # Create training energies
        self.training_energies = torch.stack(list(map(lennard_jones, norms))).reshape(-1, 1)
        # Create forces with random direction vectors
        self.training_forces = torch.stack([
            force * dr for force, dr in zip(map(lennard_jones_force, norms), self.drs)
        ])

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def get_module(self):
        return self.model, (self.norms, )

    def train(self):
        model = self.model
        optimizer = self.optimizer
        model.train()
        optimizer.zero_grad()
        energies, forces = make_prediction(model, self.drs)
        loss = loss_fn(self.training_energies, self.training_forces, energies, forces)
        loss.backward()
        optimizer.step()

    def eval(self) -> Tuple[torch.Tensor]:
        model = self.model
        model.eval()
        with torch.no_grad():
            out = make_prediction(model, self.drs)
        return out
