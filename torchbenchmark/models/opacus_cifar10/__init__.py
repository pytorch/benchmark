import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from opacus.utils.module_modification import convert_batchnorm_modules
from opacus import PrivacyEngine

from ...util.model import BenchmarkModel, STEP_FN
from torchbenchmark.tasks import OTHER


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS
    def __init__(self, device=None, jit=False):
        super().__init__()
        self.device = device
        self.jit = jit

        self.model = models.resnet18(num_classes=10)
        self.model = convert_batchnorm_modules(self.model)
        self.model = self.model.to(device)

        self.example_inputs = (
            torch.randn((64, 3, 32, 32), device=self.device),
        )
        self.example_target = torch.randint(0, 10, (64,), device=self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        # This is supposed to equal the number of data points.
        # It is only to compute stats so dwai about the value.
        sample_size = 64 * 100
        clipping = {"clip_per_layer": False, "enable_stat": False}
        self.privacy_engine = PrivacyEngine(
            self.model,
            batch_size=64,
            sample_size=sample_size,
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            secure_rng=False,
            **clipping,
        )
        self.privacy_engine.attach(self.optimizer)

    def get_module(self):
        if self.jit:
            raise NotImplementedError()

        return self.model, self.example_inputs

    def train(self, niter=1, step_fn: STEP_FN = lambda: None):
        if self.jit:
            raise NotImplementedError()

        model, (images,) = self.get_module()
        model.train()
        targets = self.example_target

        for _ in range(niter):
            with self.annotate_forward():
                output = model(images)
                loss = self.criterion(output, targets)

            with self.annotate_backward():
                self.optimizer.zero_grad()
                loss.backward()

            with self.annotate_optimizer():
                self.optimizer.step()

            step_fn()

    def eval(self, niter=1, step_fn: STEP_FN = lambda: None):
        if self.jit:
            raise NotImplementedError()

        model, (images,) = self.get_module()
        model.eval()
        with torch.no_grad():
            for _ in range(niter):
                model(images)
                step_fn()


if __name__ == "__main__":
    m = Model(device="cuda", jit=False)
    module, example_inputs = m.get_module()
    module(*example_inputs)
    m.train()
    m.eval()
