import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models
from opacus import PrivacyEngine
from opacus.validators.module_validator import ModuleValidator
from typing import Tuple

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS
    DEFAULT_TRAIN_BSIZE = 64
    DEFAULT_EVAL_BSIZE = 64

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        self.model = models.resnet18(num_classes=10)
        self.model = ModuleValidator.fix(self.model)
        self.model = self.model.to(device)

        # Cifar10 images are 32x32 and have 10 classes
        self.example_inputs = (
            torch.randn((self.batch_size, 3, 32, 32), device=self.device),
        )
        self.example_target = torch.randint(0, 10, (self.batch_size,), device=self.device)
        dataset = data.TensorDataset(self.example_inputs[0], self.example_target)
        dummy_loader = data.DataLoader(dataset, batch_size=self.batch_size)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, _ = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=dummy_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
        )

    def get_module(self):
        if self.jit:
            raise NotImplementedError()

        return self.model, self.example_inputs

    def train(self, niter=1):
        if niter != 1:
            raise NotImplementedError("niter not implemented")
        if self.jit:
            raise NotImplementedError()
        model = self.model
        (images, ) = self.example_inputs
        model.train()
        targets = self.example_target

        output = model(images)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def eval(self, niter=1) -> Tuple[torch.Tensor]:
        if niter != 1:
            raise NotImplementedError("niter not implemented")
        if self.jit:
            raise NotImplementedError()
        model = self.model
        (images, ) = self.example_inputs
        model.eval()
        targets = self.example_target
        with torch.no_grad():
            out = model(images)
        return (out, )
