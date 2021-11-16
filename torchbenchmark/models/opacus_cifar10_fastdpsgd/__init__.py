import torch
import torch.optim as optim
import torch.nn as nn
from torchbenchmark.models.opacus_cifar10_2.cifar10model import CIFAR10Model
import torchvision.models as models

from opacus.utils.module_modification import convert_batchnorm_modules
from opacus import PrivacyEngine

from .cifar10 import load_cifar10

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER

def _preload():
    pass

class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS
    def __init__(self, device=None, jit=False, train_bs=64, eval_bs=64):
        super().__init__()
        self.device = device
        self.jit = jit

        self.model = models.resnet18(num_classes=10)
        self.model = convert_batchnorm_modules(self.model)
        self.model = self.model.to(device)

        kwargs = {
            'train_bs': train_bs,
            'eval_bs': eval_bs,
            'format': 'NCHW'
        }
        train_loader, test_loader, train_sample_size = load_cifar10(**kwargs)
        self.example_inputs, self.example_target = _preload(train_loader)
        self.infer_example_inputs = _preload(test_loader)
        self.cifar10_model = CIFAR10Model(batch_size=train_bs)
        self.optimizer  = optim.SGD(self.cifar10_model.parameters(), lr=learning_rate, momentum=0)

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

    def train(self):
        if self.jit:
            raise NotImplementedError("JIT is not implemented on this model")

        model, (images,) = self.get_module()
        model.train()
        targets = self.example_target

        output = model(images)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def eval(self):
        return NotImplementedError("Eval is not implemented")
