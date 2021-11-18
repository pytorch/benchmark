import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models

from opacus.utils.module_modification import convert_batchnorm_modules
from opacus import PrivacyEngine

from .cifar10data import load_cifar10
from .cifar10model import CIFAR10Model

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER

def _prefetch(train_loader):
    inputs, targets = [], []
    for input, target in train_loader:
        inputs.append(input)
        targets.append(target)
    return inputs, targets

class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS
    def __init__(self, device=None, jit=False, train_bs=64):
        super().__init__()
        self.device = device
        self.jit = jit

        # set parameters
        learning_rate = 0.15
        noise_multiplier = 1.1
        l2_norm_clip = 1.0
        sigma = 1.0
        max_per_sample_grad_norm = 1.0
        base_model = "cifar10"

        # Build the model
        if base_model == "cifar10":
            self.model = CIFAR10Model(batch_size=train_bs)
        elif base_model == "resnet18":
            self.model = convert_batchnorm_modules(models.resnet18(num_classes=10))
        else:
            raise RuntimeError(f"only supported models are 'cifar10' or 'resnet18' got {base_model}")
        self.model = self.model.to(device)        

        # Build the input
        kwargs = {
            'train_bs': train_bs,
            'format': 'NCHW'
        }
        train_loader, train_sample_size = load_cifar10(**kwargs)
        self.example_inputs, self.example_target = _prefetch(train_loader)

        # Build optimizer and privacy engine
        self.privacy_engine = PrivacyEngine(
            self.model,
            batch_size=train_bs,
            sample_size=train_sample_size,
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=sigma,
            max_grad_norm=max_per_sample_grad_norm,
        )
        self.optimizer  = optim.SGD(self.model.parameters(),
                                    lr=learning_rate, momentum=0)
        self.privacy_engine.attach(self.optimizer)

    def get_module(self, niter=1):
        if self.jit:
            raise NotImplementedError("JIT is not implemented on this model")
        if not self.device == "cuda":
            raise NotImplementedError("CPU is not implemented on this model")
        for _, (x, y) in zip(niter, self.train_loader):
            return self.model, (x, )

    def train(self, niter=1):
        if self.jit:
            raise NotImplementedError("JIT is not implemented on this model")
        if not self.device == "cuda":
            raise NotImplementedError("CPU is not implemented on this model")
        for _, (x, y) in zip(niter, self.train_loader):
            self.model.zero_grad()
            outputs = self.model.forward(x)
            loss = nn.CrossEntropyLoss()(outputs, y)
            loss.backward()
            self.optimizer.step()

    def eval(self):
        return NotImplementedError("Eval is not implemented")
