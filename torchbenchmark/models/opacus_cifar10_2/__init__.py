import torch
import torch.optim as optim
import torch.nn as nn
from torchbenchmark.models.opacus_cifar10.cifar10model import CIFAR10Model
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
    def __init__(self, device=None, jit=False, cifar10=False, train_bs=64, eval_bs=64):
        super().__init__()
        self.device = device
        self.jit = jit

        self.model = models.resnet18(num_classes=10)
        self.model = convert_batchnorm_modules(self.model)
        self.model = self.model.to(device)

        if cifar10:
            kwargs = {
                'train_bs': train_bs,
                'eval_bs': eval_bs,
                'format': 'NCHW'
            }
            learning_rate = 
            train_loader, test_loader, train_sample_size = load_cifar10(**kwargs)
            self.example_inputs, self.example_target = _preload(train_loader)
            self.infer_example_inputs = _preload(test_loader)
            self.cifar10_model = CIFAR10Model(batch_size=train_bs)
            self.optimizer  = optim.SGD(self.cifar10_model.parameters(), lr=learning_rate, momentum=0)
        else:
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

    def train(self):
        if self.jit:
            raise NotImplementedError()

        model, (images,) = self.get_module()
        model.train()
        targets = self.example_target

        output = model(images)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def eval(self):
        if self.jit:
            raise NotImplementedError()

        model, (images,) = self.get_module()
        model.eval()
        targets = self.example_target
        with torch.no_grad():
            model(images)


if __name__ == "__main__":
    m = Model(device="cuda", jit=False)
    module, example_inputs = m.get_module()
    module(*example_inputs)
    m.train()
    m.eval()
