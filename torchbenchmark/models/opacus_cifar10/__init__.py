import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models
from opacus import PrivacyEngine
from opacus.validators.module_validator import ModuleValidator

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS

    def __init__(self, test, device, jit=False, train_bs=64, eval_bs=64, extra_args=[]):
        super().__init__()
        self.device = device
        self.jit = jit
        self.test = test
        self.extra_args = extra_args

        self.model = models.resnet18(num_classes=10)
        self.model = ModuleValidator.fix(self.model)
        self.model = self.model.to(device)

        # Cifar10 images are 32x32 and have 10 classes
        if test == "train":
            self.example_inputs = (
                torch.randn((train_bs, 3, 32, 32), device=self.device),
            )
            self.example_target = torch.randint(0, 10, (train_bs,), device=self.device)
            dataset = data.TensorDataset(self.example_inputs[0], self.example_target)
            dummy_loader = data.DataLoader(dataset, batch_size=train_bs)
        elif test == "eval":
            self.eval_example_inputs = (
                torch.randn((train_bs, 3, 32, 32), device=self.device),
            )

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

        return self.model, self.eval_example_inputs

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

    def eval(self, niter=1):
        if niter != 1:
            raise NotImplementedError("niter not implemented")
        if self.jit:
            raise NotImplementedError()
        model = self.model
        (images, ) = self.eval_example_inputs
        model.eval()
        targets = self.example_target
        with torch.no_grad():
            model(images)
