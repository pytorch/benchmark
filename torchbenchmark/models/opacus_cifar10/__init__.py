import os

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

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        # disable torchdynamo-fx2trt because it never terminates
        if "--torchdynamo" in extra_args and "fx2trt" in extra_args:
            raise NotImplementedError("TorchDynamo Fx2trt is not supported because of hanging issue. "
                                      "See: https://github.com/facebookresearch/torchdynamo/issues/109")
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)

        self.model = models.resnet18(num_classes=10)
        prev_wo_envvar = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        self.model = ModuleValidator.fix(self.model)
        if prev_wo_envvar is None:
            del os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = prev_wo_envvar
        self.model = self.model.to(device)

        # Cifar10 images are 32x32 and have 10 classes
        self.example_inputs = (
            torch.randn((self.batch_size, 3, 32, 32), device=self.device),
        )
        self.example_target = torch.randint(0, 10, (self.batch_size,), device=self.device)
        dataset = data.TensorDataset(self.example_inputs[0], self.example_target)
        self.dummy_loader = data.DataLoader(dataset, batch_size=self.batch_size)
        self.noise_multiplier: float=1.0
        self.max_grad_norm: float=1.0
        self.poisson_sampling: bool=False

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, _ = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dummy_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            poisson_sampling=self.poisson_sampling,
        )

    def get_module(self):
        return self.model, self.example_inputs

    def get_optimizer(self):
        return self.optimizer

    def set_optimizer(self, optimizer) -> None:
        self.optimizer = optimizer
        self.model, self.optimizer, _ = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dummy_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
        )

    def train(self):
        model = self.model
        (images, ) = self.example_inputs
        model.train()
        targets = self.example_target

        output = model(images)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def eval(self) -> Tuple[torch.Tensor]:
        model = self.model
        (images, ) = self.example_inputs
        model.eval()
        targets = self.example_target
        with torch.no_grad():
            out = model(images)
        return (out, )
