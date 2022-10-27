import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import vmap, grad, combine_state_for_ensemble, make_functional_with_buffers
import functools

from .util import BenchmarkCase


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    @classmethod
    def make_input(cls, bs=None):
        shape = [64, 1, 28, 28]
        if bs is None:
            return torch.randn(*shape)
        return torch.randn(bs, *shape)

    @classmethod
    def make_target(cls, bs=None):
        shape = [64]
        if bs is None:
            return torch.randint(10, shape)
        return torch.randn(10, [bs] + shape)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        output = x
        return output

    @classmethod
    def make_input(cls, bs=None):
        shape = [64, 1, 28, 28]
        if bs is None:
            return torch.randn(*shape)
        return torch.randn(bs, *shape)

    @classmethod
    def make_target(cls, bs=None):
        shape = [64]
        if bs is None:
            return torch.randint(10, shape)
        return torch.randn(10, [bs] + shape)



class VmapWrapper(BenchmarkCase):
    def __init__(self, model_cls, device):
        self.name_ = f'{model_cls.__name__}_vmap_{device}'

        self.model = model_cls().to(device)
        self.inputs = model_cls.make_input().to(device)

    def name(self):
        return self.name_

    def run(self):
        vmap(self.model)(self.inputs)


def ensemble_setup(self, model_cls, device):
    num_models = 10
    models = [model_cls().to(device) for _ in range(num_models)]
    fmodel, params, buffers = combine_state_for_ensemble(models)
    self.fmodel = fmodel
    self.params = params
    self.buffers = buffers
    self.inputs = model_cls.make_input(num_models).to(device)


class EnsembleMultiWrapper(BenchmarkCase):
    def __init__(self, model_cls, device):
        self.name_ = f'{model_cls.__name__}_ensemble_multi_{device}'
        ensemble_setup(self, model_cls, device)

    def name(self):
        return self.name_

    def run(self):
        vmap(self.fmodel)(self.params, self.buffers, self.inputs)


class EnsembleSingleWrapper(BenchmarkCase):
    def __init__(self, model_cls, device):
        self.name_ = f'{model_cls.__name__}_ensemble_single_{device}'
        ensemble_setup(self, model_cls, device)
        self.inputs = self.inputs[0]

    def name(self):
        return self.name_

    def run(self):
        vmap(self.fmodel, (0, 0, None))(self.params, self.buffers, self.inputs)


def loss_fn(predictions, targets):
    return F.nll_loss(predictions, targets)


def compute_loss(fmodel, params, buffers, sample, target):
    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    target = target.unsqueeze(0)

    prediction = fmodel(params, buffers, sample)
    return loss_fn(prediction, target)


class PerSampleGradWrapper(BenchmarkCase):
    def __init__(self, model_cls, device):
        self.name_ = f'{model_cls.__name__}_persamplegrad_{device}'
        model = model_cls().to(device)
        self.model = make_functional_with_buffers(model)
        self.inputs = model_cls.make_input().to(device)
        self.targets = model_cls.make_target().to(device)

    def name(self):
        return self.name_

    def run(self):
        fmodel, params, buffers = self.model

        loss = functools.partial(compute_loss, fmodel)
        vmap(grad(loss), (None, None, 0, 0))(params, buffers, self.inputs, self.targets)
