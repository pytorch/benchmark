#!/usr/bin/env python

# Make all randomness deterministic
import random
import argparse
import torch
import os
import numpy as np

random.seed(1337)
torch.manual_seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from .train import train_main
from .model import SSD300
from .utils import label_map
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION


class Model(BenchmarkModel):
    task = COMPUTER_VISION.DETECTION

    def __init__(self, device=None, jit=False):
        super().__init__()
        self.device = device
        self.jit = jit

    def get_module(self):
        if self.jit:
            raise NotImplementedError()

        model = SSD300(n_classes=len(label_map))
        model.to(self.device).eval()
        input = torch.rand(1, 3, 300, 300).to(self.device)
        return model, input

    def set_train(self):
        # another model instance is used for training
        # and the train mode is on by default
        pass

    def train(self, niterations=1):
        # the training process is not patched to use scripted models
        if self.jit:
            raise NotImplementedError()

        train_main(self.device, self.jit)

    def eval(self, niterations=1):
        model, input = self.get_module()
        model(input)


if __name__ == '__main__':
    m = Model(device='cuda', jit=False)
    model, example_inputs = m.get_module()
    model(*example_inputs)
    m.train()
    m.eval()
