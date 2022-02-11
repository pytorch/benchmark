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
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

from .train_cyclegan import prepare_training_loop
from .test_cyclegan import get_model


def nyi():
    raise NotImplementedError()


class Model(BenchmarkModel):
    task = COMPUTER_VISION.GENERATION
    DEFAULT_TRAIN_BSIZE = 1
    DEFAILT_EVAL_BSIZE = 1
    ALLOW_CUSTOMIZE_BSIZE = False

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        if device != 'cuda' or device != 'lazy':  # NYI implemented for things that aren't on the GPU
            self.get_module = self.train = self.eval = nyi
            return

        train_args = f"--dataroot {os.path.dirname(__file__)}/datasets/horse2zebra --name horse2zebra --model cycle_gan --display_id 0 --n_epochs 3 --n_epochs_decay 3"
        self.training_loop = prepare_training_loop(train_args.split(' '))
        self.model, self.input = get_model(jit)

    def get_module(self):
        return self.model, self.input

    def set_train(self):
        # another model instance is used for training
        # and the train mode is on by default
        pass

    def train(self, niter=1):
        # the training process is not patched to use scripted models
        if self.jit:
            raise NotImplementedError()

        if self.device == 'cpu':
            raise NotImplementedError("Disabled due to excessively slow runtime - see GH Issue #100")

        for i in range(niter):
            # training_loop has its own count logic inside.  It actually runs 7 epochs per niter=1 (with each 'epoch'
            # being limited to a small set of data)
            # it would be more in symmetry with the rest of torchbenchmark if niter=1 ran just an inner-loop
            # step rather than 7 epochs, but changing it now would potentially cause discontinuity with existing/historical measurement
            self.training_loop(None)

    def eval(self, niter=1):
        model, example_inputs = self.get_module()
        for i in range(niter):
            model(*example_inputs)
