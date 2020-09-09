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

import train_cyclegan
from train_cyclegan import prepare_training_loop
from test_cyclegan import get_model

def nyi():
    raise NotImplementedError()

class Model:
    def __init__(self, device='cpu', jit=False):
        if device != 'cuda': # NYI implemented for things that aren't on the GPU
            self.get_module = self.train = self.eval = nyi
            return

        self.device = device
        self.jit = jit
        train_args = f"--dataroot {os.path.dirname(train_cyclegan.__file__)}/datasets/horse2zebra --name horse2zebra --model cycle_gan --display_id 0 --n_epochs 3 --n_epochs_decay 3"
        self.training_loop = prepare_training_loop(train_args.split(' '))
        self.model, self.input = get_model(jit)

    def get_module(self):
        return self.model, self.input
        
    def train(self, niterations=None):
        # the training process is not patched to use scripted models
        if self.jit:
            raise NotImplementedError()
        return self.training_loop(niterations)

    
    def eval(self, niterations=1):
        model, example_inputs = self.get_module()
        for i in range(niterations):
            model(*example_inputs)


if __name__ == '__main__':
    m = Model(device='cuda', jit=False)
    model, example_inputs = m.get_module()
    model(*example_inputs)
    m.train()
    m.eval()

    m2 = Model(device='cuda', jit=True)
    m2.eval()

    m3 = Model()
    try:
        m3.train()
        finished = True
    except NotImplementedError:
        finished = False
    assert not finished