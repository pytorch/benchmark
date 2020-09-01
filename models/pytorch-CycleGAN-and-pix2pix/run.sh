#!/usr/bin/env python

# Make all randomness deterministic
import random
import argparse
import torch
import numpy as np

random.seed(1337)
torch.manual_seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from train import train
from test import run_example
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', help='eval something')
parser.add_argument('--debug', type=str, default=None, help='write reference output')
parser.add_argument('--script', action='store_true', help='run the model scripted?')

opts = parser.parse_args()
if opts.script:
    opts.eval = True

if opts.debug is not None or opts.eval:
    run_example(opts.debug, opts.script)
else:
    assert(not opts.script) # training isn't scripted
    train_args = "--dataroot ./datasets/horse2zebra --name horse2zebra --model cycle_gan --display_id 0 --n_epochs 3 --n_epochs_decay 3"
    train(train_args.split(' '))

