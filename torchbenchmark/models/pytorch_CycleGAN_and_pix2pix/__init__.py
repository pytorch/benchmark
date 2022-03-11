#!/usr/bin/env python
import torch
import os

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION
from typing import Tuple

from .train_cyclegan import prepare_training_loop
from .test_cyclegan import get_model

class Model(BenchmarkModel):
    task = COMPUTER_VISION.GENERATION
    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1
    ALLOW_CUSTOMIZE_BSIZE = False

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        device_arg = ""
        if self.device == "cpu":
            device_arg = "--gpu_ids -1"
        elif self.device == "cuda":
            device_arg = "--gpu_ids 0"
        if self.test == "train":
            train_args = f"--dataroot {os.path.dirname(__file__)}/datasets/horse2zebra --name horse2zebra --model cycle_gan --display_id 0 --n_epochs 3 --n_epochs_decay 3 {device_arg}"
            self.training_loop = prepare_training_loop(train_args.split(' '))
        args = f"--dataroot {os.path.dirname(__file__)}/datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout {device_arg}"
        self.model, self.input = get_model(args, self.device)

    def get_module(self):
        return self.model, self.input

    def set_train(self):
        # another model instance is used for training
        # and the train mode is on by default
        pass

    def train(self, niter=1):
        # the training process is not patched to use scripted models
        for i in range(niter):
            # training_loop has its own count logic inside.  It actually runs 7 epochs per niter=1 (with each 'epoch'
            # being limited to a small set of data)
            # it would be more in symmetry with the rest of torchbenchmark if niter=1 ran just an inner-loop
            # step rather than 7 epochs, but changing it now would potentially cause discontinuity with existing/historical measurement
            self.training_loop(None)

    def eval(self, niter=1) -> Tuple[torch.Tensor]:
        model, example_inputs = self.get_module()
        for i in range(niter):
            out = model(*example_inputs)
        return (out, )
