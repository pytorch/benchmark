#!/usr/bin/env python
import torch
import os
from pathlib import Path

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION
from typing import Tuple
from torchbenchmark import DATA_PATH

from .train_cyclegan import prepare_training_loop
from .test_cyclegan import get_model

def _create_data_dir(suffix):
    data_dir = Path(__file__).parent.joinpath(".data", suffix)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

class Model(BenchmarkModel):
    task = COMPUTER_VISION.GENERATION
    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1
    ALLOW_CUSTOMIZE_BSIZE = False

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        checkpoints_dir = _create_data_dir("checkpoints")
        results_dir = _create_data_dir("results")
        checkpoints_arg = f"--checkpoints_dir {checkpoints_dir}"
        results_arg = f"--results_dir {results_dir}"
        data_root = os.path.join(DATA_PATH, "pytorch_CycleGAN_and_pix2pix_inputs")
        device_arg = ""
        if self.device == "cpu":
            device_arg = "--gpu_ids -1"
        elif self.device == "cuda":
            device_arg = "--gpu_ids 0"
        if self.test == "train":
            train_args = f"--dataroot {data_root}/datasets/horse2zebra --name horse2zebra --model cycle_gan --display_id 0 --n_epochs 3 " + \
                         f"--n_epochs_decay 3 {device_arg} {checkpoints_arg}"
            print(train_args)
            self.training_loop = prepare_training_loop(train_args.split(' '))
        args = f"--dataroot {data_root}/datasets/horse2zebra/testA --name horse2zebra_pretrained --model test " + \
               f"--no_dropout {device_arg} {checkpoints_arg} {results_arg}"
        self.model, self.example_inputs = get_model(args, self.device)

    def get_module(self):
        return self.model, self.example_inputs

    def set_train(self):
        # another model instance is used for training
        # and the train mode is on by default
        pass

    def train(self):
        # the training process is not patched to use scripted models
        # training_loop has its own count logic inside.  It actually runs 7 epochs
        # (with each 'epoch' being limited to a small set of data)
        # it would be more in symmetry with the rest of torchbenchmark if it ran just one step
        # rather than 7 epochs, but changing it now would potentially cause
        # discontinuity with existing/historical measurement
        self.training_loop(None)

    def eval(self) -> Tuple[torch.Tensor]:
        model, example_inputs = self.get_module()
        out = model(*example_inputs)
        return (out, )
