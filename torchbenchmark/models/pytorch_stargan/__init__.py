#!/usr/bin/env python
import random
import argparse
import torch
import os
import numpy as np
from .solver import Solver
from .data_loader import get_loader
from .main import parse_config, makedirs
from ...util.model import BenchmarkModel, STEP_FN
from torchbenchmark.tasks import COMPUTER_VISION


# Make all randomness deterministic
random.seed(1337)
torch.manual_seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Model(BenchmarkModel):
    task = COMPUTER_VISION.GENERATION
    optimized_for_inference = True

    def __init__(self, device=None, jit=False):
        super().__init__()
        self.device = device
        self.jit = jit
        # init config
        config = parse_config()
        config.celeba_image_dir = os.path.join(os.path.dirname(__file__), 'data/celeba/images')
        config.attr_path = os.path.join(os.path.dirname(__file__), 'data/celeba/list_attr_celeba.txt')
        config.num_iters = 1
        config.batch_size = 24
        config.use_tensorboard = False
        config.device = device
        config.should_script = jit

        makedirs(config)

        self.data_loader = self.get_data_loader(config)
        self.solver = Solver(celeba_loader=self.data_loader,
                             rafd_loader=None,
                             config=config,
                             should_script=config.should_script)
        self.model = self.solver.G

        eval_solver = Solver(celeba_loader=self.data_loader,
                             rafd_loader=None,
                             config=config,
                             should_script=config.should_script)
        self.eval_model = eval_solver.G
        self.eval_model.eval()

        if self.jit:
            self.eval_model = torch.jit.optimize_for_inference(self.eval_model)

        self.example_inputs = self.generate_example_inputs()

    def get_data_loader(self, config):
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)
        return celeba_loader

    def generate_example_inputs(self):
        for x_real, c_trg_list in self.solver.get_test_inputs():
            return x_real, c_trg_list[0] # batch > #images

    def get_module(self):
        return self.model, self.example_inputs

    def set_train(self):
        # another model instance is used for training
        # and the train mode is on by default
        pass

    def set_eval(self):
        # eval_model is already set to `eval()`
        pass

    def train(self, niter=1, step_fn: STEP_FN = lambda: None):
        for _ in range(niter):
            self.solver.train()
            step_fn()

    def eval(self, niter=1, step_fn: STEP_FN = lambda: None):
        model = self.eval_model
        example_inputs = self.example_inputs
        for _ in range(niter):
            model(*example_inputs)
            step_fn()


if __name__ == '__main__':
    m = Model(device='cuda', jit=False)
    m.eval()
    model, example_inputs = m.get_module()
    model(*example_inputs)

    m.train(1)
    m.eval()

    m2 = Model(device='cpu', jit=True)
    m2.eval()
