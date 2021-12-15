#!/usr/bin/env python
import random
import argparse
import torch
import os
import numpy as np
from torch.utils import data
from .solver import Solver
from .data_loader import get_loader
from .main import parse_config, makedirs
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION


# Make all randomness deterministic
random.seed(1337)
torch.manual_seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

def _prefetch(loader, size, collate_fn):
    result = []
    for _, item in zip(range(size), loader):
        result.append(collate_fn(item))
    return result

class Model(BenchmarkModel):
    task = COMPUTER_VISION.GENERATION

    # Original train batch size: 16
    # Source: https://github.com/yunjey/stargan/blob/94dd002e93a2863d9b987a937b85925b80f7a19f/main.py#L73
    # This model doesn't support setting eval batch size and will use the same bs as train
    def __init__(self, device=None, jit=False, train_bs=16, eval_bs=16, prefetch=True):
        super().__init__()
        self.device = device
        self.jit = jit
        # init config
        config = parse_config()
        config.celeba_image_dir = os.path.join(os.path.dirname(__file__), 'data/celeba/images')
        config.attr_path = os.path.join(os.path.dirname(__file__), 'data/celeba/list_attr_celeba.txt')
        config.num_iters = 1
        config.batch_size = train_bs
        config.use_tensorboard = False
        config.device = device
        config.should_script = jit

        makedirs(config)

        self.data_loader = self.get_data_loader(config)
        eval_config = config
        eval_config.batch_size =eval_bs
        self.eval_data_loader = self.get_data_loader(eval_config)
        if prefetch:
            self.data_loader = _prefetch(self.data_loader, size=config.num_iters, collate_fn=lambda item: tuple([m.to(self.device) for m in item]))
            self.eval_data_loader = _prefetch(self.eval_data_loader, size=config.num_iters, collate_fn=lambda item: tuple([m.to(self.device) for m in item]))
        self.solver = Solver(celeba_loader=self.data_loader,
                             rafd_loader=None,
                             config=config,
                             should_script=config.should_script)
        self.model = self.solver.G

        self.eval_solver = Solver(celeba_loader=self.eval_data_loader,
                             rafd_loader=None,
                             config=eval_config,
                             should_script=eval_config.should_script)
        self.eval_model = self.eval_solver.G
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
        for x_real, c_trg_list in self.eval_solver.get_test_inputs():
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

    def train(self, niterations=1):
        for _ in range(niterations):
            self.solver.train()

    def eval(self, niterations=1):
        model = self.eval_model
        example_inputs = self.example_inputs
        for _ in range(niterations):
            model(*example_inputs)


if __name__ == '__main__':
    m = Model(device='cuda', jit=False)
    m.eval()
    model, example_inputs = m.get_module()
    model(*example_inputs)

    m.train(1)
    m.eval()

    m2 = Model(device='cpu', jit=True)
    m2.eval()
