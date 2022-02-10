#!/usr/bin/env python
import os
import torch
import random
import numpy as np
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
    optimized_for_inference = True

    # Original train batch size: 16
    # Source: https://github.com/yunjey/stargan/blob/94dd002e93a2863d9b987a937b85925b80f7a19f/main.py#L73
    # This model doesn't support customizing eval batch size and will use the same bs as train
    DEFAULT_TRAIN_BSIZE = 16
    DEFAULT_EVAL_BSIZE = 16
    ALLOW_CUSTOMIZE_BSIZE = False

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        # init config
        config = parse_config()
        config.celeba_image_dir = os.path.join(os.path.dirname(__file__), 'data/celeba/images')
        config.attr_path = os.path.join(os.path.dirname(__file__), 'data/celeba/list_attr_celeba.txt')
        config.num_iters = 1
        config.batch_size = self.batch_size
        config.use_tensorboard = False
        config.device = device
        config.should_script = jit
        config.prefetch = True

        makedirs(config)
        self.data_loader = self.get_data_loader(config)
        if config.prefetch:
            self.data_loader = _prefetch(self.data_loader, size=config.num_iters, collate_fn=lambda item: tuple([m.to(self.device) for m in item]))
        self.solver = Solver(celeba_loader=self.data_loader,
                             rafd_loader=None,
                             config=config,
                             should_script=config.should_script)
        self.model = self.solver.G

        if self.jit and test == "eval":
            self.model = torch.jit.optimize_for_inference(self.model)

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
        self.model.train()

    def set_eval(self):
        self.model.eval()

    def train(self, niter=1):
        for _ in range(niter):
            self.solver.train()

    def eval(self, niter=1):
        model = self.model
        example_inputs = self.example_inputs
        for _ in range(niter):
            model(*example_inputs)
