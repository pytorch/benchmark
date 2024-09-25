#!/usr/bin/env python
import os
from typing import Tuple

import torch
from torchbenchmark import DATA_PATH
from torchbenchmark.tasks import COMPUTER_VISION

from ...util.model import BenchmarkModel
from .data_loader import get_loader
from .main import makedirs, parse_config
from .solver import Solver

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
    # This model doesn't support customizing eval batch size and will use the same bs as train
    DEFAULT_TRAIN_BSIZE = 16
    DEFAULT_EVAL_BSIZE = 16
    ALLOW_CUSTOMIZE_BSIZE = False

    # TODO: Customizing the optimizer is nontrivial, perhaps a next step.
    CANNOT_SET_CUSTOM_OPTIMIZER = True

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )

        # init config
        config = parse_config()
        config.celeba_image_dir = os.path.join(
            DATA_PATH, "pytorch_stargan_inputs/data/celeba/images"
        )
        config.attr_path = os.path.join(
            DATA_PATH, "pytorch_stargan_inputs/data/celeba/list_attr_celeba.txt"
        )
        config.num_iters = 1
        config.num_workers = 0
        config.batch_size = self.batch_size
        config.use_tensorboard = False
        config.device = device
        config.should_script = False
        config.prefetch = True

        makedirs(config)
        self.data_loader = self.get_data_loader(config)
        if config.prefetch:
            self.data_loader = _prefetch(
                self.data_loader,
                size=config.num_iters,
                collate_fn=lambda item: tuple([m.to(self.device) for m in item]),
            )
        self.solver = Solver(
            celeba_loader=self.data_loader,
            rafd_loader=None,
            config=config,
            should_script=config.should_script,
        )
        self.model = self.solver.G
        if self.test == "train":
            self.model.train()
        elif self.test == "eval":
            self.model.eval()

        self.example_inputs = self.generate_example_inputs()

    def get_data_loader(self, config):
        celeba_loader = get_loader(
            config.celeba_image_dir,
            config.attr_path,
            config.selected_attrs,
            config.celeba_crop_size,
            config.image_size,
            config.batch_size,
            "CelebA",
            config.mode,
            config.num_workers,
        )
        return celeba_loader

    def generate_example_inputs(self):
        for x_real, c_trg_list in self.solver.get_test_inputs():
            return x_real, c_trg_list[0]  # batch > #images

    def jit_callback(self):
        self.solver.G = torch.jit.script(self.solver.G)
        self.solver.D = torch.jit.script(self.solver.D)

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        self.solver.train()

    def eval(self) -> Tuple[torch.Tensor]:
        model = self.model
        example_inputs = self.example_inputs
        out = model(*example_inputs)
        return (out,)
