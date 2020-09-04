# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import numpy as np
import random

import argparse
import os
import torch
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')



torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Model:
    def __init__(self, device='cpu', jit=False):
        self.device = device
        self.jit = jit

        # TODO - currently not supported
        if self.device == 'cpu':
            return
        assert cfg.MODEL.DEVICE == 'cuda'
        cfg.merge_from_file('configs/e2e_mask_rcnn_R_50_FPN_1x.yaml')
        cfg.merge_from_list(['SOLVER.IMS_PER_BATCH', '2', 
                             'SOLVER.BASE_LR', '0.0025',
                             'SOLVER.MAX_ITER', '720000', 
                             'SOLVER.STEPS', '(480000, 640000)', 
                             'TEST.IMS_PER_BATCH', '1', 
                             'MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN', '2000'])
        cfg.freeze()
        self.module = build_detection_model(cfg)
        start_iter = 0
        is_distributed = False

        # TODO haven't tried yet
        # if self.jit:
            # self.module = torch.jit.script(self.module)

        self.module.to(device)

        self.optimizer = make_optimizer(cfg, self.module)
        self.scheduler = make_lr_scheduler(cfg, self.optimizer)
        
        # not using 
        self.module, self.optimizer = amp.initialize(
            self.module, self.optimizer, opt_level='O0')

        self.train_loader = make_data_loader(
            cfg,
            is_train=True,
            is_distributed=is_distributed,
            start_iter=start_iter,
        )

        images, targets, _ = next(iter(self.train_loader))
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        self.train_inputs = (images, targets)
        
        # self.eval_loader = make_data_loader(
        #     cfg,
        #     is_train=False,
        #     is_distributed=is_distributed
        # )
        # # make_data_loader returns a list of loaders this time?
        # self.eval_loader = self.eval_loader[0]

        # images, _, _ = next(iter(self.eval_loader))
        # images = images.to(device)
        # self.example_inputs = (images,)

    def get_module(self):
        if self.jit:
            raise NotImplementedError("JIT not supported")
        raise NotImplementedError("eval not supported")
        # self.module.eval()
        # return self.module, self.example_inputs

    def eval(self, niter=1):
        if self.jit:
            raise NotImplementedError("JIT not supported")
        raise NotImplementedError("eval not supported")
        # self.module.eval()
        # for _ in range(niter):
            # self.module(*self.example_inputs)

    def train(self, niter=1):
        if self.jit:
            raise NotImplementedError("JIT not supported")
        if self.device == 'cpu':
            raise NotImplementedError("CPU not supported")
        self.module.train()
        for _ in range(niter):
            images, targets = self.train_inputs
            loss_dict = self.module(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            self.optimizer.zero_grad()

            # Note: If mixed precision is not used, this ends up doing nothing
            # Otherwise apply loss scaling for mixed-precision recipe
            # with amp.scale_loss(losses, self.optimizer) as scaled_losses:
                # scaled_losses.backward()
            losses.backward()

            self.optimizer.step()
            self.scheduler.step()


if __name__ == '__main__':
    m = Model(device='cuda', jit=False)
    # module, example_inputs = m.get_module()
    # module(*example_inputs)
    m.train(niter=1)
    # m.eval(niter=1)
