#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from argparse import Namespace
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from .moco.builder import MoCo
from .main_moco import adjust_learning_rate
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER

torch.manual_seed(1058467)
random.seed(1058467)
cudnn.deterministic = True


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS
    def __init__(self, device=None, jit=False):
        super().__init__()
        """ Required """
        self.device = device
        self.jit = jit
        self.opt = Namespace(**{
            'arch': 'resnet50',
            'batch_size': 32,
            'epochs': 2,
            'start_epoch': 0,
            'lr': 0.03,
            'schedule': [120, 160],
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'gpu': None,
            'moco_dim': 128,
            'moco_k': 32000,
            'moco_m': 0.999,
            'moco_t': 0.07,
            'mlp': False,
            'aug_plus': False,
            'cos': False,
            'fake_data': True,
            'distributed': True,
        })

        if device != "cuda":
            return

        try:
            dist.init_process_group(backend='nccl', init_method='tcp://localhost:10001',
                                    world_size=1, rank=0)
        except RuntimeError:
            pass # already initialized?


        self.model = MoCo(
            models.__dict__[self.opt.arch],
            self.opt.moco_dim, self.opt.moco_k, self.opt.moco_m, self.opt.moco_t, self.opt.mlp)

        self.model.cuda(0)
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[0])

        # if self.jit:
        #     self.model = torch.jit.script(self.model)

        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda(0)

        self.optimizer = torch.optim.SGD(self.model.parameters(), self.opt.lr,
                                         momentum=self.opt.momentum,
                                         weight_decay=self.opt.weight_decay)

        batches = []

        for i in range(4):
          batches.append(torch.randn(self.opt.batch_size, 3, 224, 224))

        def collate_fn(data):
            ind = data[0]
            return [batches[2*ind], batches[2*ind+1]], 0

        self.train_loader = torch.utils.data.DataLoader(
            range(2), collate_fn=collate_fn)

        for i, (images, _) in enumerate(self.train_loader):
            images[0] = images[0].cuda(device=0, non_blocking=True)
            images[1] = images[1].cuda(device=0, non_blocking=True)

    def get_module(self):
        """ Recommended
        Returns model, example_inputs
        model should be torchscript model if self.jit is True.
        Both model and example_inputs should be on self.device properly.
        `model(*example_inputs)` should execute one step of model forward.
        """
        if self.device != "cuda":
            raise NotImplementedError("GPU only")

        # figure out why self.jit was commented out
        if self.jit:
            raise NotImplementedError("eager only")

        images = []
        for (i, _) in self.train_loader:
            images = (i[0], i[1])
        return (self.model, images)

    def train(self, niterations=1):
        """ Recommended
        Runs training on model for `niterations` times. When `niterations` is left
        to its default value, it should run for at most two minutes, and be representative
        of the performance of a traditional training loop. One iteration should be sufficient
        to warm up the model for the purpose of profiling.

        Avoid unnecessary benchmark noise by keeping any tensor creation, memcopy operations in __init__.

        Leave warmup to the caller (e.g. don't do it inside)
        """
        if self.device != "cuda":
            raise NotImplementedError("GPU only")

        self.model.train()
        for e in range(niterations):
            adjust_learning_rate(self.optimizer, e, self.opt)
            for i, (images, _) in enumerate(self.train_loader):
                # compute output
                output, target = self.model(im_q=images[0], im_k=images[1])
                loss = self.criterion(output, target)

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def eval(self, niterations=1):
        """ Recommended
        Run evaluation on model for `niterations` inputs. One iteration should be sufficient
        to warm up the model for the purpose of profiling.
        In most cases this can use the `get_module` API but in some cases libraries
        do not have a single Module object used for inference. In these case, you can
        write a custom eval function.

        Avoid unnecessary benchmark noise by keeping any tensor creation, memcopy operations in __init__.

        Leave warmup to the caller (e.g. don't do it inside)
        """
        if self.device != "cuda":
            raise NotImplementedError("GPU only")

        for i in range(niterations):
            for i, (images, _) in enumerate(self.train_loader):
                self.model(im_q=images[0], im_k=images[1])


if __name__ == '__main__':

    m = Model(device='cuda', jit=False)
    module, example_inputs = m.get_module()
    module(*example_inputs)
    m.train(2)
    m.eval()
