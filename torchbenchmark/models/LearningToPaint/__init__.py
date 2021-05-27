import cv2
import torch
import numpy as np
import sys

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .baseline.utils.tensorboard import TensorBoard
from .baseline.Renderer.model import FCN
from .baseline.Renderer.stroke_gen import *
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import REINFORCEMENT_LEARNING

from argparse import Namespace

torch.manual_seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Model(BenchmarkModel):
    task = REINFORCEMENT_LEARNING.OTHER_RL
    def __init__(self, device=None, jit=False):
        super(Model, self).__init__()
        self.device = device
        self.jit = jit
        self.criterion = nn.MSELoss()
        net = FCN()
        self.step = 0
        self.opt = Namespace(**{
            'batch_size': 64,
            'debug': '',
            'script': False,
            })

        train_batch = []
        self.ground_truth = []
        for i in range(self.opt.batch_size):
            f = np.random.uniform(0, 1, 10)
            train_batch.append(f)
            self.ground_truth.append(draw(f))

        train_batch = torch.tensor(train_batch).float()
        self.ground_truth = torch.tensor(self.ground_truth).float()

        net = net.to(self.device)
        train_batch = train_batch.to(self.device)
        self.ground_truth = self.ground_truth.to(self.device)

        if self.jit:
            net = torch.jit._script_pdt(net, example_inputs = [(train_batch, ), ])

        self.module = net
        self.example_inputs = train_batch
        self.optimizer = optim.Adam(self.module.parameters(), lr=3e-6)

    def get_module(self):
        return self.module,[self.example_inputs]

    def train(self, niter=1):
        for _ in range(niter):
            gen = self.module(self.example_inputs)
            self.optimizer.zero_grad()
            loss = self.criterion(gen, self.ground_truth)
            loss.backward()
            self.optimizer.step()

    def eval(self, niter=1):
        for _ in range(niter):
            self.module(self.example_inputs)

if __name__ == '__main__':
    m = Model(device='cpu', jit=False)
    module,example_inputs = m.get_module()
    while m.step < 100:
        m.train(niter=1)
        if m.step%100 == 0:
            m.eval(niter=1)
        m.step += 1
