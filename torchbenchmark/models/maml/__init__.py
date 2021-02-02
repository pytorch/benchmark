import numpy as np
import random
import time
import torch
from argparse import Namespace
from .meta import Meta
from pathlib import Path

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Model:
    def __init__(self, device='cpu', jit=False):
        self.device = device
        self.jit = jit
        root = str(Path(__file__).parent)
        args = Namespace(**{
            'n_way': 5,
            'k_spt': 1,
            'k_qry': 15,
            'imgsz': 28,
            'imgc': 1,
            'task_num': 32,
            'meta_lr': 1e-3,
            'update_lr': 0.4,
            'update_step': 5,
            'update_step_test': 10
        })
        config = [
            ('conv2d', [64, 1, 3, 3, 2, 0]),
            ('relu', [True]),
            ('bn', [64]),
            ('conv2d', [64, 64, 3, 3, 2, 0]),
            ('relu', [True]),
            ('bn', [64]),
            ('conv2d', [64, 64, 3, 3, 2, 0]),
            ('relu', [True]),
            ('bn', [64]),
            ('conv2d', [64, 64, 2, 2, 1, 0]),
            ('relu', [True]),
            ('bn', [64]),
            ('flatten', []),
            ('linear', [args.n_way, 64])
        ]
        self.module = Meta(args, config).to(device)
        self.example_inputs = torch.load(f'{root}/batch.pt')
        self.example_inputs = tuple([torch.from_numpy(i).to(self.device) for i in self.example_inputs])

    def get_module(self):
        if self.jit:
            raise NotImplementedError()
        return self.module, self.example_inputs

    def set_eval(self):
        self.set_mode(False)

    def set_train(self):
        self.set_mode(True)

    def set_mode(self, train):
        (model, _) = self.get_module()
        model.train(train)

    def set_eval(self):
        self.set_mode(False)

    def set_train(self):
        self.set_mode(True)

    def set_mode(self, train):
        (model, _) = self.get_module()
        model.train(train)

    def eval(self, niter=1):
        if self.jit:
            raise NotImplementedError()
        for _ in range(niter):
            self.module(*self.example_inputs)

    def train(self, niter=1):
        if self.jit:
            raise NotImplementedError()
        for _ in range(niter):
            self.module(*self.example_inputs)


if __name__ == '__main__':
    m = Model(device='cpu', jit=False)
    module, example_inputs = m.get_module()
    module(*example_inputs)
    begin = time.time()
    m.train(niter=1)
    print(time.time() - begin)
    begin = time.time()
    m.eval(niter=1)
    print(time.time() - begin)
