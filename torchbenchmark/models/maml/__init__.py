import numpy as np
import random
import time
import torch
from argparse import Namespace
from .meta import Meta
from pathlib import Path
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS

    def __init__(self, test="eval", device=None, jit=False):
        super().__init__()

        # load from disk or synthesize data
        use_data_file = False
        debug_print = False

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
            ('conv2d', [64, args.imgc, 3, 3, 2, 0]),
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

        if use_data_file:
            self.example_inputs = torch.load(f'{root}/batch.pt')
            self.example_inputs = tuple([torch.from_numpy(i).to(self.device) for i in self.example_inputs])
        else:
            # synthesize data parameterized by arg values
            self.example_inputs = (
                torch.randn(args.task_num, args.n_way, args.imgc, args.imgsz, args.imgsz).to(device),
                torch.randint(0, args.n_way, [args.task_num, args.n_way], dtype=torch.long).to(device),
                torch.randn(args.task_num, args.n_way * args.k_qry, args.imgc, args.imgsz, args.imgsz).to(device),
                torch.randint(0, args.n_way, [args.task_num, args.n_way * args.k_qry], dtype=torch.long).to(device))

        # print input shapes
        if debug_print:
            for i in range(len(self.example_inputs)):
                print(self.example_inputs[i].shape)

    def get_module(self):
        if self.jit:
            raise NotImplementedError()
        return self.module, self.example_inputs

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

    def eval_in_nograd(self):
        return False

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
