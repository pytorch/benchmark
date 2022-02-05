from .dataloader import SuperSloMo
from .model_wrapper import Model as ModelWrapper
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import random
import numpy as np

from argparse import Namespace
from pathlib import Path
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _prefetch(data, device):
    result = []
    for item in data:
        result.append(item.to(device))
    return tuple(result)

class Model(BenchmarkModel):
    task = COMPUTER_VISION.OTHER_COMPUTER_VISION
    # Original code config:
    #    train batch size: 6
    #    eval batch size: 10
    #    hardware platform: Nvidia GTX 1080 Ti
    # Source: https://github.com/avinashpaliwal/Super-SloMo/blob/master/train.ipynb
    def __init__(self, test, device, jit=False, train_bs=6, eval_bs=10, extra_args=[]):
        super().__init__()
        self.device = device
        self.jit = jit
        self.train_bs = train_bs
        self.eval_bs = eval_bs
        self.test = test
        self.extra_args = extra_args
        self.module = ModelWrapper(device)

        root = str(Path(__file__).parent)
        self.args = args = Namespace(**{
            'dataset_root': f'{root}/dataset',
            'train_batch_size': train_bs,
            'eval_batch_size': eval_bs,
            'init_learning_rate': 0.0001,
        })

        self.optimizer = optim.Adam(self.module.parameters(),
                                    lr=args.init_learning_rate)

        mean = [0.429, 0.431, 0.397]
        std = [1, 1, 1]
        normalize = transforms.Normalize(mean=mean,
                                         std=std)
        transform = transforms.Compose([transforms.ToTensor(), normalize])

        trainset = SuperSloMo(root=args.dataset_root + '/train',
                                         transform=transform, train=True)
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.args.train_batch_size,
            shuffle=False)
        evalloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.args.eval_batch_size,
            shuffle=False)

        trainData, trainFrameIndex = next(iter(trainloader))
        trainData = _prefetch(trainData, self.device)
        self.example_inputs = trainFrameIndex.to(self.device), *trainData

        evalData, evalFrameIndex = next(iter(evalloader))
        evalData = _prefetch(evalData, self.device)
        self.infer_example_inputs = evalFrameIndex.to(self.device), *evalData

        if jit:
            if hasattr(torch.jit, '_script_pdt'):
                self.module = torch.jit._script_pdt(self.module, example_inputs=[self.example_inputs, ])
            else:
                self.module = torch.jit.script(self.module, example_inputs=[self.example_inputs, ])

    def get_module(self):
        return self.module, self.example_inputs

    def eval(self, niter=1):
        if self.device == 'cpu':
            raise NotImplementedError("Disabled due to excessively slow runtime - see GH Issue #100")

        for _ in range(niter):
            self.module(*self.infer_example_inputs)

    def train(self, niter=1):
        if self.device == 'cpu':
            raise NotImplementedError("Disabled due to excessively slow runtime - see GH Issue #100")

        for _ in range(niter):
            self.optimizer.zero_grad()

            Ft_p, loss = self.module(*self.example_inputs)

            loss.backward()
            self.optimizer.step()
