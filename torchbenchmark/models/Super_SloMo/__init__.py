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
    DEFAULT_TRAIN_BSIZE = 6
    DEFAULT_EVAL_BSIZE = 10

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        self.model = ModelWrapper(device)
        root = str(Path(__file__).parent)
        self.args = args = Namespace(**{
            'dataset_root': f'{root}/dataset',
            'batch_size': self.batch_size,
            'init_learning_rate': 0.0001,
        })

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=args.init_learning_rate)

        mean = [0.429, 0.431, 0.397]
        std = [1, 1, 1]
        normalize = transforms.Normalize(mean=mean,
                                         std=std)
        transform = transforms.Compose([transforms.ToTensor(), normalize])

        trainset = SuperSloMo(root=args.dataset_root + '/train',
                                         transform=transform, train=True)
        loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            shuffle=False)

        data, frameIndex = next(iter(loader))
        data = _prefetch(data, self.device)
        self.example_inputs = frameIndex.to(self.device), *data

        if jit:
            if hasattr(torch.jit, '_script_pdt'):
                self.model = torch.jit._script_pdt(self.model, example_inputs=[self.example_inputs, ])
            else:
                self.model = torch.jit.script(self.model, example_inputs=[self.example_inputs, ])

    def get_module(self):
        return self.model, self.example_inputs

    def eval(self, niter=1):
        if self.device == 'cpu':
            raise NotImplementedError("Disabled due to excessively slow runtime - see GH Issue #100")

        for _ in range(niter):
            self.model(*self.example_inputs)

    def train(self, niter=1):
        if self.device == 'cpu':
            raise NotImplementedError("Disabled due to excessively slow runtime - see GH Issue #100")

        for _ in range(niter):
            self.optimizer.zero_grad()

            Ft_p, loss = self.model(*self.example_inputs)

            loss.backward()
            self.optimizer.step()
