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
from ...util.model import BenchmarkModel, STEP_FN
from torchbenchmark.tasks import COMPUTER_VISION

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Model(BenchmarkModel):
    task = COMPUTER_VISION.OTHER_COMPUTER_VISION
    def __init__(self, device=None, jit=False):
        super().__init__()
        self.device = device
        self.jit = jit
        self.module = ModelWrapper(device)

        root = str(Path(__file__).parent)
        self.args = args = Namespace(**{
            'dataset_root': f'{root}/dataset',
            'train_batch_size': 6,
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
            batch_size=args.train_batch_size,
            shuffle=False)

        trainData, trainFrameIndex = next(iter(trainloader))
        frame0, frameT, frame1 = trainData
        trainData = (frame0.to(device),
                     frameT.to(device),
                     frame1.to(device))
        self.example_inputs = trainFrameIndex.to(self.device), *trainData

        if jit:
            self.module = torch.jit.script(self.module, example_inputs=[self.example_inputs, ])

    def get_module(self):
        return self.module, self.example_inputs

    def eval(self, niter=1, step_fn: STEP_FN = lambda: None):
        if self.device == 'cpu':
            raise NotImplementedError("Disabled due to excessively slow runtime - see GH Issue #100")

        for _ in range(niter):
            self.module(*self.example_inputs)
            step_fn()

    def train(self, niter=1, step_fn: STEP_FN = lambda: None):
        if self.device == 'cpu':
            raise NotImplementedError("Disabled due to excessively slow runtime - see GH Issue #100")

        for _ in range(niter):
            with self.annotate_forward():
                Ft_p, loss = self.module(*self.example_inputs)

            with self.annotate_backward():
                self.optimizer.zero_grad()
                loss.backward()

            with self.annotate_optimizer():
                self.optimizer.step()

            step_fn()


if __name__ == '__main__':
    m = Model(device='cuda', jit=False)
    module, example_inputs = m.get_module()
    module(*example_inputs)
    m.train(niter=1)
    m.eval(niter=1)
