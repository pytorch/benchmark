from .dataloader import SuperSloMo
from .model_wrapper import Model as ModelWrapper
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import random
import numpy as np

from argparse import Namespace

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Model:
    def __init__(self, device=None, jit=False):
        self.device = device
        self.jit = jit
        self.module = ModelWrapper(device)
        if jit:
            self.module = torch.jit.script(self.module)

        self.args = args = Namespace(**{
            'dataset_root': 'dataset',
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
        self.example_inputs = trainFrameIndex, *trainData

    def get_module(self):
        return self.module, self.example_inputs

    def eval(self, niter=1):
        self.module.eval()
        for _ in range(niter):
            self.module(*self.example_inputs)

    def train(self, niter=1):
        self.module.train()
        for _ in range(niter):
            self.optimizer.zero_grad()
            
            Ft_p, loss = self.module(*self.example_inputs)
            
            loss.backward()
            self.optimizer.step()


if __name__ == '__main__':
    m = Model(device='cuda', jit=False)
    module, example_inputs = m.get_module()
    module(*example_inputs)
    m.train(niter=1)
    m.eval(niter=1)
