import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

random.seed(1337)
torch.manual_seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from .pytorch_unet.unet import UNet
from .pytorch_unet.utils.dice_score import dice_loss

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION


class Model(BenchmarkModel):

    task = COMPUTER_VISION.SEGMENTATION

    def __init__(self, device=None, jit=False):
        super().__init__()
        self.device = device
        self.jit = jit

        self.args = self._get_args()
        # The sample inputs shape used here mimic the setting of the original repo
        # Source image link: https://www.kaggle.com/c/carvana-image-masking-challenge/code
        # Source images are 1280 x 1918, but the original code scales it in half to 640 x 959
        # The batch size is 1 and there are 3 channels for the image inputs and 1 for the mask
        self.sample_inputs = torch.rand((1, 3, 640, 959), dtype=torch.float32).to(self.device)
        self.sample_masks = torch.randint(0, 1, (1, 640, 959), dtype=torch.int64).to(self.device)
        self.model = UNet(n_channels=3, n_classes=2, bilinear=True).to(self.device)
        if self.jit:
            self.model = torch.jit.script(self.model)

    def get_module(self):
        return self.model, (self.sample_inputs,)

    def train(self, niter=1):
        optimizer = optim.RMSprop(self.model.parameters(), lr=self.args.lr, weight_decay=1e-8, momentum=0.9)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.args.amp)
        criterion = nn.CrossEntropyLoss()

        self.model.train()

        with torch.cuda.amp.autocast(enabled=self.args.amp):
            for _ in range(niter):
                masks_pred = self.model(self.sample_inputs)
                masks_true = self.sample_masks
                loss = criterion(masks_pred, masks_true) + \
                    dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(masks_true, self.model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

    def eval(self, niter=1):
        self.model.eval()
        with torch.no_grad():
            for _ in range(niter):
                mask_pred = self.model(self.sample_inputs)

                if self.model.n_classes == 1:
                    mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                else:
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), self.model.n_classes).permute(0, 3, 1, 2).float()

    def _get_args(self):
        parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
        parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                            help='Learning rate', dest='lr')
        parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
        return parser.parse_args([])

if __name__ == '__main__':
    m = Model(device='cuda', jit=False)
    module, sample_inputs = m.get_module()
    module(sample_inputs)
    m.train()
    m.eval()
