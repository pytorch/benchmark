"""
Maskrcnn model from torchvision
"""

import torch
import torch.optim as optim
import torchvision.models.detection as models
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(BenchmarkModel):
    task = COMPUTER_VISION.DETECTION

    def __init__(self, device=None, jit=False):
        self.device = device
        self.jit = jit
        self.model = models.maskrcnn_resnet50_fpn(pretrained=True)
        self.bs = 1
        self.C = 2
        self.H = 3
        self.W = 4
        self.example_inputs = self._gen_inputs()

    def _gen_inputs(self, bs, C, H, W):
        return torch.rand(bs, C, H, W, device=self.device)

    def train(self, niter=1):
        pass

    def eval(self, niter=1):
        predictions = self.model(self.infer_example_inputs)

if __name__ == "__main__":
    pass
