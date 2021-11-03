"""
maskrcnn model implementation, rewritten from MLPerf v1.0
"""

import torch
import torch.optim as optim
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(BenchmarkModel):
    task = COMPUTER_VISION.DETECTION

    def __init__(self, device=None, jit=False):
        pass

    def train(self, niter=1):
        pass

    def eval(self, niter=1):
        raise NotImplementedError("MLPerf Maskrcnn does not support inference.")

if __name__ == "__main__":
    pass
