import torch
import torch.optim as optim
from .mobilenetv3 import MobileNetV3
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION


class Model(BenchmarkModel):
    task = COMPUTER_VISION.CLASSIFICATION
    def __init__(self, device=None, jit=False):
        super().__init__()
        """ Required """
        self.device = device
        self.jit = jit
        self.model = MobileNetV3()
        if self.jit:
            self.model = torch.jit.script(self.model)
        input_size = (1, 3, 224, 224)
        self.example_inputs = (torch.randn(input_size),)

    def get_module(self):
        return self.model, self.example_inputs

    def train(self, niter=3):
        optimizer = optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        loss = torch.nn.CrossEntropyLoss()
        for _ in range(niter):
            optimizer.zero_grad()
            pred = self.model(*self.example_inputs)
            y = torch.empty(pred.shape[0], dtype=torch.long).random_(pred.shape[1])
            loss(pred, y).backward()
            optimizer.step()

    def eval(self, niter=1):
        model, example_inputs = self.get_module()
        model.eval()
        for i in range(niter):
            model(*example_inputs)


if __name__ == "__main__":
    m = Model(device="cuda", jit=False)
    module, example_inputs = m.get_module()
    module(*example_inputs)
    m.train(niter=1)
    m.eval(niter=1)
