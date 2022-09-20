from torchbenchmark.tasks import COMPUTER_VISION
import torch.optim as optim
import torch
from torchbenchmark.util.model import BenchmarkModel

class Model(BenchmarkModel):
    task = COMPUTER_VISION.GENERATION

    DEFAULT_TRAIN_BSIZE = 32
    DEFAULT_EVAL_BSIZE = 16

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(model_name="squeezenet1_1", test=test, device=device, jit=jit,
                         batch_size=batch_size, extra_args=extra_args)
        self.epoch_size = 16

    def train(self):
        optimizer = optim.Adam(self.model.parameters())
        loss = torch.nn.CrossEntropyLoss()
        optimizer.zero_grad()
        for _ in range(self.epoch_size):
            pred = self.model(*self.example_inputs)
            y = torch.empty(pred.shape[0], dtype=torch.long, device=self.device).random_(pred.shape[1])
            loss(pred, y).backward()
        optimizer.step()
