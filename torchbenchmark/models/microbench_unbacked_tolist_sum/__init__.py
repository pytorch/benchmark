from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER
import torch


class MicrobenchUnbackedTolistSum(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1))

    def forward(self, f, ti, tv):
        l = ti.tolist()
        [torch._check_is_size(i) for i in l]
        sum = 0
        for i in l:
            torch._check(i < tv.size(0))
            sum += tv[i].item()
        return f * self.weight * sum  # force a tensor output


class Model(BenchmarkModel):
    task = OTHER.MICROBENCH
    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)
        self.model = MicrobenchUnbackedTolistSum().to(self.device)
        self.example_inputs = (
            torch.randn(self.batch_size, device=self.device),
            torch.arange(1000, dtype=torch.int32, device=self.device),
            torch.ones(1000, dtype=torch.int32, device=self.device),
        )

    def get_module(self):
        return self.model, self.example_inputs

    def eval(self):
        out = self.model(*self.example_inputs)
        return (out,)

    def train(self):
        raise NotImplementedError("Train test is not implemented.")
