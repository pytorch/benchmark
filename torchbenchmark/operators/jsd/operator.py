import argparse
from typing import Callable, Generator, List, Optional

import torch

from torchbenchmark.util.triton_op import BenchmarkOperator, register_benchmark

try:
    from liger_kernel.transformers.jsd import LigerJSD
except ModuleNotFoundError:
    LigerJSD = None

# Reference: https://github.com/linkedin/Liger-Kernel/
# blob/main/benchmark/scripts/benchmark_jsd.py


class TorchJSD(torch.nn.Module):
    def __init__(
        self,
        beta: float = 0.5,
        ignore_index: int = -100,
        dtype: torch.dtype = torch.float,
    ):
        super(TorchJSD, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction="none", log_target=True)
        self.beta = beta
        self.ignore_index = ignore_index
        self.dtype = dtype

    def forward(
        self,
        log_q: torch.Tensor,  # input
        log_p: torch.Tensor,  # target
        label=None,
    ):
        log_p, log_q = log_p.to(torch.float), log_q.to(torch.float)
        log_p, log_q = log_p.view(-1, log_p.size(-1)), log_q.view(-1, log_q.size(-1))
        m = torch.lerp(torch.exp(log_q), torch.exp(log_p), self.beta)
        loss = self.beta * self.kl(torch.log(m), log_p).sum(dim=-1) + (
            1 - self.beta
        ) * self.kl(torch.log(m), log_q).sum(dim=-1)

        if label is not None:
            loss = torch.where(label != self.ignore_index, loss, 0.0)
            n_non_ignore = (label != self.ignore_index).sum().item()
            if n_non_ignore == 0:
                loss = 0.0
            else:
                loss = (loss / n_non_ignore).sum()
        else:
            loss = (loss / log_q.shape[0]).sum()
        return loss.to(self.dtype)


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.B = 4
        self.T = 2048
        self.baseline_op = TorchJSD()
        self.liger_op = LigerJSD()
        self.use_cuda_graphs = False

    def get_input_iter(self) -> Generator:
        for V in [2**i for i in range(12, 18)]:
            _input = torch.randn(
                self.B * self.T, V, requires_grad=True, device=self.device
            ).log_softmax(dim=-1)
            target = torch.randn(self.B * self.T, V, device=self.device).log_softmax(
                dim=-1
            )
            yield _input, target

    @register_benchmark(baseline=True)
    def torch_jsd(self, _input, target) -> Callable:
        return lambda: self.baseline_op(_input, target)

    @register_benchmark()
    def liger_jsd(self, _input, target) -> Callable:
        return lambda: self.liger_op(_input, target)

    @register_benchmark()
    def inductor_jsd(self, _input, target) -> Callable:
        compiled = torch.compile(self.baseline_op, dynamic=False)
        return lambda: compiled(_input, target)

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        y = fwd_fn()
        return lambda: y.backward(retain_graph=True)

    def get_grad_to_none(self, args) -> List[torch.Tensor]:
        x = args[0]
        return [x]
