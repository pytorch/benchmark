import argparse
from typing import Callable, Generator, List, Optional

import torch

from torchbenchmark.util.triton_op import BenchmarkOperator, register_benchmark

try:
    from liger_kernel.transformers.kl_div import LigerKLDIVLoss
except ModuleNotFoundError:
    LigerKLDIVLoss = None

# Reference: https://github.com/linkedin/Liger-Kernel/
# blob/main/benchmark/scripts/benchmark_kl_div.py


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.B = 8
        self.T = 512
        self.baseline_op = torch.nn.KLDivLoss(reduction="batchmean").to(self.device)
        self.liger_op = LigerKLDIVLoss(reduction="batchmean").to(self.device)
        self.use_cuda_graphs = False

    def get_input_iter(self) -> Generator:
        for V in [2**i for i in range(12, 18)]:
            _input = torch.randn(
                self.B * self.T, V, requires_grad=True, device=self.device
            ).log_softmax(dim=-1)
            target = torch.randn(self.B * self.T, V, device=self.device).softmax(dim=-1)
            yield _input, target

    @register_benchmark(baseline=True)
    def torch_kl_div(self, input, target) -> Callable:
        return lambda: self.baseline_op(input, target)

    @register_benchmark()
    def liger_kl_div(self, input, target) -> Callable:
        return lambda: self.liger_op(input, target)

    @register_benchmark()
    def inductor_kl_div(self, input, target) -> Callable:
        compiled = torch.compile(self.baseline_op, dynamic=False)
        return lambda: compiled(input, target)

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        y = fwd_fn()
        return lambda: y.backward(retain_graph=True)

    def get_grad_to_none(self, args) -> List[torch.Tensor]:
        return [args[0]]
