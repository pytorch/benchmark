import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch

from torch.nn import CrossEntropyLoss

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_x_val,
)

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
except ModuleNotFoundError:
    LigerCrossEntropyLoss = None

# Reference: https://github.com/linkedin/Liger-Kernel/
# blob/main/benchmark/scripts/benchmark_cross_entropy.py


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.B = 8
        self.T = 2048
        self.baseline_op = CrossEntropyLoss()
        self.liger_op = LigerCrossEntropyLoss()
        self.use_cuda_graphs = False

    def get_input_iter(self) -> Generator:
        for V in [2**i for i in range(12, 18)]:
            _input = torch.randn(
                self.B * self.T,
                V,
                requires_grad=True,
                device=self.device,
            )
            target = torch.randint(V, (self.B * self.T, 1), device=self.device).squeeze(
                1
            )
            yield _input, target

    @register_benchmark(baseline=True)
    def cross_entropy_loss(self, input, target) -> Callable:
        return lambda: self.baseline_op(input, target)

    @register_benchmark()
    def liger_cross_entropy_loss(self, input, target) -> Callable:
        return lambda: self.liger_op(input, target)

    @register_benchmark()
    def inductor_cross_entropy_loss(self, input, target) -> Callable:
        compiled = torch.compile(self.baseline_op, dynamic=False)
        return lambda: compiled(input, target)

    @register_x_val(label="(B, T, V)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        v = example_inputs[0].size(-1)
        return (self.B, self.T, v)

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        y = fwd_fn()
        # TODO: how to pass grad_to_none=[_input]?
        return lambda: y.backward(retain_graph=True)

    def get_grad_to_none(self, args) -> List[torch.Tensor]:
        x = args[0]
        return [x]
