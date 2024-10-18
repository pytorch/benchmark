import argparse
from typing import Callable, Generator, List, Optional

import torch

from torchbenchmark.util.triton_op import BenchmarkOperator, register_benchmark

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
except ModuleNotFoundError:
    LigerFusedLinearCrossEntropyLoss = None

# Reference: https://github.com/linkedin/Liger-Kernel/blob/\
# 3d0653b035222cbb845435a1994854e4fd219107/benchmark/scripts/benchmark_fused_linear_cross_entropy.py


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=4096, help="hidden size")
    parser.add_argument("--vocab-size", type=int, default=128256, help="vocab size")
    return parser.parse_args(args)


class TorchLMHeadCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param reduction: reduction method
    """

    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype
        )
        self.ce_loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, input, target):
        logits = self.lin(input)
        return self.ce_loss(logits, target)


class LigerLMHeadCE(torch.nn.Module):
    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype
        )
        self.ce_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, input, target):
        return self.ce_loss(self.lin.weight, input, target)


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        op_args = parse_op_args(self.extra_args)
        self.hidden_size = op_args.hidden_size
        self.vocab_size = op_args.vocab_size
        self.baseline_model = TorchLMHeadCE(
            H=self.hidden_size, V=self.vocab_size, dtype=self.dtype
        ).to(self.device)
        self.liger_model = LigerLMHeadCE(
            H=self.hidden_size, V=self.vocab_size, dtype=self.dtype
        ).to(self.device)
        self.use_cuda_graphs = False

    def get_input_iter(self) -> Generator:
        for BT in [2**i for i in range(12, 16)]:
            _input = torch.randn(
                BT,
                self.hidden_size,
                requires_grad=True,
                dtype=self.dtype,
                device=self.device,
            )
            target = torch.randint(
                self.vocab_size, (BT, 1), dtype=torch.long, device=self.device
            ).squeeze(1)
            yield _input, target

    @register_benchmark(baseline=True)
    def LMHeadCE(self, input, target) -> Callable:
        return lambda: self.baseline_model(input, target)

    @register_benchmark()
    def LigerLMHeadCE(self, input, target) -> Callable:
        return lambda: self.liger_model(input, target)

    @register_benchmark()
    def inductor_fused_linear_cross_entropy(self, input, target) -> Callable:
        compiled = torch.compile(self.baseline_model, dynamic=False)
        return lambda: compiled(input, target)

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        y = fwd_fn()
        return lambda: y.backward(retain_graph=True)
