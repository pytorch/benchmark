import argparse
from typing import Generator, List, Optional

import torch

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)


class Operator(BenchmarkOperator):

    DEFAULT_METRICS = ["test_metric"]

    def __init__(self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None):
        super().__init__(tb_args, extra_args)

    @register_benchmark(label="new_op_label")
    def test_op(self, x: torch.Tensor):
        return lambda: x

    def get_x_val(self, example_inputs):
        return example_inputs[0].shape

    def get_x_vals(self) -> List[int]:
        return [2**n for n in [1, 2, 3]]

    def get_input_iter(self) -> Generator:
        for x in self.get_x_vals():
            yield (torch.Tensor(torch.randn(x, device=self.device, dtype=self.dtype)),)

    @register_metric(x_only=True)
    def test_metric(
        self, fn_name: str, example_inputs, metrics: BenchmarkOperatorMetrics
    ):
        return [ex.shape[0] + 2 for ex in example_inputs]

    @register_metric()
    def test_metric_per_benchmark(
        self, fn_name: str, example_inputs, metrics: BenchmarkOperatorMetrics
    ):
        return [ex.shape[0] + 3 for ex in example_inputs]
