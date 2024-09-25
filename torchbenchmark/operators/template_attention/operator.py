import argparse
import csv
import os
import statistics
from typing import Any, Callable, Generator, List, Optional

import numpy
import torch
import triton
from torch._dynamo.testing import rand_strided, same

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from .triton_attention import (
    triton_attention_no_exp2 as triton_test_no_exp2,
    triton_attention_with_exp2 as triton_test_with_exp2,
)


BUILDIN_SHAPES = [
    (16, 16, 4096, 64),
]


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "speedup", "accuracy"]

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.shapes = BUILDIN_SHAPES

    @register_benchmark(baseline=True)
    def test_no_exp2(self, p1, p2, p3) -> Callable:
        return lambda: triton_test_no_exp2(p1, p2, p3)

    @register_benchmark()
    def test_with_exp2(self, p1, p2, p3) -> Callable:
        return lambda: triton_test_with_exp2(p1, p2, p3)

    def get_x_val(self, example_inputs) -> float:
        p1, p2, p3 = example_inputs
        batch_size, num_heads, num_queries, m = p3.size()
        return num_queries

    def get_input_iter(self) -> Generator:
        for shape in self.shapes:
            batch_size, num_heads, num_queries, m = shape
            arg0_1 = rand_strided(
                (16, 16, 4096, 64),
                (4194304, 262144, 64, 1),
                device="cuda:0",
                dtype=torch.float16,
            )
            arg1_1 = rand_strided(
                (16, 16, 4096, 64),
                (4194304, 262144, 64, 1),
                device="cuda:0",
                dtype=torch.float16,
            )
            arg2_1 = rand_strided(
                (16, 16, 4096, 64),
                (4194304, 262144, 64, 1),
                device="cuda:0",
                dtype=torch.float16,
            )
            yield arg0_1, arg1_1, arg2_1

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        return same(output, baseline_output)
