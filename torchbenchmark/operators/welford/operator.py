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

from .triton_welford import (
    fused_native_layer_norm as triton_welford,
    fused_native_layer_norm_no_welford as triton_no_welford,
)


BUILDIN_SHAPES = [
    (262144, 1024),
    (262144, 1536),
    (262144, 2048),
    (262144, 2560),
    (262144, 3072),
    (262144, 4096),
    (262144, 5120),
    (262144, 6144),
    (262144, 7168),
    (262144, 8192),
]


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "speedup", "accuracy"]

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.shapes = BUILDIN_SHAPES

    @register_benchmark()
    def test_welford(self, p1, p2, p3) -> Callable:
        return lambda: triton_welford(p1, p2, p3)

    @register_benchmark(baseline=True)
    def test_no_welford(self, p1, p2, p3) -> Callable:
        return lambda: triton_no_welford(p1, p2, p3)

    def get_x_val(self, example_inputs) -> float:
        p1, p2, p3 = example_inputs
        s, d = p3.size()
        return d

    def get_input_iter(self) -> Generator:
        for shape in self.shapes:
            s, d = shape
            p1 = rand_strided((d,), (1,), device="cuda:0", dtype=torch.bfloat16)
            p2 = rand_strided((d,), (1,), device="cuda:0", dtype=torch.bfloat16)
            p3 = rand_strided((s, d), (d, 1), device="cuda:0", dtype=torch.bfloat16)
            yield p1, p2, p3

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        return same(output, baseline_output)
