"""
Based on PT2 test case: https://github.com/pytorch/pytorch/issues/121661
Motivated by https://www.thonking.ai/p/short-supporting-mixtral-in-gpt-fast,
gather + gemv is the primary kernel driving mixtral perf.
"""

import argparse
import csv
import os
import statistics
from typing import Any, Callable, Generator, List, Optional

import numpy
import torch
import triton


from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from .triton_gather_gemv import triton_gemv_0 as triton_test_0
from torch._dynamo.testing import rand_strided


class Operator(BenchmarkOperator):

    @register_metric()
    def gbps(self, fn_name, example_inputs, metrics: BenchmarkOperatorMetrics):
        arg0_1, arg1_1, arg2_1 = example_inputs
        return (
            2
            * arg2_1.size(0) * arg2_1.size(0)
            * arg0_1.element_size()
            / metrics.latency
            * 1e-6
        )

    def __init__(self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None):
        super().__init__(tb_args, extra_args)

    @register_benchmark(baseline=True)
    def test_0(self, p1, p2, p3) -> Callable:
        return lambda: triton_test_0(p1, p2, p3)

    @register_benchmark(baseline=True)
    def test_eager(self, w, idx, x):
        return lambda: w[idx].to(x.dtype) @ x

    @register_benchmark()
    def test_inductor(self, w, idx, x):
        @torch.compile
        def gather_gemv(w, idx, x):
            return w[idx].to(x.dtype) @ x

        gather_gemv(w, idx, x)  # warmup
        return lambda: gather_gemv(w, idx, x)

    def get_x_val(self, example_inputs) -> float:
        arg0_1, arg1_1, arg2_1 = example_inputs
        s = arg2_1.size(0)
        return s

    def get_input_iter(self) -> Generator:
        for i in range(11, 15):
            S = 2 ** i
            arg0_1 = rand_strided((8, S, S), (S*S, S, 1), device='cuda:0', dtype=torch.int8)
            arg1_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.int64)
            arg2_1 = rand_strided((S, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
            yield arg0_1, arg1_1, arg2_1

