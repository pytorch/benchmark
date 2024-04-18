import csv
import os
import statistics
from typing import Any, Callable, Generator, List, Optional

import numpy
import torch
import triton
from hammer.ops.triton.triton_hstu_linear import triton_addmm

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from .data_io import parse_args


BUILDIN_SHAPES = [(M * 128, 512, N) for N in [1536, 512] for M in range(4, 17)]


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "speedup", "accuracy"]
    DEFAULT_PRECISION = "bf16"

    def __init__(self, mode: str, device: str, extra_args: List[str] = []):
        super().__init__(mode=mode, device=device, extra_args=extra_args)
        if not self.extra_args:
            self.DEFAULT_NUM_BATCH = len(BUILDIN_SHAPES)
            self.shapes = BUILDIN_SHAPES
        else:
            self.shapes = [(self.tb_args.m, self.tbargs.k, self.tbargs.n)]
            self.DEFAULT_NUM_BATCH = len(self.shapes)

    @register_benchmark()
    def triton_addmm(self, a, mat1, mat2) -> Callable:
        return lambda: triton_addmm(a, mat1, mat2)

    @register_benchmark(baseline=True)
    def aten_addmm(self, a, mat1, mat2) -> Callable:
        return lambda: torch.addmm(a, mat1, mat2)

    def get_x_val(self, example_inputs) -> float:
        _, mat1, mat2 = example_inputs
        m, k = mat1.size()
        k, n = mat2.size()
        return f"{m}-{k}-{n}"

    @register_metric()
    def gbps(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        a, mat1, mat2 = example_inputs
        numel = (
            a.numel()
            + mat1.numel()
            + mat2.numel()
            + (torch.addmm(a, mat1, mat2).numel())
        )
        numel = numel * a.element_size() / 1e9
        gbps = list(map(lambda x: numel / x * 1e3, metrics.latency))
        return statistics.median(gbps)

    @register_metric()
    def tflops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        _, mat1, mat2 = example_inputs
        m, k = mat1.size()
        k, n = mat2.size()
        flops = m * k * 2 * n
        return [flops / x / 1e12 * 1e3 for x in metrics.latency]

    def get_input_iter(self) -> Generator:
        for shape in self.shapes:
            m, k, n = shape
            a = torch.randn(
                (m, n), device=self.device, dtype=self.dtype
            ).requires_grad_(False)
            mat1 = torch.randn(
                (m, k), device=self.device, dtype=self.dtype
            ).requires_grad_(False)
            mat2 = torch.randn(
                (k, n), device=self.device, dtype=self.dtype
            ).requires_grad_(False)
            yield a, mat1, mat2
        while True:
            yield None

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        accuracy = True
        try:
            torch.testing.assert_close(output, baseline_output, atol=1e-2, rtol=0.5)
        except Exception:
            accuracy = False
        finally:
            return accuracy

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["density"],  # argument names to use as an x-axis for the plot
                x_vals=self.output.x_vals,  # different possible values for `x_name`
                line_arg="provider",  # argument name whose value corresponds to a different line in the plot
                line_vals=[
                    "aten_addmm",
                    "triton_addmm",
                ],  # possible values for `line_arg``
                line_names=[
                    "ATen AddMM",
                    "Triton AddMM",
                ],  # label name for the lines
                styles=[("blue", "-"), ("green", "-")],  # line styles
                ylabel="tflops",  # label name for the y-axis
                plot_name="gemm-performance",  # name for the plot. Used also as a file name for saving the plot.
                args={},  # values for function arguments not in `x_names` and `y_name`
            )
        )
        def _plot(density, provider):
            tflops = self.output.get_y_vals(density, provider, "tflops")
            return tflops

        save_path = "/tmp/test_addmm"

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        _plot.run(show_plots=True, print_data=True, save_path="/tmp/test_addmm")
