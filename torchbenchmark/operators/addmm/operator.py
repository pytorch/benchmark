import os
import argparse
from typing import Any, Callable, Generator, List, Optional, Tuple

import numpy
import torch
import torch._inductor.config as inductor_config
import triton
from hammer.ops.triton.triton_hstu_linear import _addmm_fwd, triton_addmm

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val,
)

from .data_io import parse_args


BUILDIN_SHAPES = [
    (20120, 1536, 512),
    (34579, 1536, 512),
    (34839, 1536, 512),
    (35561, 1536, 512),
    (35916, 1536, 512),
    (19735, 1536, 512),
    (34533, 1536, 512),
    (35791, 1536, 512),
    (35844, 1536, 512),
    (20116, 1536, 512),
    (33887, 1536, 512),
    (20203, 1536, 512),
    (33961, 1536, 512),
    (19747, 1536, 512),
    (34181, 1536, 512),
    (35541, 1536, 512),
    (36032, 1536, 512),
    (15168, 1536, 512),
    (35249, 1536, 512),
    (33894, 1536, 512),
    (20067, 1536, 512),
    (27456, 1536, 512),
    (19410, 1536, 512),
    (35884, 1536, 512),
    (35917, 1536, 512),
    (19632, 1536, 512),
    (35656, 1536, 512),
    (35405, 1536, 512),
    (35503, 1536, 512),
    (35504, 1536, 512),
    (35605, 1536, 512),
    (34238, 1536, 512),
    (33660, 1536, 512),
    (35410, 1536, 512),
    (20211, 1536, 512),
    (34308, 1536, 512),
    (34516, 1536, 512),
    (20224, 1536, 512),
    (35678, 1536, 512),
    (35380, 1536, 512),
    (35901, 1536, 512),
    (20068, 1536, 512),
]


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["tflops", "best_config"]
    DEFAULT_PRECISION = "bf16"

    def __init__(self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None):
        super().__init__(tb_args, extra_args)
        addmm_args = parse_args(self.extra_args)
        if addmm_args.m and addmm_args.n and addmm_args.k:
            self.shapes = [(addmm_args.m, addmm_args.k, addmm_args.n)]
        else:
            self.shapes = BUILDIN_SHAPES
        self.col_major = addmm_args.col_major

    @register_benchmark()
    def triton_addmm(self, a, mat1, mat2) -> Callable:
        return lambda: triton_addmm(a, mat1, mat2)

    @register_benchmark(baseline=True)
    def aten_addmm(self, a, mat1, mat2) -> Callable:
        return lambda: torch.addmm(a, mat1, mat2)

    @register_benchmark()
    def pt2_triton_matmul(self, a, mat1, mat2) -> Callable:
        torch._dynamo.reset()
        with inductor_config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON",
            autotune_fallback_to_aten=False,
        ):
            f = lambda a, mat1, mat2: torch.addmm(a, mat1, mat2)
            compiled = torch.compile(f, dynamic=False)
            compiled(a, mat1, mat2)
        return lambda: compiled(a, mat1, mat2)

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
        return numel / metrics.latency * 1e3

    @register_metric()
    def tflops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        _, mat1, mat2 = example_inputs
        m, k = mat1.size()
        k, n = mat2.size()
        flops = m * k * 2 * n
        return flops / metrics.latency / 1e12 * 1e3

    @register_x_val(label="(M, N, K)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        # x-value: computation intensity
        a, mat1, mat2 = example_inputs
        m, k = mat1.size()
        k, n = mat2.size()
        return (m, n, k)

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
            if self.col_major:
                mat2 = mat2.T.contiguous().T
            yield a, mat1, mat2

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
