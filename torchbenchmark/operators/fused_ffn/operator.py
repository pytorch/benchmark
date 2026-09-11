import argparse
import os
from typing import Any, Callable, Generator, List, Optional, Tuple

import torch
import triton

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val,
)

from .kernel import eager_ffn, fused_ffn


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TorchBench for fused FFN operator Benchmark"
    )
    parser.add_argument("--b-t", type=int)
    parser.add_argument("--h-d", type=int)
    parser.add_argument("--d", type=int)
    args = parser.parse_args(args)
    return args


BUILDIN_SHAPES = [
    (b_t, h_d, d)
    for h_d, d in [(128, 256), (1024, 512), (8192, 2048)]
    for b_t in [1024, 2048, 4096, 8192, 16384]
]


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency"]
    DEFAULT_PRECISION = "bf16"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        addmm_args = parse_args(self.extra_args)
        if addmm_args.m and addmm_args.n and addmm_args.k:
            self.shapes = [(addmm_args.m, addmm_args.n, addmm_args.k)]
        else:
            self.shapes = BUILDIN_SHAPES

    @register_benchmark()
    def fused_ffn_op(self, x, w13, w2) -> Callable:
        return lambda: fused_ffn(x, w13, w2)

    @register_benchmark()
    def eager_ffn_op(self, x, w13, w2) -> Callable:
        return lambda: eager_ffn(x, w13, w2)

    @register_metric()
    def tflops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> List[float]:
        x, w13, w2 = example_inputs
        B_T, _ = x.size()
        H_D_2, _ = w13.size()
        H_D, D = w2.size()
        # gemm #1
        flops = 2 * B_T * H_D_2 * D
        # gemm #2
        flops += 2 * B_T * H_D * D
        return flops / metrics.latency / 1e12 * 1e3

    @register_x_val(label="(B_T, Hidden_D, D)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        x, w13, w2 = example_inputs
        B_T, D = x.size()
        H_D, D = w2.size()
        return (B_T, H_D, D)

    def get_input_iter(self) -> Generator:
        for shape in self.shapes:
            b_t, h_d, d = shape
            x = torch.randn((b_t, d), device=self.device, dtype=self.dtype)
            w13 = torch.randn((2 * h_d, d), device=self.device, dtype=self.dtype)
            w2 = torch.randn((h_d, d), device=self.device, dtype=self.dtype)

            yield x, w13, w2

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["shape"],  # argument names to use as an x-axis for the plot
                x_vals=self.output.x_vals,  # different possible values for `x_name`
                line_arg="provider",  # argument name whose value corresponds to a different line in the plot
                line_vals=[
                    "fused_ffn",
                    "eager_ffn",
                ],  # possible values for `line_arg``
                line_names=[
                    "Fused FFN",
                    "Eager FFN",
                ],  # label name for the lines
                styles=[
                    ("blue", "-"),
                    ("green", "-"),
                ],  # line styles
                ylabel="tflops",  # label name for the y-axis
                plot_name="gemm-performance",  # name for the plot. Used also as a file name for saving the plot.
                args={},  # values for function arguments not in `x_names` and `y_name`
            )
        )
        def _plot(density, provider):
            tflops = self.output.get_y_vals(density, provider, "tflops")
            return tflops

        save_path = self.get_temp_path()

        os.mkdirs(save_path, exist_ok=True)

        _plot.run(show_plots=True, print_data=True, save_path=save_path)
