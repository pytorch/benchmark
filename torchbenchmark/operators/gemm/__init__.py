import argparse
import os
import statistics
from typing import Callable, Generator, List, Optional

import numpy
import torch
import triton

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from .triton_matmul import matmul as triton_matmul

BUILDIN_SHAPES = [
    (256, 256, 256),
    (384, 384, 384),
    (512, 512, 512),
    (640, 640, 640),
    (768, 768, 768),
    (896, 896, 896),
    (1024, 1024, 1024),
    (1152, 1152, 1152),
    (1280, 1280, 1280),
    (1408, 1408, 1408),
    (1536, 1536, 1536),
    (1664, 1664, 1664),
    (1792, 1792, 1792),
    (1920, 1920, 1920),
    (2048, 2048, 2048),
    (2176, 2176, 2176),
    (2304, 2304, 2304),
    (2432, 2432, 2432),
    (2560, 2560, 2560),
    (2688, 2688, 2688),
    (2816, 2816, 2816),
    (2944, 2944, 2944),
    (3072, 3072, 3072),
    (3200, 3200, 3200),
    (3328, 3328, 3328),
    (3456, 3456, 3456),
    (3584, 3584, 3584),
    (3712, 3712, 3712),
    (3840, 3840, 3840),
    (3968, 3968, 3968),
    (4096, 4096, 4096),
]


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TorchBench Gemm operator Benchmark")
    parser.add_argument("--m", default=8, type=int)
    parser.add_argument("--k", default=8, type=int)
    parser.add_argument("--n", default=8, type=int)
    args = parser.parse_args(args)
    return args


class Operator(BenchmarkOperator):
    USE_BUILTIN_SHAPES = True

    def __init__(self, mode: str, device: str, extra_args: List[str] = []):
        if not extra_args:
            self.USE_BUILTIN_SHAPES = True
            self.DEFAULT_NUM_BATCH = len(BUILDIN_SHAPES)
            self.extra_builtin_metrics = ["speedup", "accuracy"]
        else:
            self.USE_BUILTIN_SHAPES = False
            self.DEFAULT_NUM_BATCH = 1
            self.tbargs = parse_args(self.extra_args)
        super().__init__(mode=mode, device=device, extra_args=extra_args)
        self.required_metrics = list(
            set(self.required_metrics + self.extra_builtin_metrics)
        )

    @register_benchmark()
    def triton_matmul(self, a, b) -> Callable:
        return lambda: triton_matmul(a, b)

    @register_benchmark(baseline=True)
    def aten_matmul(self, a, b) -> Callable:
        return lambda: torch.matmul(a, b)

    def get_x_val(self, example_inputs) -> float:
        # x-value: computation intensity
        a, w = example_inputs
        m, k = a.size()
        k, n = w.size()
        # computation intensity for the shape m, n, k
        intensity = 1 / (1 / n + 1 / m + 1 / k)
        return intensity

    @register_metric()
    def gbps(self, fn, example_inputs, metrics: BenchmarkOperatorMetrics) -> float:
        a, w = example_inputs
        numel = a.numel() + w.numel() + (torch.mm(a, w).numel())
        numel = numel * a.element_size() / 1e9
        gbps = list(map(lambda x: numel / x * 1e3, metrics.latency))
        return statistics.median(gbps)

    @register_metric(skip_baseline=True)
    def xShape(self, fn, example_inputs, metrics: BenchmarkOperatorMetrics) -> list[int]:
        a, w = example_inputs
        m, k = a.size()
        k, n = w.size()
        return [m, k, n]

    @register_metric()
    def tflops(self, fn, example_inputs, metrics: BenchmarkOperatorMetrics) -> float:
        a, w = example_inputs
        m, k = a.size()
        k, n = w.size()
        flops = m * k * 2 * n
        latency = numpy.median(metrics.latency)
        return flops / latency / 1e12 * 1e3

    def get_input_iter(self) -> Generator:
        if self.USE_BUILTIN_SHAPES:
            for shape in BUILDIN_SHAPES:
                m, k, n = shape
                a = torch.randn(
                    (m, k), device=self.device, dtype=torch.float16
                ).requires_grad_(False)
                w = torch.randn(
                    (k, n), device=self.device, dtype=torch.float16
                ).requires_grad_(False)
                yield a, w
            while True:
                yield None
        else:
            meta_tensor = torch.randn((self.tbargs.m, self.tbargs.k), device="meta")
            yield torch.randn_like(meta_tensor, device=self.device).requires_grad(False)

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        accuracy = True
        try:
            torch.testing.assert_close(output, baseline_output, rol=1e-5)
            # if not (loss == None and baseline_loss == None):
            #     torch.testing.assert_close(loss.grad, baseline_loss.grad)
        except AssertionError:
            # either the output tensor or the loss grad tensor does not match
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
                    "triton_matmul",
                ],  # possible values for `line_arg``
                line_names=[
                    "Triton GEMM",
                ],  # label name for the lines
                styles=[("blue", "-"), ("green", "-")],  # line styles
                ylabel="speedup",  # label name for the y-axis
                plot_name="gemm-performance",  # name for the plot. Used also as a file name for saving the plot.
                args={},  # values for function arguments not in `x_names` and `y_name`
            )
        )
        def _plot(density, provider):
            speedup = self.output.get_y_vals(density, provider, "speedup")
            return speedup

        save_path = "/tmp/test_gemm"

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        _plot.run(show_plots=True, print_data=True, save_path="/tmp/test_gemm")
