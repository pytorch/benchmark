import argparse
import os
from typing import Any, Callable, Generator, List, Optional, Tuple

import fbgemm_gpu.experimental.gen_ai  # noqa: F401

import torch
import triton

try:
    from gen_ai.llm_inference.fb.llm.llama_layers import (
        quantize_fp8_row,
        rms_norm,
        rms_norm_fp8_rowwise_quant,
        silu_mul,
        silu_mul_fp8_rowwise_quant,
    )

    HAS_FB_IMPORT = True
except ImportError:
    HAS_FB_IMPORT = False

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val,
)


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TorchBench FP8 fused quant gemm rowwise operator Benchmark"
    )
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    args = parser.parse_args(args)
    return args


from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
    matmul_fp8_row as triton_fp8_row,
)

BUILDIN_SHAPES = [
    (1, 2304, 2048),
    (1, 8192, 16384),
    (4, 4096, 2304),
    (4, 13312, 2048),
    (8, 2304, 2304),
    (8, 8192, 6656),
    (16, 4096, 6656),
    (16, 13312, 13312),
    (32, 2304, 16384),
    (32, 8192, 13312),
    (64, 4096, 2048),
    (64, 13312, 2048),
    (128, 2304, 6656),
    (128, 8192, 2304),
    (2048, 8192, 2048),
    (2048, 13312, 6656),
    (4096, 2304, 13312),
    (4096, 13312, 2304),
    (16384, 4096, 16384),
    (16384, 8192, 13312),
]


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["tflops"]
    DEFAULT_PRECISION = "fp32"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        addmm_args = parse_args(self.extra_args)
        if addmm_args.m and addmm_args.n and addmm_args.k:
            self.shapes = [(addmm_args.m, addmm_args.n, addmm_args.k)]
        else:
            self.shapes = BUILDIN_SHAPES

    @register_benchmark(enabled=HAS_FB_IMPORT)
    def rms_norm_fused(self, x1, x2, wq, w_scale, wd) -> Callable:
        def _impl(x1, x2, wq, w_scale, wd):
            xq, x_scale = rms_norm_fp8_rowwise_quant(x1, wd)
            if torch.version.hip:
                # use CK kernel for AMD
                return torch.ops.fbgemm.f8f8bf16_rowwise(xq, wq, x_scale, w_scale)
            return triton_fp8_row(xq, wq, x_scale, w_scale)

        return lambda: _impl(x1, x2, wq, w_scale, wd)

    @register_benchmark(enabled=HAS_FB_IMPORT)
    def rms_norm_quant(self, x1, x2, wq, w_scale, wd) -> Callable:
        def _impl(x1, x2, wq, w_scale, wd):
            (m, k) = x1.shape
            x = rms_norm(x1.view(1, m, k), wd).view(m, k)
            xq, x_scale = quantize_fp8_row(x, use_triton=True)
            if torch.version.hip:
                # use CK kernel for AMD
                return torch.ops.fbgemm.f8f8bf16_rowwise(xq, wq, x_scale, w_scale)
            return triton_fp8_row(xq, wq, x_scale, w_scale)

        return lambda: _impl(x1, x2, wq, w_scale, wd)

    @register_benchmark(enabled=HAS_FB_IMPORT)
    def silu_mul_fused(self, x1, x2, wq, w_scale, wd) -> Callable:
        def _impl(x1, x2, wq, w_scale, wd):
            xq, x_scale = silu_mul_fp8_rowwise_quant(x1, x2)
            if torch.version.hip:
                # use CK kernel for AMD
                return torch.ops.fbgemm.f8f8bf16_rowwise(xq, wq, x_scale, w_scale)
            return triton_fp8_row(xq, wq, x_scale, w_scale)

        return lambda: _impl(x1, x2, wq, w_scale, wd)

    @register_benchmark(enabled=HAS_FB_IMPORT)
    def silu_mul_quant(self, x1, x2, wq, w_scale, wd) -> Callable:
        def _impl(x1, x2, wq, w_scale, wd):
            y = torch.empty_like(x1)
            x = silu_mul(x1, x2, y)
            xq, x_scale = quantize_fp8_row(x, use_triton=True)
            if torch.version.hip:
                # use CK kernel for AMD
                return torch.ops.fbgemm.f8f8bf16_rowwise(xq, wq, x_scale, w_scale)
            return triton_fp8_row(xq, wq, x_scale, w_scale)

        return lambda: _impl(x1, x2, wq, w_scale, wd)

    @register_metric()
    def tflops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> List[float]:
        x1, _, wq, _, _ = example_inputs
        m, k = x1.size()
        n, k = wq.size()
        flops = m * k * 2 * n
        return flops / metrics.latency / 1e12 * 1e3

    @register_x_val(label="(M, N, K)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        x1, _, wq, _, _ = example_inputs
        m, k = x1.size()
        n, k = wq.size()
        return (m, n, k)

    def get_input_iter(self) -> Generator:
        for shape in self.shapes:
            m, n, k = shape
            x1 = torch.randn((m, k), device=self.device, dtype=torch.bfloat16)
            x2 = torch.randn((m, k), device=self.device, dtype=torch.bfloat16)
            w = torch.randn((n, k), device=self.device, dtype=torch.bfloat16)
            wd = torch.randn((k), device=self.device, dtype=torch.bfloat16)

            wq, w_scale = quantize_fp8_row(w, use_triton=True)

            yield x1, x2, wq, w_scale, wd

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["density"],  # argument names to use as an x-axis for the plot
                x_vals=self.output.x_vals,  # different possible values for `x_name`
                line_arg="provider",  # argument name whose value corresponds to a different line in the plot
                line_vals=[
                    "_torch",
                    "_rms_norm_fused",
                    "_silu_mul_fused",
                    "_rms_norm_quant",
                    "_silu_mul_quant",
                ],  # possible values for `line_arg``
                line_names=[
                    "Torch",
                    "rms_norm_fused",
                    "silu_mul_fused",
                    "rms_norm_quant",
                    "silu_mul_quant",
                ],  # label name for the lines
                styles=[
                    ("blue", "-"),
                    ("green", "-"),
                    ("yellow", "-"),
                    ("orange", "-"),
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
