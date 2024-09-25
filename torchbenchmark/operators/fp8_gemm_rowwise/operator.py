import argparse
import os
from typing import Any, Callable, Generator, List, Optional, Tuple

import fbgemm_gpu.experimental.gen_ai  # noqa: F401

import torch
import triton

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val,
)


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TorchBench Addmm operator Benchmark")
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument(
        "--no_fp8_fast_accum", dest="fp8_fast_accum", action="store_false"
    )
    parser.add_argument("--no_use_tma", dest="use_tma", action="store_false")
    args = parser.parse_args(args)
    return args


try:
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
        matmul_fp8_row as triton_fp8_row,
    )

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


try:
    cutlass_fp8_row = torch.ops.fbgemm.f8f8bf16_rowwise
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False

try:
    cublas_fp8_row = torch.ops.fbgemm.f8f8bf16_cublas
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import scale_fp8_row

    HAS_CUBLAS = True
except ImportError:
    HAS_CUBLAS = False


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

E4M3_MAX_POS: float = torch.finfo(torch.float8_e4m3fn).max
EPS: float = 1e-12
FP16_MAX_POS: float = torch.finfo(torch.float16).max


def fp8_row_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Quantize an input tensor and return the fp8 tensor and its inverse scale.
    x_row_max = torch.max(torch.abs(x), dim=1).values
    scale = E4M3_MAX_POS / torch.clamp(x_row_max, EPS)
    if x.dtype is torch.float16:
        scale = torch.clamp(scale, max=FP16_MAX_POS)
    xq = torch.clamp(x * scale[:, None], min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS).to(
        torch.float8_e4m3fn
    )
    return xq, scale.reciprocal().to(torch.float32)


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["tflops", "speedup", "accuracy"]
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
        self.fp8_fast_accum = addmm_args.fp8_fast_accum
        self.use_tma = addmm_args.use_tma

    @register_benchmark(enabled=HAS_TRITON, baseline=True)
    def _triton(self, xq, wq, x_scale, w_scale) -> Callable:
        return lambda: triton_fp8_row(
            xq,
            wq,
            x_scale,
            w_scale,
            fp8_fast_accum=self.fp8_fast_accum,
            tma_persistent=self.use_tma,
        )

    @register_benchmark(enabled=HAS_CUTLASS)
    def _cutlass(self, xq, wq, x_scale, w_scale) -> Callable:
        return lambda: cutlass_fp8_row(
            xq, wq, x_scale, w_scale, use_fast_accum=self.fp8_fast_accum
        )

    @register_benchmark(enabled=HAS_CUBLAS, label=f"cublas_{torch.version.cuda}")
    def _cublas(self, xq, wq, x_scale, w_scale) -> Callable:
        return lambda: scale_fp8_row(
            cublas_fp8_row(xq, wq, use_fast_accum=self.fp8_fast_accum), x_scale, w_scale
        )

    # TODO: add cublas rowwise FP8 kernel
    # @register_benchmark(baseline=True)
    # def _torch(self, xq, wq, x_scale, w_scale) -> Callable:
    #     def _cublass(xq, wq, x_scale, w_scale):
    #         output, _ = torch._scaled_mm(
    #             xq, wq.T, use_fast_accum=True, out_dtype=torch.bfloat16
    #         )
    #         return output * x_scale[:, None] * w_scale[None, :]

    #     return lambda: _cublass(xq, wq, x_scale, w_scale)

    @register_metric()
    def tflops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> List[float]:
        xq, wq, _, _ = example_inputs
        m, k = xq.size()
        n, k = wq.size()
        flops = m * k * 2 * n
        return flops / metrics.latency / 1e12 * 1e3

    @register_x_val(label="(M, N, K)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        xq, wq, _, _ = example_inputs
        m, k = xq.size()
        n, k = wq.size()
        return (m, n, k)

    def get_input_iter(self) -> Generator:
        for shape in self.shapes:
            m, n, k = shape
            x = torch.randn(
                (m, k), device=self.device, dtype=torch.bfloat16
            ).requires_grad_(False)
            w = torch.randn(
                (n, k), device=self.device, dtype=torch.bfloat16
            ).requires_grad_(False)
            xq, x_scale = fp8_row_quantize(x)
            wq, w_scale = fp8_row_quantize(w)
            yield xq, wq, x_scale, w_scale

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
                    "_torch",
                    "_triton",
                    "_cutlass",
                    "_cublas",
                ],  # possible values for `line_arg``
                line_names=[
                    "Torch",
                    "Triton",
                    "Cutlass",
                    "cuBLAS",
                ],  # label name for the lines
                styles=[("blue", "-"), ("green", "-"), ("yellow", "-")],  # line styles
                ylabel="tflops",  # label name for the y-axis
                plot_name="gemm-performance",  # name for the plot. Used also as a file name for saving the plot.
                args={},  # values for function arguments not in `x_names` and `y_name`
            )
        )
        def _plot(density, provider):
            tflops = self.output.get_y_vals(density, provider, "tflops")
            return tflops

        save_path = "/tmp/test_fp8_gemm_rowwise"

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        _plot.run(show_plots=True, print_data=True, save_path=save_path)
