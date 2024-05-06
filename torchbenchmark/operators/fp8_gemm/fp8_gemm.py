import argparse
import os
import statistics
import torch
import triton.ops
import triton.language as tl

from triton.runtime.jit import reinterpret

from typing import Any

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)


def parse_args(args):
    parser = argparse.ArgumentParser(description="TritonBench fp8_gemm")
    parser.add_argument("--llama", action="store_true")
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["tflops", "gbps", "latency"]

    def __init__(self, mode, device, extra_args):
        super().__init__(mode=mode, device=device, extra_args=extra_args)
        self.extra_args = parse_args(extra_args)

    def get_input_iter(self):
        def args(m, n, k):
            a = torch.randn(m, k, device=self.device).to(torch.float8_e4m3fn)
            b = (
                torch.randn(k, n, device=self.device)
                .to(torch.float8_e4m3fn)
                .T.contiguous()
                .T
            )
            return (a, b)

        if self.extra_args.llama:
            name_to_shapes_70b = {
                "attn.wqkv": (8192, 1280),
                "attn.w0": (1024, 8192),
                "ffn.w13": (8192, 7168),
                "ffn.w2": (3584, 8192),
            }
            for (name, (k, n)) in name_to_shapes_70b.items():
                bsz, seq_len = 4, 4096
                m = bsz * seq_len
                yield args(m, n, k)
        else:
            for i in range(10, 15):
                for j in range(0, 4):
                    k = 2**i
                    k += k // 4 * j
                    m = n = k
                    yield args(m, n, k)

    def get_x_val(self, example_inputs) -> float:
        a, b = example_inputs
        m, k = a.size()
        _, n = b.size()
        return (m, n, k)

    @register_benchmark(baseline=True)
    def torch_fp8_gemm(self, a, b):
        return lambda: torch._scaled_mm(
            a, b, use_fast_accum=True, out_dtype=torch.float16
        )

    @register_benchmark()
    def triton_fp8_gemm(self, a, b):
        a = reinterpret(a, tl.float8e4nv)
        b = reinterpret(b, tl.float8e4nv)
        return lambda: triton.ops.matmul(a, b)

    @register_metric()
    def gbps(
        self, fn, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        def nbytes(t):
            return t.numel() * t.element_size()

        a, b = example_inputs
        c = fn()
        c = c[0] if isinstance(c, tuple) else c

        m, k = a.shape
        _, n = b.shape
        gb = (nbytes(a) + nbytes(b) + nbytes(c)) / 1e9
        return list(map(lambda x: gb / x * 1e3, metrics.latency))

    @register_metric()
    def tflops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        a, b = example_inputs
        m, k = a.size()
        _, n = b.size()
        flops = 2 * m * n * k
        return [flops / x / 1e12 * 1e3 for x in metrics.latency]

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=[
                    "m",
                    "n",
                    "k",
                ],  # argument names to use as an x-axis for the plot
                x_vals=self.output.x_vals,  # different possible values for `x_name`
                line_arg="provider",  # argument name whose value corresponds to a different line in the plot
                line_vals=[
                    "torch_fp8_gemm",
                    "triton_fp8_gemm",
                ],  # possible values for `line_arg``
                line_names=[
                    "torch_fp8_gemm",
                    "triton_fp8_gemm",
                ],  # label name for the lines
                styles=[("blue", "-"), ("green", "-")],
                ylabel="tflops",  # label name for the y-axis
                plot_name="fp8-gemm-performance",  # name for the plot. Used also as a file name for saving the plot.
                args={},  # values for function arguments not in `x_names` and `y_name`
            )
        )
        def _plot(m, n, k, provider):
            tflops = self.output.get_y_vals((m, n, k), provider, "tflops")
            return tflops

        save_path = "/tmp/fp8_gemm"

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        _plot.run(show_plots=True, print_data=True, save_path=save_path)
