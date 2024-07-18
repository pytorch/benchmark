"""
Compute a bf16 (activation) x int4 (weight) gemm.
Inspired by [gpt-fast](https://github.com/pytorch-labs/gpt-fast)
ATen kernels from tinygemm
Triton implementation by @jlebar: https://gist.github.com/jlebar/3435b2c00deea53258887ce37231e5e2
"""

import argparse
import os
import statistics
import torch
import triton.ops
import triton.language as tl

from typing import Any, Optional, List

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)
from .kernel import pack_2xint4, matmul, matmul_kernel, _group_quantize_tensor


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["tflops", "gbps", "latency", "best_config"]

    def __init__(self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None):
        super().__init__(tb_args, extra_args)
        # `Group size` and `inner K tiles` are defaults from gpt-fast.
        self.group_size = 32
        self.inner_k_tiles = 8

    def get_input_iter(self):
        def args(B, L, Dout, Din):
            x = torch.randn(B, L, Din, device=self.device, dtype=torch.bfloat16)
            w = torch.randint(-8, 7, (Din, Dout), device=self.device, dtype=torch.int32)
            return (x, w)

        # LLama-2 shapes w/ 8-way tensor parallelism.
        name_to_shapes_70b = {
            "attn.wqkv": (8192, 1280),
            "attn.w0": (1024, 8192),
            "ffn.w13": (8192, 7168),
            "ffn.w2": (3584, 8192),
        }
        for seq_len in (1, 4096):
            for bsz in (1, 4, 16, 64):
                for name, (k, n) in name_to_shapes_70b.items():
                    yield args(bsz, seq_len, n, k)

    def get_x_val(self, example_inputs) -> float:
        x, w = example_inputs
        B, m, k = x.size()
        _, n = w.size()
        return (B, m, n, k)

    @register_benchmark(baseline=True)
    def tinygemm(self, x, w):
        x = x.reshape(-1, x.size(-1))
        w_bf16 = w.to(torch.bfloat16)
        w_uint8, scales_and_zeros = _group_quantize_tensor(
            w_bf16, n_bit=4, q_group_size=self.group_size
        )
        w_int4 = torch.ops.aten._convert_weight_to_int4pack(
            w_uint8, self.inner_k_tiles
        )
        return lambda: torch.ops.aten._weight_int4pack_mm(
            x, w_int4, self.group_size, scales_and_zeros
        )

    @register_benchmark()
    def triton(self, x, w):
        x = x.reshape(-1, x.size(-1))
        w_int4 = pack_2xint4(w).T.contiguous().T
        print(w_int4.dtype)
        return lambda: matmul(x, w_int4)

    @register_metric()
    def gbps(self, fn, example_inputs: Any, metrics: BenchmarkOperatorMetrics) -> float:
        def nbytes(t):
            return t.numel() * t.element_size()

        x, w = example_inputs
        w, scales_and_zeros = _group_quantize_tensor(
            w.to(torch.bfloat16), n_bit=4, q_group_size=self.group_size
        )
        c = fn()

        gb = (sum(nbytes(t) for t in (x, scales_and_zeros, c)) + nbytes(w) // 8) / 1e9
        return gb / metrics.latency * 1e3

    @register_metric()
    def tflops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        a, b = example_inputs
        B, m, k = a.size()
        m = B * m
        _, n = b.size()
        flops = 2 * m * n * k
        return flops / metrics.latency / 1e12 * 1e3

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=[
                    "B",
                    "m",
                    "n",
                    "k",
                ],  # argument names to use as an x-axis for the plot
                x_vals=self.output.x_vals,  # different possible values for `x_name`
                line_arg="provider",  # argument name whose value corresponds to a different line in the plot
                line_vals=[
                    "tinygemm",
                    "triton",
                ],  # possible values for `line_arg``
                line_names=[
                    "tinygemm",
                    "triton",
                ],  # label name for the lines
                styles=[("blue", "-"), ("green", "-")],
                ylabel="tflops",  # label name for the y-axis
                plot_name="int4-gemm-performance",  # name for the plot. Used also as a file name for saving the plot.
                args={},  # values for function arguments not in `x_names` and `y_name`
            )
        )
        def _plot(B, m, n, k, provider):
            tflops = self.output.get_y_vals((B, m, n, k), provider, "tflops")
            return tflops

        save_path = "/tmp/int4_gemm"

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        _plot.run(show_plots=True, print_data=True, save_path=save_path)
