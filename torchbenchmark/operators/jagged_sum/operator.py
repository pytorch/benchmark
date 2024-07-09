import argparse
import itertools
import math
import os
import random
from typing import Callable, Generator, List, Optional, Tuple

import torch
import triton

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from .kernels import (
    triton_jagged_sum_kernel_simple_fused_buffer_then_sum,
    triton_jagged_sum_kernel_simple_fused_sum_then_buffer,
    triton_jagged_sum_kernel_variable_length_loop_buffer_then_sum,
    triton_jagged_sum_kernel_variable_length_loop_sum_then_buffer,
)

seed = 16
random.seed(seed)
torch.manual_seed(seed)

GIGABYTES_PER_BYTE = 1e-6
RANDOM_CHOICE_MARGIN = 0.3
ABSOLUTE_TOLERANCE = 1e-4
RELATIVE_TOLERANCE = 1e-3
TENSOR_BYTES_LIMIT = 8 * 1e9  # allocate tensors no greater than 8GB


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--B",
        type=int,
        help="[Optional] Size of dimension 0 in shape (B, *, M) (integer)",
    )
    parser.add_argument(
        "--M",
        type=int,
        help="[Optional] Size of dimension 2 in shape (B, *, M) (integer)",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        help="[Optional] Maximum sequence length on ragged dimension (integer)",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        help="[Optional] Average sparsity for nested tensor (float, (0.0-1.0))",
    )
    parser.add_argument(
        "--sum-then-buffer",
        type=int,  # 1: sum then buffer, 0: buffer then sum
        default=0,
        help="[Optional] For Triton kernels, determines whether to sum individual blocks then add to a buffer or add to a buffer then sum; 1: sum then buffer, 0: buffer then sum; default 0",
    )
    parser.add_argument(
        "--plot-benchmarks",
        type=str,
        default="all",
        help="[Optional] Determines which benchmarks to plot: all, torch, triton",
    )
    return parser.parse_args(args)


def execute_kernel_simple_fused(x, max_seqlen, sum_then_buffer):
    B, M = x.shape[0], x.shape[2]
    grid = lambda meta: ((len(x.offsets()) - 1) * triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    kernel_output = torch.zeros((B, M), device=x.device)

    if sum_then_buffer:
        triton_jagged_sum_kernel_simple_fused_sum_then_buffer[grid](
            x.values(),
            x.offsets(),
            kernel_output,
            M=M,
            MAX_SEQLEN=max_seqlen,
        )
    else:
        triton_jagged_sum_kernel_simple_fused_buffer_then_sum[grid](
            x.values(),
            x.offsets(),
            kernel_output,
            M=M,
            MAX_SEQLEN=max_seqlen,
        )

    return kernel_output


def execute_kernel_variable_length_loop(x, sum_then_buffer):
    B, M = x.shape[0], x.shape[2]
    grid = lambda meta: ((len(x.offsets()) - 1) * triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    kernel_output = torch.zeros((B, M), device=x.device)

    if sum_then_buffer:
        triton_jagged_sum_kernel_variable_length_loop_sum_then_buffer[grid](
            x.values(),
            x.offsets(),
            kernel_output,
            M=M,
        )
    else:
        triton_jagged_sum_kernel_variable_length_loop_buffer_then_sum[grid](
            x.values(),
            x.offsets(),
            kernel_output,
            M=M,
        )

    return kernel_output


class Operator(BenchmarkOperator):

    DEFAULT_METRICS = ["latency", "accuracy", "best_config"]
    use_cuda_graphs = (
        False  # enables GPU/CPU sync (for methods like NestedTensor unbind)
    )

    def __init__(self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None):
        super().__init__(tb_args, extra_args)
        self.sizes = list(range(2, 12, 4)) + list(
            range(12, 23, 3)
        )  # bias towards larger sizes, which are more representative of real-world shapes

        args = parse_op_args(self.extra_args)
        self.B = args.B
        self.M = args.M
        self.seqlen = args.seqlen
        self.sparsity = args.sparsity
        self.sum_then_buffer = args.sum_then_buffer
        self.plot_benchmarks = args.plot_benchmarks

    @register_benchmark(baseline=True)
    def torch_jagged_sum_no_pad(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        return lambda: torch.tensor(
            [
                torch.sum(t, dim=0).tolist() for t in x.unbind()
            ],  # in 3D tensor (B, *, M), sums B 2D tensors (*, M)
            device=self.device,
            dtype=self.dtype,
        )

    @register_benchmark()
    def torch_jagged_sum_pad(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        return lambda: torch.sum(
            torch.ops.aten._jagged_to_padded_dense_forward(
                x.values(),
                [x.offsets()],  # pyre-ignore: Undefined attribute [16]: `torch._tensor.Tensor` has no attribute `offsets`.
                max_lengths=[seqlen],  # max length of ragged dimension
            ),
            dim=1,
        )  # sum along ragged dimension (dim == 1)

    @register_benchmark()
    def triton_jagged_sum_no_pad_simple_fused(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        def _inner():
            return execute_kernel_simple_fused(x, seqlen, self.sum_then_buffer)

        return _inner

    @register_benchmark()
    def triton_jagged_sum_no_pad_variable_length_loop(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        def _inner():
            return execute_kernel_variable_length_loop(x, self.sum_then_buffer)

        return _inner

    def get_x_val(self, example_inputs):
        if self.B is None:
            return example_inputs[1]
        if self.M is None:
            return example_inputs[2]
        if self.seqlen is None:
            return example_inputs[3]
        if self.sparsity is None:
            return example_inputs[4]

    def get_x_vals(self) -> Tuple[List[int], List[int], List[int], List[float]]:
        B_vals, M_vals, seqlen_vals, sparsity_vals = [], [], [], []

        def get_dim_vals():
            vals = []
            vals.extend([2**n for n in self.sizes])
            vals.extend(
                [
                    (n - 1) * (n + 1)
                    for n in self.sizes
                    if n - 1 > 0 and (n - 1) * (n + 1) not in vals
                ]
            )
            return vals

        if self.B is None:
            B_vals.extend(get_dim_vals())
        else:
            B_vals.extend([self.B])

        if self.M is None:
            M_vals.extend(get_dim_vals())
        else:
            M_vals.extend([self.M])

        if self.seqlen is None:
            seqlen_vals.extend(
                list(range(100, 1000, 100)) + list(range(1000, 20000, 1000))
            )
        else:
            seqlen_vals.extend([self.seqlen])

        if self.sparsity is None:
            sparsity_vals.extend([n / 10 for n in range(1, 10)])
        else:
            sparsity_vals.extend([self.sparsity])

        return B_vals, M_vals, seqlen_vals, sparsity_vals

    def get_input_iter(self) -> Generator:
        """
        Generate random nested tensors of shape (B, *, M), where * is the ragged dimension
        """

        def get_size_in_bytes(shape) -> int:
            num_elements = math.prod(shape)
            element_size = self.dtype.itemsize
            return math.floor(num_elements * element_size)

        B_vals, M_vals, seqlen_vals, sparsity_vals = self.get_x_vals()
        vals = itertools.product(B_vals, M_vals, seqlen_vals, sparsity_vals)

        for B, M, max_seqlen, sparsity in vals:
            if (
                get_size_in_bytes((B, M, max_seqlen)) < TENSOR_BYTES_LIMIT
            ):  # ensure that GPU memory is not exceeded
                tensors = []

                # greater sparsity --> shorter sequence lengths on ragged dimension
                seqlen_avg = math.floor(
                    max_seqlen * (1 - sparsity)
                )  # average sequence length across all tensors in nested tensor
                seqlen_margin = math.floor(
                    max_seqlen * RANDOM_CHOICE_MARGIN
                )  # use margin to constrain sequence lengths to range [seqlen_avg - seqlen_margin, seqlen_avg + seqlen_margin] to approximate an average sequence length, which correlates with sparsity

                for _ in range(B):
                    seqlen_randint = random.randint(
                        max(
                            seqlen_avg - seqlen_margin, 1
                        ),  # seqlen_randint must be at least 1
                        min(
                            seqlen_avg + seqlen_margin, max_seqlen
                        ),  # seqlen_randint must not exceed self.seqlen
                    )
                    tensor_2d = torch.randn(
                        (seqlen_randint, M), device=self.device, dtype=self.dtype
                    )
                    tensors.append(tensor_2d)

                nt = torch.nested.nested_tensor(
                    tensors,
                    layout=torch.jagged,
                    device=self.device,
                    dtype=self.dtype,
                )

                yield (nt, B, M, max_seqlen, sparsity)

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        return torch.allclose(
            output, baseline_output, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE
        )

    @register_metric()
    def gbps(self, fn_name, example_inputs, metrics: BenchmarkOperatorMetrics):
        return (
            example_inputs[0].element_size()
            * example_inputs[0].numel()
            / metrics.latency
            * GIGABYTES_PER_BYTE
        )

    @register_metric(x_only=True)
    def input_shape(
        self, fn_name: str, example_inputs, metrics: BenchmarkOperatorMetrics
    ):
        return (
            f"B: {example_inputs[1]}",  # B
            "*",
            f"M: {example_inputs[2]}",  # M
            f"max seqlen: {example_inputs[3]}",  # seqlen
            f"sparsity: {example_inputs[4]}",  # sparsity
        )  # return (B, '*', M, max seqlen, sparsity) for each example input

    def plot(self):
        str_B, str_M, str_seqlen, str_sparsity = f"-B-{self.B}", f"-M-{self.M}", f"-seqlen-{self.seqlen}", f"-sparsity-{self.sparsity}"
        if self.B is None:
            x_axis = "B"
            x_log = True
            params = str_M + str_seqlen + str_sparsity
        elif self.M is None:
            x_axis = "M"
            x_log = True
            params = str_B + str_seqlen + str_sparsity
        elif self.seqlen is None:
            x_axis = "seqlen"
            x_log = False
            params = str_B + str_M + str_sparsity
        else:
            x_axis = "sparsity"
            x_log = False
            params = str_B + str_M + str_seqlen

        line_vals_all = [
            "torch_jagged_sum_no_pad",
            "torch_jagged_sum_pad",
            "triton_jagged_sum_no_pad_simple_fused",
            "triton_jagged_sum_no_pad_variable_length_loop",
        ]
        line_names_all = [
            "PyTorch jagged sum, no padding",
            "PyTorch jagged sum, padding",
            "Triton kernel jagged sum, simple fused",
            "Triton kernel jagged sum, variable length loop",
        ]
        styles_all = [
            ("blue", "-"),
            ("red", "-"),
            ("green", "-"),
            ("yellow", "-"),
        ]
        if self.plot_benchmarks == "all":
            line_vals, line_names, styles = line_vals_all, line_names_all, styles_all
        elif self.plot_benchmarks == "torch":
            line_vals = line_vals_all[:2]
            line_names = line_names_all[:2]
            styles = styles_all[:2]
        else:
            line_vals = line_vals_all[2:]
            line_names = line_names_all[2:]
            styles = styles_all[2:]

        plot_name = f"jagged-sum-perf-var-{x_axis}-xlog-{x_log}" + params

        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["x_axis"],
                x_vals=self.output.x_vals,
                line_arg="provider",
                line_vals=line_vals,
                line_names=line_names,
                styles=styles,
                xlabel=x_axis,
                ylabel="latency",
                x_log=x_log,
                plot_name=plot_name,
                args={},
            )
        )
        def _plot(x_axis, provider):
            return self.output.get_y_vals(x_axis, provider, "latency")

        save_path = (
            os.getcwd()
            + f"/pytorch/benchmark/torchbenchmark/operators/jagged_sum/jagged_sum_performance/{plot_name}"
        )

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        _plot.run(show_plots=True, print_data=True, save_path=save_path)
