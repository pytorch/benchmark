import argparse
import itertools
import math
import os
import random
from typing import Callable, Generator, List, Optional, Tuple

import torch
import triton
from torchbenchmark.util.jagged_utils import (
    ABSOLUTE_TOLERANCE,
    generate_input_vals,
    generate_random_nested_tensors,
    get_param_fstrings,
    get_parse_op_args,
    get_plot_args,
    get_styles,
    get_tensor_bytes_limit,
    GIGABYTES_PER_BYTE,
    RANDOM_CHOICE_MARGIN,
    RELATIVE_TOLERANCE,
)

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from .kernels import (
    triton_jagged_softmax_kernel_simple_fused_buffer_then_sum,
    triton_jagged_softmax_kernel_variable_length_loop_buffer_then_sum,
)


def execute_kernel_simple_fused(x, max_seqlen):
    B, M = x.shape[0], x.shape[2]  # logical shape (B, *, M)
    grid = lambda meta: (B * triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    kernel_output = torch.zeros_like(x.values(), device=x.device)

    triton_jagged_softmax_kernel_simple_fused_buffer_then_sum[grid](
        x.values(),
        x.offsets(),
        kernel_output,
        M=M,
        MAX_SEQLEN=max_seqlen,
    )

    return kernel_output


def execute_kernel_variable_length_loop(x):
    B, M = x.shape[0], x.shape[2]  # logical shape (B, *, M)
    grid = lambda meta: (B * triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    kernel_output = torch.zeros_like(x.values(), device=x.device)

    triton_jagged_softmax_kernel_variable_length_loop_buffer_then_sum[grid](
        x.values(),
        x.offsets(),
        kernel_output,
        M=M,
    )

    return kernel_output


def parse_op_args(args: List[str]):
    parser = get_parse_op_args("B", "M", "seqlen", "sparsity", "plot_benchmarks")
    return parser.parse_args(args)


class Operator(BenchmarkOperator):

    DEFAULT_METRICS = ["latency", "accuracy", "best_config"]
    use_cuda_graphs = (
        False  # enables GPU/CPU sync (for methods like NestedTensor unbind)
    )

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.sizes = list(range(2, 12, 4)) + list(
            range(12, 23, 3)
        )  # bias towards larger sizes, which are more representative of real-world shapes

        args = parse_op_args(self.extra_args)
        self.B = args.B
        self.M = args.M
        self.seqlen = args.seqlen
        self.sparsity = args.sparsity
        self.plot_benchmarks = args.plot_benchmarks

        self.tensor_bytes_limit = get_tensor_bytes_limit(tb_args.test_only)

    @register_benchmark(baseline=True)
    def torch_jagged_softmax_unbind_torch_softmax(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        return lambda: torch.cat(
            [
                torch.softmax(t, dim=0) for t in x.unbind()
            ],  # torch.softmax already stabilizes the input (x - max(x))
            dim=0,
        )  # in 3D tensor (B, *, M), takes the softmax of B 2D tensors (*, M)

    @register_benchmark()
    def torch_jagged_softmax_torch_sum(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        def _inner():
            padded = torch.ops.aten._jagged_to_padded_dense_forward(
                x.values(),
                [x.offsets()],  # pyre-ignore: Undefined attribute [16]: `torch._tensor.Tensor` has no attribute `offsets`.
                max_lengths=[seqlen],  # max length of ragged dimension
                padding_value=float("-inf"),  # e^-inf = 0
            )
            padded_softmax = torch.softmax(padded, dim=1)

            return torch.ops.aten._padded_dense_to_jagged_forward(
                padded_softmax,
                [x.offsets()],
                total_L=x.values().shape[
                    0
                ],  # providing this parameter helps avoid a GPU/CPU sync
            )

        return _inner

    @register_benchmark()
    def triton_jagged_softmax_simple_fused(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        def _inner():
            return execute_kernel_simple_fused(x, seqlen)

        return _inner

    @register_benchmark()
    def triton_jagged_softmax_variable_length_loop(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        def _inner():
            return execute_kernel_variable_length_loop(x)

        return _inner

    @register_benchmark()
    def torch_compile_nested_tensor_integration(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        def _inner(x: torch.Tensor):  # softmax along ragged dimension
            return torch.softmax(x, dim=x._ragged_idx)  # pyre-ignore: Undefined attribute [16]: `torch._tensor.Tensor` has no attribute `_ragged_idx`.

        torch_compile_func = torch.compile(_inner)
        return lambda: torch_compile_func(
            x
        )._values  # compare values tensor against other benchmarks

    def get_x_val(self, example_inputs):
        if self.B is None:
            return example_inputs[1]
        if self.M is None:
            return example_inputs[2]
        if self.seqlen is None:
            return example_inputs[3]
        return example_inputs[4]

    def get_x_vals(self) -> Tuple[List[int], List[int], List[int], List[float]]:
        return generate_input_vals(
            self.B, self.M, self.seqlen, self.sparsity, self.sizes
        )

    def get_input_iter(self) -> Generator:
        """
        Generate random nested tensors of shape (B, *, M), where * is the ragged dimension
        """

        B_vals, M_vals, seqlen_vals, sparsity_vals = self.get_x_vals()

        for nt, B, M, max_seqlen, sparsity in generate_random_nested_tensors(
            B_vals,
            M_vals,
            seqlen_vals,
            sparsity_vals,
            device=self.device,
            dtype=self.dtype,
            TENSOR_BYTES_LIMIT=self.tensor_bytes_limit,
            RANDOM_CHOICE_MARGIN=RANDOM_CHOICE_MARGIN,
        ):
            yield (nt, B, M, max_seqlen, sparsity)

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        return torch.allclose(
            output, baseline_output, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE
        )

    @register_metric(skip_baseline=True)
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
        x_axis, params = get_param_fstrings(self.B, self.M, self.seqlen, self.sparsity)

        line_vals_all = [
            "torch_jagged_softmax_torch_sum",
            "triton_jagged_softmax_simple_fused",
            "triton_jagged_softmax_variable_length_loop",
            "torch_compile_nested_tensor_integration",
        ]
        line_names_all = [
            "PyTorch jagged softmax, torch.sum",
            "Triton kernel jagged softmax, simple fused",
            "Triton kernel jagged softmax, variable length loop",
            "Inductor, NestedTensor integration",
        ]
        styles_all = get_styles(len(line_vals_all))

        line_vals, line_names, styles = get_plot_args(
            self.plot_benchmarks, 1, line_vals_all, line_names_all, styles_all
        )

        plot_name = f"jagged-softmax-perf-var-{x_axis}" + params

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
                plot_name=plot_name,
                args={},
            )
        )
        def _plot(x_axis, provider):
            return self.output.get_y_vals(x_axis, provider, "latency")

        save_path = (
            os.getcwd()
            + f"/pytorch/benchmark/torchbenchmark/operators/jagged_softmax/jagged_softmax_performance/{plot_name}"
        )

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        _plot.run(show_plots=True, print_data=True, save_path=save_path)
