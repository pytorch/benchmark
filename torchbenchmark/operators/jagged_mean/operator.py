import argparse
import itertools
import math
import os
import random
from typing import Callable, Generator, List, Optional, Tuple

import torch
import triton
from torchbenchmark.util.jagged_utils import (
    generate_input_vals,
    generate_random_nested_tensors,
    get_parse_op_args,
)

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    dump_autotuner_best_config,
    register_benchmark,
    register_metric,
)

from .kernels import (
    triton_jagged_mean_kernel_simple_fused_buffer_then_sum,
    triton_jagged_mean_kernel_simple_fused_sum_then_buffer,
    triton_jagged_mean_kernel_variable_length_loop_buffer_then_sum,
    triton_jagged_mean_kernel_variable_length_loop_sum_then_buffer,
)


seed = 16
random.seed(seed)

GIGABYTES_PER_BYTE = 1e-6
RANDOM_CHOICE_MARGIN = 0.3
ABSOLUTE_TOLERANCE = 1e-4
RELATIVE_TOLERANCE = 1e-3
TENSOR_BYTES_LIMIT = 8 * 1e9  # allocate tensors no greater than 8GB


def parse_op_args(args: List[str]):
    parser = get_parse_op_args(
        "B", "M", "seqlen", "sparsity", "sum_then_buffer", "plot_benchmarks"
    )
    return parser.parse_args(args)


def execute_kernel_simple_fused(x, max_seqlen, sum_then_buffer):
    B, M = x.shape[0], x.shape[2]
    grid = lambda meta: ((len(x.offsets()) - 1) * triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    kernel_output = torch.zeros((B, M), device=x.device)

    if sum_then_buffer:
        triton_jagged_mean_kernel_simple_fused_sum_then_buffer[grid](
            x.values(),
            x.offsets(),
            kernel_output,
            M=M,
            MAX_SEQLEN=max_seqlen,
        )
    else:
        triton_jagged_mean_kernel_simple_fused_buffer_then_sum[grid](
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
        triton_jagged_mean_kernel_variable_length_loop_sum_then_buffer[grid](
            x.values(),
            x.offsets(),
            kernel_output,
            M=M,
        )
    else:
        triton_jagged_mean_kernel_variable_length_loop_buffer_then_sum[grid](
            x.values(),
            x.offsets(),
            kernel_output,
            M=M,
        )

    return kernel_output


class Operator(BenchmarkOperator):

    DEFAULT_METRICS = ["latency", "accuracy"]
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
    def torch_jagged_mean_unbind_torch_mean(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        return lambda: torch.cat(
            [torch.mean(t, dim=0).unsqueeze(0) for t in x.unbind()]
        )  # in 3D tensor (B, *, M), takes the mean of B 2D tensors (*, M)

    @register_benchmark()
    def torch_jagged_mean_torch_nanmean(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        return lambda: torch.nanmean(
            torch.ops.aten._jagged_to_padded_dense_forward(
                x.values(),
                [x.offsets()],  # pyre-ignore: Undefined attribute [16]: `torch._tensor.Tensor` has no attribute `offsets`.
                max_lengths=[seqlen],  # max length of ragged dimension
                padding_value=float("nan"),
            ),
            dim=1,
        )

    @register_benchmark()
    def torch_jagged_mean_torch_sum(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        return lambda: torch.sum(
            torch.ops.aten._jagged_to_padded_dense_forward(
                x.values(),
                [x.offsets()],  # pyre-ignore: Undefined attribute [16]: `torch._tensor.Tensor` has no attribute `offsets`.
                max_lengths=[seqlen],  # max length of ragged dimension
            ),
            dim=1,
        ) / x.offsets().diff().unsqueeze(1)

    @register_benchmark()
    def triton_jagged_mean_simple_fused(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        def _inner():
            return execute_kernel_simple_fused(x, seqlen, self.sum_then_buffer)

        return _inner

    @register_benchmark()
    def triton_jagged_mean_variable_length_loop(
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
            TENSOR_BYTES_LIMIT=TENSOR_BYTES_LIMIT,
            RANDOM_CHOICE_MARGIN=RANDOM_CHOICE_MARGIN,
        ):
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

    @register_metric(skip_baseline=True)
    def best_config(
        self, fn_name, example_inputs, metrics: BenchmarkOperatorMetrics
    ) -> str:
        fn_name_str = str(fn_name).split(".")[1]

        if "simple_fused" in fn_name_str:
            if self.sum_then_buffer:
                return dump_autotuner_best_config(
                    triton_jagged_mean_kernel_simple_fused_sum_then_buffer
                )
            return dump_autotuner_best_config(
                triton_jagged_mean_kernel_simple_fused_buffer_then_sum
            )
        elif "variable_length_loop" in fn_name_str:
            if self.sum_then_buffer:
                return dump_autotuner_best_config(
                    triton_jagged_mean_kernel_variable_length_loop_sum_then_buffer
                )
            return dump_autotuner_best_config(
                triton_jagged_mean_kernel_variable_length_loop_buffer_then_sum
            )
        return ""

    def plot(self):
        str_B, str_M, str_seqlen, str_sparsity = (
            f"-B-{self.B}",
            f"-M-{self.M}",
            f"-seqlen-{self.seqlen}",
            f"-sparsity-{self.sparsity}",
        )
        if self.B is None:
            x_axis = "B"
            params = str_M + str_seqlen + str_sparsity
        elif self.M is None:
            x_axis = "M"
            params = str_B + str_seqlen + str_sparsity
        elif self.seqlen is None:
            x_axis = "seqlen"
            params = str_B + str_M + str_sparsity
        else:
            x_axis = "sparsity"
            params = str_B + str_M + str_seqlen

        line_vals_all = [
            "torch_jagged_mean_unbind_torch_mean",
            "torch_jagged_mean_torch_nanmean",
            "torch_jagged_mean_torch_sum",
            "triton_jagged_mean_simple_fused",
            "triton_jagged_mean_variable_length_loop",
        ]
        line_names_all = [
            "PyTorch jagged mean, torch.mean",
            "PyTorch jagged mean, torch.nanmean",
            "PyTorch jagged mean, torch.sum",
            "Triton jagged mean, simple fused",
            "Triton jagged mean, variable length loop",
        ]
        styles_all = [
            ("blue", "-"),
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("magenta", "-"),
        ]

        if self.plot_benchmarks == "all":
            line_vals, line_names, styles = line_vals_all, line_names_all, styles_all
        elif self.plot_benchmarks == "torch":
            line_vals = line_vals_all[:3]
            line_names = line_names_all[:3]
            styles = styles_all[:3]
        else:
            line_vals = line_vals_all[3:]
            line_names = line_names_all[3:]
            styles = styles_all[3:]

        plot_name = f"jagged-mean-perf-var-{x_axis}" + params

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
            + f"/pytorch/benchmark/torchbenchmark/operators/jagged_mean/jagged_mean_performance/{plot_name}"
        )

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        _plot.run(show_plots=True, print_data=True, save_path=save_path)
