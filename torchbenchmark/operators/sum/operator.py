import argparse
import itertools
import math
import os
from typing import Callable, Generator, List, Optional, Tuple

import matplotlib.pyplot as plt

import torch
import triton
import triton.language as tl
from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    dump_autotuner_best_config,
    register_benchmark,
    register_metric,
)

from .kernels import (
    triton_sum_kernel_1D_result_buffer_then_sum,
    triton_sum_kernel_1D_result_sum_then_buffer,
    triton_sum_kernel_2D_result_dim_1,
    triton_sum_kernel_2D_result_dim_1_sum_then_buffer,
    triton_sum_kernel_scalar_result,
)

GIGABYTES_PER_BYTE = 1e-6
ABSOLUTE_TOLERANCE = 1e-4
RELATIVE_TOLERANCE = 1e-3
TENSOR_BYTES_LIMIT = 8 * 1e9  # allocate tensors no greater than 10GB


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dim",
        type=int,
        default=1,
        help="Number of dimensions desired in input tensor; e.g. --input-dim 2 for a 2D input tensor",
    )
    parser.add_argument(
        "--reduce-dim",
        type=int,
        default=None,  # reduce to a scalar result
        help="[Optional] Dimension on which kernel performs reduction; e.g. --reduce-dim 0",
    )
    parser.add_argument(
        "--sum-then-buffer",
        type=int,  # 1: sum then buffer, 0: buffer then sum
        default=0,
        help="[Optional] For 1D results, determines whether to sum individual blocks then add to a buffer or add to a buffer then sum; 1: sum then buffer, 0: buffer then sum; default 0",
    )
    parser.add_argument(
        "--M",
        type=int,
        help="[Optional] Size of dimension 0 in input shape (integer)",
    )
    parser.add_argument(
        "--N",
        type=int,
        help="[Optional] Size of dimension 1 in input shape, if input_dim >= 2 (integer)",
    )
    parser.add_argument(
        "--K",
        type=int,
        help="[Optional] Size of dimension 2 in input shape, if input_dim >= 3 (integer)",
    )
    return parser.parse_args(args)


# helper functions to get kernel parameters based on output dimension


def execute_kernel_scalar_result(x):
    kernel_input = x.view(-1)
    M = kernel_input.shape[0]
    BLOCK_SIZE_M = triton.next_power_of_2(
        M
    )  # race condition in cases where BLOCK_SIZE < n_elements^2
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    kernel_output = torch.zeros(
        (), device=x.device, dtype=x.dtype
    )  # scalar tensor output

    triton_sum_kernel_scalar_result[grid](
        kernel_input,
        kernel_output,
        M=M,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )

    return kernel_output


def execute_kernel_1D_result(x, reduce_dim, sum_then_buffer):
    kernel_input = x
    M, N = x.shape
    grid = lambda meta: (
        max(
            triton.cdiv(M, meta["BLOCK_SIZE_REDUCE_DIM"]),
            triton.cdiv(N, meta["BLOCK_SIZE_NON_REDUCE_DIM"]),
            triton.cdiv(M, meta["BLOCK_SIZE_NON_REDUCE_DIM"]),
            triton.cdiv(N, meta["BLOCK_SIZE_REDUCE_DIM"]),
        ),
    )
    if reduce_dim == 0:
        kernel_output = torch.empty(N, device=x.device, dtype=x.dtype)
    else:  # reduce_dim == 1
        kernel_output = torch.empty(M, device=x.device, dtype=x.dtype)

    if sum_then_buffer:
        triton_sum_kernel_1D_result_sum_then_buffer[grid](
            kernel_input,
            kernel_output,
            M=M,
            N=N,
            dim=reduce_dim,
        )
    else:
        triton_sum_kernel_1D_result_buffer_then_sum[grid](
            kernel_input,
            kernel_output,
            M=M,
            N=N,
            dim=reduce_dim,
        )

    return kernel_output


def execute_kernel_2D_result(x):
    kernel_input = x
    M, N, K = x.shape
    grid = lambda meta: (M * triton.cdiv(K, meta["BLOCK_SIZE_K"]),)
    kernel_output = torch.empty((M, K), device=x.device, dtype=x.dtype)

    triton_sum_kernel_2D_result_dim_1_sum_then_buffer[
        grid
    ](  # variable block sizes on N and K dimensions
        kernel_input,
        kernel_output,
        M=M,
        N=N,
        K=K,
    )

    return kernel_output


class Operator(BenchmarkOperator):

    DEFAULT_METRICS = ["latency", "accuracy"]

    def __init__(self, mode: str, device: str, extra_args: Optional[List[str]] = None):
        super().__init__(mode=mode, device=device, extra_args=extra_args)
        args = parse_op_args(self.extra_args)
        self.input_dim = args.input_dim
        self.reduce_dim = args.reduce_dim
        self.sum_then_buffer = args.sum_then_buffer
        self.M = args.M
        self.N = args.N if self.input_dim >= 2 else None
        self.K = args.K if self.input_dim >= 3 else None
        self.sizes = range(9, 22, 2)

    @register_benchmark()
    def triton_sum(self, x: torch.Tensor):
        assert (
            x.is_contiguous()
        ), "Existing sum Triton kernels only support contiguous tensors"

        assert (
            self.reduce_dim is None or self.reduce_dim <= 1
        ), f"Existing sum Triton kernels do not support reducing along dimension {self.reduce_dim}"

        def _inner():
            if self.reduce_dim is None or self.input_dim == 1:
                kernel_output = execute_kernel_scalar_result(x)
            elif self.input_dim == 2:
                kernel_output = execute_kernel_1D_result(
                    x, self.reduce_dim, self.sum_then_buffer
                )
            elif self.input_dim == 3:
                assert (
                    self.reduce_dim == 1
                ), f"Existing sum Triton kernels do not support reducing {self.input_dim}-D input along dimension {self.reduce_dim}"
                kernel_output = execute_kernel_2D_result(x)
            else:
                raise NotImplementedError(
                    f"Existing sum Triton kernels do not support {self.input_dim}-D inputs"
                )

            return kernel_output

        return _inner

    @register_benchmark(baseline=True)
    def torch_sum(self, x: torch.Tensor):
        return lambda: torch.sum(x, dim=self.reduce_dim)

    def get_x_val(self, example_inputs):
        if self.M is None:
            return example_inputs[0].shape[0]
        if self.N is None:
            return example_inputs[0].shape[1]
        return example_inputs[0].shape[2]

    def get_x_vals(self):
        M_vals, N_vals, K_vals = [], [], []
        M_vals_large_middle_dim, N_vals_large_middle_dim, K_vals_large_middle_dim = (
            [],
            [],
            [],
        )

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

        if self.M is None:
            M_vals.extend(get_dim_vals())
            M_vals_large_middle_dim.extend([8, 16])
        else:
            M_vals.extend([self.M])
            M_vals_large_middle_dim.extend([self.M])

        if self.N is None:
            N_vals.extend(get_dim_vals())
            N_vals_large_middle_dim.extend([2**n for n in range(12, 22, 2)])
        else:
            N_vals.extend([self.N])
            N_vals_large_middle_dim.extend([self.N])

        if self.K is None:
            K_vals.extend(get_dim_vals())
            K_vals_large_middle_dim.extend([8, 16])
        else:
            K_vals.extend([self.K])
            K_vals_large_middle_dim.extend([self.K])

        if self.input_dim == 1:
            return M_vals
        if self.input_dim == 2:
            return M_vals, N_vals
        return (
            M_vals,
            N_vals,
            K_vals,
            M_vals_large_middle_dim,
            N_vals_large_middle_dim,
            K_vals_large_middle_dim,
        )

    def get_input_iter(self) -> Generator:
        assert (
            self.input_dim <= 3
        ), f"Existing sum Triton kernels do not support input dimension {self.input_dim}"

        def get_size_in_bytes(shape) -> int:
            num_elements = math.prod(shape)
            element_size = torch.tensor([], dtype=self.dtype).element_size()
            return num_elements * element_size

        x_vals = self.get_x_vals()
        if self.input_dim == 1:
            sizes = itertools.product(x_vals)
        elif self.input_dim == 2:
            sizes = itertools.product(x_vals[0], x_vals[1])
        else:
            sizes = list(itertools.product(x_vals[0], x_vals[1], x_vals[2])) + list(
                itertools.product(x_vals[3], x_vals[4], x_vals[5])
            )  # small- to mid-range dimensions + large middle dimension

        for size in sizes:
            if get_size_in_bytes(size) < TENSOR_BYTES_LIMIT:
                input_tensor = torch.randn(
                    size,  # tuple with self.input_dim dimensions
                    device=self.device,
                    dtype=self.dtype,
                )
                yield (input_tensor,)

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        return torch.allclose(output, baseline_output, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE)

    @register_metric()
    def gbps(self, fn_name, example_inputs, metrics: BenchmarkOperatorMetrics):
        return (
            example_inputs[0].element_size()
            * example_inputs[0].numel()
            / metrics.latency
            * GIGABYTES_PER_BYTE
        )

    @register_metric(skip_baseline=True)
    def best_config(
        self, fn_name, example_inputs, metrics: BenchmarkOperatorMetrics
    ) -> str:
        if self.input_dim == 2:
            if self.sum_then_buffer:
                return dump_autotuner_best_config(
                    triton_sum_kernel_1D_result_sum_then_buffer
                )
            return dump_autotuner_best_config(
                triton_sum_kernel_1D_result_buffer_then_sum
            )
        elif self.input_dim == 3:
            return dump_autotuner_best_config(
                triton_sum_kernel_2D_result_dim_1_sum_then_buffer
            )
        else:
            return ""

    @register_metric(x_only=True)
    def input_shape(
        self, fn_name: str, example_inputs, metrics: BenchmarkOperatorMetrics
    ):
        return example_inputs[0].shape

    def plot(self):
        if self.M is None:
            variable_dim = "M"
        elif self.N is None:
            variable_dim = "N"
        else:
            variable_dim = "K"

        plot_name = f"sum-perf-var-{variable_dim}-input-{self.input_dim}-reduce-{self.reduce_dim}"

        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["dim"],
                x_vals=self.output.x_vals,
                line_arg="provider",
                line_vals=[
                    "torch_sum",
                    "triton_sum",
                ],
                line_names=[
                    "PyTorch sum",
                    "Triton kernel sum",
                ],
                styles=[
                    ("blue", "-"),
                    ("red", "-"),
                ],
                xlabel=variable_dim,
                ylabel="latency",
                plot_name=plot_name,
                args={},
            )
        )
        def _plot(dim, provider):
            return self.output.get_y_vals(dim, provider, "latency")

        save_path = (
            os.getcwd()
            + f"/pytorch/benchmark/torchbenchmark/operators/sum/sum_performance/{plot_name}"
        )

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        _plot.run(show_plots=True, print_data=True, save_path=save_path)
