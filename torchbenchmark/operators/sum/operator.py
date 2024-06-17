import argparse
import itertools
from typing import Callable, Generator, List, Optional, Tuple

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
    triton_sum_kernel_scalar_result,
)

GIGABYTES_PER_BYTE = 1e-6
ABSOLUTE_TOLERANCE = 1e-3


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
        default=1,
        help="[Optional] For 1D results, determines whether to sum individual blocks then add to a buffer or add to a buffer then sum; 1: sum then buffer, 0: buffer then sum",
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
        kernel_output = torch.empty(N, device=x.device)
    else:  # reduce_dim == 1
        kernel_output = torch.empty(M, device=x.device)

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
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    grid = lambda meta: (M * triton.cdiv(K, meta["BLOCK_SIZE_K"]),)
    kernel_output = torch.empty((M, K), device=x.device)

    triton_sum_kernel_2D_result_dim_1[grid](
        kernel_input,
        kernel_output,
        M=M,
        N=N,
        K=K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
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
        self.sizes = range(1, 11, 2)

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
        return len(example_inputs[0])

    def get_x_vals(self) -> List[int]:
        x_vals = []

        x_vals.extend([2**n for n in self.sizes])
        x_vals.extend(
            [
                (n - 1) * (n + 1)
                for n in self.sizes
                if n - 1 > 0 and (n - 1) * (n + 1) not in x_vals
            ]
        )

        return x_vals

    def get_input_iter(self) -> Generator:
        assert (
            self.input_dim <= 3
        ), f"Existing sum Triton kernels do not support input dimension {self.input_dim}"

        sizes = itertools.product(self.get_x_vals(), repeat=self.input_dim)
        for size in sizes:
            input_tensor = torch.randn(
                size,  # tuple with self.input_dim dimensions
                device=self.device,
                dtype=self.dtype,
            )
            yield (input_tensor,)

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        return torch.allclose(output, baseline_output, atol=ABSOLUTE_TOLERANCE)

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
            return dump_autotuner_best_config(triton_sum_kernel_2D_result_dim_1)
        else:
            return ""

    @register_metric(x_only=True)
    def input_shape(
        self, fn_name: str, example_inputs, metrics: BenchmarkOperatorMetrics
    ):
        return example_inputs[0].shape  # return (B, M) for example input
