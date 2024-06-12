import argparse
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
    triton_sum_kernel_1D_result,
    triton_sum_kernel_2D_result_dim_1,
    triton_sum_kernel_scalar_result,
)


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reduce-dim",
        type=int,
        nargs="*",
        default=None,
        help="[Optional] Dimension(s) on which kernel performs reduction; e.g. --reduce-dim 0, --reduce-dim 0 1",
    )
    return parser.parse_args(args)


class Operator(BenchmarkOperator):

    DEFAULT_METRICS = ["latency", "accuracy"]

    def __init__(self, mode: str, device: str, extra_args: Optional[List[str]] = None):
        super().__init__(mode=mode, device=device, extra_args=extra_args)
        args = parse_op_args(self.extra_args)
        self.reduce_dim = (
            args.reduce_dim if args.reduce_dim else None
        )  # for 2D case, guaranteed to be a list with 1 integer
        self.sizes = range(1, 11)

    @register_benchmark()
    def triton_sum(self, x: torch.Tensor):
        num_output_dims = 0 if not self.reduce_dim else x.dim() - len(self.reduce_dim)
        kernel_input = x

        assert (
            x.is_contiguous()
        ), "Existing sum Triton kernels only support contiguous tensors"

        if num_output_dims == 0:
            kernel_input = x.view(-1)
            M = kernel_input.shape[0]
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
            BLOCK_SIZE_M = triton.next_power_of_2(
                M
            )  # race condition in cases where BLOCK_SIZE < n_elements^2
        elif x.dim() == 2 and num_output_dims == 1:
            M, N = x.shape
            BLOCK_SIZE_M, BLOCK_SIZE_N = triton.next_power_of_2(
                M
            ), triton.next_power_of_2(N)
            grid = lambda meta: (
                max(
                    triton.cdiv(M, meta["BLOCK_SIZE_REDUCE_DIM"]),
                    triton.cdiv(N, meta["BLOCK_SIZE_NON_REDUCE_DIM"]),
                ),
            )
        elif x.dim() == 3 and num_output_dims == 2 and self.reduce_dim[0] == 1:
            M, N, K = x.shape
            BLOCK_SIZE_N = triton.next_power_of_2(N)
            grid = lambda meta: (M * triton.cdiv(K, meta["BLOCK_SIZE_K"]),)
        else:
            raise Exception(
                f"Existing sum Triton kernels do not support input shape {x.shape} and reduction dimension(s) {self.reduce_dim}"
            )

        def _inner():
            if num_output_dims == 0:
                kernel_output = torch.zeros(
                    (), device=x.device, dtype=x.dtype
                )  # scalar tensor output

                triton_sum_kernel_scalar_result[grid](
                    kernel_input,
                    kernel_output,
                    M=M,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                )
            elif kernel_input.dim() == 2 and num_output_dims == 1:
                if self.reduce_dim[0] == 0:
                    kernel_output = torch.empty(N, device=self.device)
                    BLOCK_SIZE_REDUCE_DIM = BLOCK_SIZE_M
                elif self.reduce_dim[0] == 1:
                    kernel_output = torch.empty(M, device=self.device)
                    BLOCK_SIZE_REDUCE_DIM = BLOCK_SIZE_N
                else:
                    raise Exception(
                        f"Existing sum Triton kernels do not support reducing input with shape {kernel_input.size} along dimension(s) {self.reduce_dim}"
                    )

                triton_sum_kernel_1D_result[grid](
                    kernel_input,
                    kernel_output,
                    M=M,
                    N=N,
                    BLOCK_SIZE_REDUCE_DIM=BLOCK_SIZE_REDUCE_DIM,
                    dim=self.reduce_dim[0],
                )
            elif (
                kernel_input.dim() == 3
                and num_output_dims == 2
                and self.reduce_dim[0] == 1
            ):
                kernel_output = torch.empty((M, K), device=self.device)

                triton_sum_kernel_2D_result_dim_1[grid](
                    kernel_input,
                    kernel_output,
                    M=M,
                    N=N,
                    K=K,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
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
        if not self.reduce_dim:
            for size in self.get_x_vals():  # 1D tensor
                input_1d = torch.randn(size, device=self.device, dtype=self.dtype)
                yield (input_1d,)

        if not self.reduce_dim or (self.reduce_dim and len(self.reduce_dim) <= 2):
            for size in self.get_x_vals():  # 2D tensor
                input_2d = torch.randn(
                    (size, size), device=self.device, dtype=self.dtype
                )
                yield (input_2d,)

        if not self.reduce_dim or (
            self.reduce_dim and len(self.reduce_dim) <= 3 and 0 not in self.reduce_dim
        ):  # in current kernels, cannot reduce a 3D tensor on the 0-th dimension
            for size in self.get_x_vals():  # 3D tensor
                input_3d = torch.randn(
                    (size, size, size), device=self.device, dtype=self.dtype
                )
                yield (input_3d,)

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        return torch.allclose(output, baseline_output, atol=1e-4)

    @register_metric(skip_baseline=True)
    def input_dims(
        self, fn_name: str, example_inputs, metrics: BenchmarkOperatorMetrics
    ):
        return [ex.dim() for ex in example_inputs]

    @register_metric()
    def gbps(self, fn_name, example_inputs, metrics: BenchmarkOperatorMetrics):
        gbps = (
            lambda ms: example_inputs[0].element_size()
            * example_inputs[0].numel()
            / ms
            * 1e-6
        )
        return list(map(gbps, metrics.latency if metrics.latency else [0]))

    @register_metric(skip_baseline=True)
    def best_config(
        self, fn_name, example_inputs, metrics: BenchmarkOperatorMetrics
    ) -> str:
        if example_inputs[0].dim() == 3 and self.reduce_dim and self.reduce_dim[0] == 1:
            return dump_autotuner_best_config(triton_sum_kernel_2D_result_dim_1)
        elif self.reduce_dim and len(self.reduce_dim) < example_inputs[0].dim():
            return dump_autotuner_best_config(triton_sum_kernel_1D_result)
        else:
            return ""
