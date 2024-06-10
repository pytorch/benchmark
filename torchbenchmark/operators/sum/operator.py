import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch
import triton
import triton.language as tl
from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from .kernels import triton_sum_kernel_scalar


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
        self.sizes = range(1, 17)

    @register_benchmark()
    def triton_sum(self, x: torch.Tensor):
        x_1d = x.view(-1)
        M = x_1d.shape[0]
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
        BLOCK_SIZE_M = triton.next_power_of_2(
            M
        )  # race condition in cases where BLOCK_SIZE < n_elements^2

        def _inner():
            output = torch.zeros(1, device=x.device, dtype=x.dtype)

            triton_sum_kernel_scalar[grid](
                x_1d,
                output,
                M=M,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
            )

            return output

        return _inner

    @register_benchmark(baseline=True)
    def torch_sum(self, x: torch.Tensor):
        result = torch.sum(x)
        return lambda: result

    def get_x_val(self, example_inputs):
        return len(example_inputs[0])

    def get_x_vals(self) -> List[int]:
        x_vals = []

        x_vals.extend([2**n for n in self.sizes])
        x_vals.extend([(n - 1) * (n + 1) for n in self.sizes if n - 1 > 0])

        return x_vals

    def get_input_iter(self) -> Generator:
        # reduce to a scalar value
        for size in self.get_x_vals():  # 1D matrix
            input_1d = torch.randn(size, device=self.device, dtype=self.dtype)
            yield (input_1d,)

        for size in self.get_x_vals():  # 2D matrix
            if size < pow(2, 8):  # ensure we don't exceed floating point limitations
                input_2d = torch.randn(
                    (size, size), device=self.device, dtype=self.dtype
                )
                yield (input_2d,)

        for size in self.get_x_vals():  # 3D matrix
            if size < pow(2, 4):  # ensure we don't exceed floating point limitations
                input_2d = torch.randn(
                    (size, size, size), device=self.device, dtype=self.dtype
                )
                yield (input_2d,)

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
            lambda ms: 3
            * example_inputs[0].element_size()
            * example_inputs[0].numel()
            / ms
            * 1e-6
        )
        return list(map(gbps, metrics.latency if metrics.latency else [0]))
