import argparse
import itertools
import math
import random
from typing import Callable, Generator, List, Optional, Tuple

import torch
import triton

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    dump_autotuner_best_config,
    register_benchmark,
    register_metric,
)

from .kernels import (
    triton_jagged_sum_kernel_simple_fused_buffer_then_sum,
    triton_jagged_sum_kernel_simple_fused_sum_then_buffer,
)

seed = 16
random.seed(seed)
torch.manual_seed(seed)

GIGABYTES_PER_BYTE = 1e-6
RANDOM_CHOICE_MARGIN = 0.3
ABSOLUTE_TOLERANCE = 1e-3


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
        default=1,
        help="[Optional] For Triton kernels, determines whether to sum individual blocks then add to a buffer or add to a buffer then sum; 1: sum then buffer, 0: buffer then sum",
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


class Operator(BenchmarkOperator):

    DEFAULT_METRICS = ["latency", "accuracy"]
    use_cuda_graphs = (
        False  # enables GPU/CPU sync (for methods like NestedTensor unbind)
    )

    def __init__(self, mode: str, device: str, extra_args: Optional[List[str]] = None):
        super().__init__(mode=mode, device=device, extra_args=extra_args)
        self.sizes = list(range(2, 8, 2)) + list(
            range(8, 12)
        )  # bias towards larger sizes, which are more representative of real-world shapes

        args = parse_op_args(self.extra_args)
        self.B = args.B if args.B is not None else None
        self.M = args.M if args.M is not None else None
        self.seqlen = args.seqlen if args.seqlen is not None else None
        self.sparsity = args.sparsity if args.sparsity is not None else None
        self.sum_then_buffer = args.sum_then_buffer

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
    def triton_jagged_sum_no_pad(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        def _inner():
            return execute_kernel_simple_fused(x, seqlen, self.sum_then_buffer)

        return _inner

    def get_x_val(self, example_inputs):
        return len(example_inputs[0])

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
                list(range(100, 1000, 100))
                + list(range(1000, 10000, 1000))
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

        B_vals, M_vals, seqlen_vals, sparsity_vals = self.get_x_vals()
        vals = itertools.product(B_vals, M_vals, seqlen_vals, sparsity_vals)

        for B, M, seqlen, sparsity in vals:
            tensors = []

            # greater sparsity --> shorter sequence lengths on ragged dimension
            seqlen_avg = math.floor(
                seqlen * (1 - sparsity)
            )  # average sequence length across all tensors in nested tensor
            seqlen_margin = math.floor(
                seqlen * RANDOM_CHOICE_MARGIN
            )  # use margin to constrain sequence lengths to range [seqlen_avg - seqlen_margin, seqlen_avg + seqlen_margin] to approximate an average sequence length, which correlates with sparsity

            for _ in range(B):
                seqlen_randint = random.randint(
                    max(
                        seqlen_avg - seqlen_margin, 1
                    ),  # seqlen_randint must be at least 1
                    min(
                        seqlen_avg + seqlen_margin, seqlen
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

            yield (nt, B, M, seqlen, sparsity)

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

    @register_metric(x_only=True)  # TODO modify!!!!
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
        if self.sum_then_buffer:
            return dump_autotuner_best_config(
                triton_jagged_sum_kernel_simple_fused_sum_then_buffer
            )
        return dump_autotuner_best_config(
            triton_jagged_sum_kernel_simple_fused_buffer_then_sum
        )
