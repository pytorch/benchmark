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

seed = 16
random.seed(seed)
torch.manual_seed(seed)

GIGABYTES_PER_BYTE = 1e-6
RANDOM_CHOICE_MARGIN = 0.3
ABSOLUTE_TOLERANCE = 1e-3


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seqlen",
        type=int,
        default=500,
        help="Maximum sequence length on ragged dimension (integer)",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="Average sparsity for nested tensor (float, (0.0-1.0))",
    )
    return parser.parse_args(args)


class Operator(BenchmarkOperator):

    DEFAULT_METRICS = ["latency", "accuracy"]
    use_cuda_graphs = False  # enables GPU/CPU sync (for methods like NestedTensor unbind)

    def __init__(self, mode: str, device: str, extra_args: Optional[List[str]] = None):
        super().__init__(mode=mode, device=device, extra_args=extra_args)
        self.sizes = range(4, 10, 2)

        args = parse_op_args(self.extra_args)
        self.seqlen = args.seqlen
        self.sparsity = args.sparsity

    @register_benchmark(baseline=True)
    def torch_jagged_sum_no_pad(self, x: torch.Tensor):
        return lambda: torch.tensor(
            [
                torch.sum(t, dim=0).tolist() for t in x.unbind()
            ],  # in 3D tensor (B, *, M), sums B 2D tensors (*, M)
            device=self.device,
            dtype=self.dtype,
        )

    @register_benchmark()
    def torch_jagged_sum_pad(self, x: torch.Tensor):
        return lambda: torch.sum(
            torch.ops.aten._jagged_to_padded_dense_forward(
                x.values(),
                [x.offsets()],  # pyre-ignore: Undefined attribute [16]: `torch._tensor.Tensor` has no attribute `offsets`.
                max_lengths=[self.seqlen],  # max length of ragged dimension
            ),
            dim=1,
        )  # sum along ragged dimension (dim == 1)

    def get_x_val(self, example_inputs):
        return len(example_inputs[0])

    def get_x_vals(self) -> Tuple[List[int], List[int]]:
        B_vals, M_vals = [], []

        B_vals.extend([2**n for n in self.sizes])
        B_vals.extend(
            [
                (n - 1) * (n + 1)
                for n in self.sizes
                if n - 1 > 0 and (n - 1) * (n + 1) not in B_vals
            ]
        )

        M_vals.extend([2**n for n in self.sizes])
        M_vals.extend(
            [
                (n - 1) * (n + 1)
                for n in self.sizes
                if n - 1 > 0 and (n - 1) * (n + 1) not in M_vals
            ]
        )

        return B_vals, M_vals

    def get_input_iter(self) -> Generator:
        """
        Generate random nested tensors of shape (B, *, M), where * is the ragged dimension
        """

        B_vals, M_vals = self.get_x_vals()
        B_M_vals = itertools.product(B_vals, M_vals)

        for B, M in B_M_vals:
            tensors = []

            # greater sparsity --> shorter sequence lengths on ragged dimension
            seqlen_avg = math.floor(
                self.seqlen * (1 - self.sparsity)
            )  # average sequence length across all tensors in nested tensor
            seqlen_margin = math.floor(
                self.seqlen * RANDOM_CHOICE_MARGIN
            )  # use margin to constrain sequence lengths to range [seqlen_avg - seqlen_margin, seqlen_avg + seqlen_margin] to approximate an average sequence length, which correlates with sparsity

            for _ in range(B):
                seqlen_randint = random.randint(
                    max(
                        seqlen_avg - seqlen_margin, 1
                    ),  # seqlen_randint must be at least 1
                    min(
                        seqlen_avg + seqlen_margin, self.seqlen
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

            yield (nt,)

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
    
    @register_metric(x_only=True)
    def input_shape(
        self, fn_name: str, example_inputs, metrics: BenchmarkOperatorMetrics
    ):
        return (
            example_inputs[0].shape[0],
            "*",
            example_inputs[0].shape[2],
        )  # return (B, '*', M) for each example input
