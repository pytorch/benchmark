from typing import Callable, Generator, List

import torch
import triton
import triton.language as tl

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode,
    register_benchmark,
    register_metric,
)

from .kernels import triton_group_gemm_fn


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "fp16"
    DEFAULT_METRICS = ["latency", "speedup", "accuracy"]
    use_cuda_graphs = False

    @register_benchmark(baseline=True)
    def torch(self, group_A, group_B):
        def _inner():
            out = []
            for a, b in zip(group_A, group_B):
                out.append(torch.matmul(a, b))
            return out

        return _inner

    @register_benchmark()
    def triton(self, group_A, group_B):
        return lambda: triton_group_gemm_fn(group_A, group_B)

    def get_input_iter(self) -> Generator:
        self.group_size = 4
        x_vals = [2**i for i in range(7, 11)]
        for N in x_vals:
            group_A = []
            group_B = []
            for i in range(self.group_size):
                A = torch.rand((N, N), device=self.device, dtype=self.dtype)
                B = torch.rand((N, N), device=self.device, dtype=self.dtype)
                group_A.append(A)
                group_B.append(B)
            yield group_A, group_B

    def get_x_val(self, example_inputs):
        N = example_inputs[0][0].shape[0]
        return N
