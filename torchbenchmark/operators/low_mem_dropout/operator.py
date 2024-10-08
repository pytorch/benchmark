import os
from typing import Generator, List

import torch
import triton

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from .kernels import _seeded_triton_dropout, _triton_dropout


class Operator(BenchmarkOperator):
    @register_metric()
    def gbps(self, fn_name, example_inputs, metrics: BenchmarkOperatorMetrics):
        return (
            3
            * example_inputs[1].element_size()
            * example_inputs[1].numel()
            / metrics.latency
            * 1e-6
        )

    @register_metric()
    def tflops(self, fn_name: str, example_inputs, metrics: BenchmarkOperatorMetrics):
        p, a = example_inputs
        flops = 2 * len(a)
        return flops / metrics.latency

    @register_benchmark()
    def triton_dropout(self, p, x):
        output = torch.empty_like(x)
        assert x.is_contiguous()
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        x_keep = (torch.rand(size=(n_elements,)) > p).to(torch.int32).cuda()

        def _inner():
            return _triton_dropout[grid](
                x, x_keep, output, n_elements, p, BLOCK_SIZE=1024
            )

        return _inner

    @register_benchmark(baseline=True)
    def torch_dropout(self, p, x):
        def _inner():
            m = torch.nn.Dropout(p=p)
            output = m(x)
            return output

        return _inner

    @register_benchmark()
    def seeded_dropout(self, p, x):
        output = torch.empty_like(x)
        assert x.is_contiguous()
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        seed = 123

        def _inner():
            return _seeded_triton_dropout[grid](
                x, output, n_elements, p, seed, BLOCK_SIZE=1024
            )

        return _inner

    def get_x_val(self, example_inputs) -> float:
        return len(example_inputs[1])

    def get_x_vals(self) -> List[int]:
        return [2**i for i in range(5, 20, 2)]

    def get_input_iter(self) -> Generator:
        p = 0.25
        for size in self.get_x_vals():
            yield p, torch.randn(size=(size,)).cuda()
