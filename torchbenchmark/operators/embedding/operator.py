import argparse
from typing import Callable, Generator, List, Optional

import torch
from torch.nn import Embedding

from torchbenchmark.util.triton_op import BenchmarkOperator, register_benchmark

try:
    from liger_kernel.transformers.experimental.embedding import LigerEmbedding
except ModuleNotFoundError:
    LigerEmbedding = None

# Reference: https://github.com/linkedin/Liger-Kernel/
# blob/main/benchmark/scripts/benchmark_embedding.py


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        # they are generated later
        self.baseline_op = None
        self.liger_op = None
        self.use_cuda_graphs = False

    def get_input_iter(self) -> Generator:
        for B, T, D in [(32, 512, 768), (8, 2048, 4096)]:
            for V in [2**i for i in range(10, 18)]:
                _input = torch.randint(0, V, (B, T), device=self.device)
                yield V, D, _input

    @register_benchmark(baseline=True)
    def torch_embedding(self, V, D, input) -> Callable:
        self.baseline_op = Embedding(V, D).to(self.device).to(self.dtype)
        return lambda: self.baseline_op(input)

    @register_benchmark()
    def liger_embedding(self, V, D, input) -> Callable:
        self.liger_op = LigerEmbedding(V, D).to(self.device).to(self.dtype)
        return lambda: self.liger_op(input)

    @register_benchmark()
    def inductor_embedding(self, V, D, input) -> Callable:
        self.baseline_op = Embedding(V, D).to(self.device).to(self.dtype)
        compiled = torch.compile(self.baseline_op, dynamic=False)
        return lambda: compiled(input)

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        y = fwd_fn()
        do = torch.randn_like(y)
        return lambda: y.backward(do)
