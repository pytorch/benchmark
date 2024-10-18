import argparse
from typing import Callable, Generator, List, Optional

import torch

from torchbenchmark.util.triton_op import BenchmarkOperator, register_benchmark

try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
except ModuleNotFoundError:
    LigerRMSNorm = None

# Reference: https://github.com/linkedin/Liger-Kernel/
# blob/main/benchmark/scripts/benchmark_rms_norm.py


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.M = 2048
        self.eps = 1e-6
        # they are generated later
        self.llama_rms_op = None
        self.liger_rms_op = None
        self.use_cuda_graphs = False

    def get_input_iter(self) -> Generator:
        for H in [2**i for i in range(10, 16)]:
            x_shape = (self.M, H)
            _input = torch.randn(x_shape, dtype=self.dtype, device=self.device)
            yield H, _input

    @register_benchmark(baseline=True)
    def llama_rms(self, H, input) -> Callable:
        self.llama_rms_op = LlamaRMSNorm(hidden_size=H, eps=self.eps).to(self.device)
        return lambda: self.llama_rms_op(input)

    @register_benchmark()
    def liger_rms(self, H, input) -> Callable:
        self.liger_rms_op = LigerRMSNorm(hidden_size=H, eps=self.eps).to(self.device)
        return lambda: self.liger_rms_op(input)

    @register_benchmark()
    def inductor_rms(self, H, input) -> Callable:
        compiled = torch.compile(self.llama_rms_op, dynamic=False)
        return lambda: compiled(input)

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        y = fwd_fn()
        do = torch.randn_like(y)
        return lambda: y.backward(do, retain_graph=True)
