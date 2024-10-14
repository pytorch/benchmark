import argparse
from typing import Callable, Generator, List, Optional

import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP

from torchbenchmark.util.triton_op import BenchmarkOperator, register_benchmark

try:
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
except ModuleNotFoundError:
    LigerSwiGLUMLP = None

# Reference: https://github.com/linkedin/Liger-Kernel/
# blob/main/benchmark/scripts/benchmark_swiglu.py


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.B = 4
        self.hidden_size = 4096
        self.dtype = torch.bfloat16
        self.intermediate_size = 11008
        self.hidden_act = "silu"
        llama_config = LlamaConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
        )
        self.baseline_op = LlamaMLP(config=llama_config).to(self.device).to(self.dtype)
        self.liger_op = (
            LigerSwiGLUMLP(config=llama_config).to(self.device).to(self.dtype)
        )
        self.use_cuda_graphs = False

    def get_input_iter(self) -> Generator:
        for seq_len in [2**i for i in range(10, 14)]:
            x_shape = (self.B, seq_len, self.hidden_size)
            x = torch.randn(
                *x_shape, device=self.device, dtype=self.dtype, requires_grad=True
            )

            yield (x,)

    @register_benchmark(baseline=True)
    def torch_swiglu(self, input) -> Callable:
        return lambda: self.baseline_op(input)

    @register_benchmark()
    def liger_swiglu(self, input) -> Callable:
        return lambda: self.liger_op(input)

    @register_benchmark()
    def inductor_swiglu(self, input) -> Callable:
        compiled = torch.compile(self.baseline_op, dynamic=False)
        return lambda: compiled(input)

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        y = fwd_fn()
        # TODO: we should get rid of this randn_like
        return lambda: y.backward(torch.randn_like(y), retain_graph=True)
