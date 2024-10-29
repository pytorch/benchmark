import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch
from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_x_val,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP


try:
    from liger_kernel.transformers.geglu import LigerGEGLUMLP
except ModuleNotFoundError:
    LigerGEGLUMLP = None

# Reference: https://github.com/linkedin/Liger-Kernel/
# blob/main/benchmark/scripts/benchmark_geglu.py


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.bsz = 8
        self.hidden_size = 4096
        self.intermediate_size = 11008
        self.hidden_act = "gelu_pytorch_tanh"
        self.llama_config = LlamaConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
        )
        self.baseline_op = LlamaMLP(self.llama_config).to(self.device).to(self.dtype)
        self.liger_op = LigerGEGLUMLP(self.llama_config).to(self.device).to(self.dtype)
        self.use_cuda_graphs = False

    def get_input_iter(self) -> Generator:
        for T in [2**i for i in range(10, 14)]:
            x_shape = (self.bsz, T, self.hidden_size)
            input = torch.randn(
                *x_shape, device=self.device, dtype=self.dtype, requires_grad=True
            )
            yield (input,)

    @register_benchmark(baseline=True)
    def torch_geglu(self, input) -> Callable:
        return lambda: self.baseline_op(input)

    @register_benchmark()
    def liger_geglu(self, input) -> Callable:
        return lambda: self.liger_op(input)

    @register_benchmark()
    def inductor_geglu(self, input) -> Callable:
        compiled = torch.compile(self.baseline_op, dynamic=False)
        return lambda: compiled(input)

    @register_x_val(label="(B, T, H)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        return (
            example_inputs[0].size(0),
            example_inputs[0].size(1),
            example_inputs[0].size(2),
        )

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        y = fwd_fn()
        do = torch.randn_like(y)
        return lambda: y.backward(do, retain_graph=True)
