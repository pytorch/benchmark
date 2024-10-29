import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_x_val,
)
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaRotaryEmbedding,
)

try:
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
except ModuleNotFoundError:
    liger_rotary_pos_emb = None

# Reference: https://github.com/linkedin/Liger-Kernel/
# blob/main/benchmark/scripts/benchmark_rope.py


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        # they are generated later
        self.baseline_op = None
        self.liger_op = None
        self.use_cuda_graphs = False
        self.num_q_heads = 32
        self.num_kv_heads = 8

    def get_input_iter(self) -> Generator:
        hidden_size = 8192
        for seq_length in [2**i for i in range(10, 15)]:
            yield hidden_size, seq_length

        seq_length = 2048
        for hidden_size in [32 * (2**i) for i in range(4, 10, 2)]:
            yield hidden_size, seq_length

    def prepare_input(self, hidden_size, seq_length):
        head_dim = hidden_size // self.num_q_heads
        rotary_emb = LlamaRotaryEmbedding(head_dim, device=self.device)
        q = torch.randn(
            (1, seq_length, self.num_q_heads, head_dim),
            device=self.device,
            requires_grad=True,
            dtype=self.dtype,
        ).transpose(1, 2)
        k = torch.randn(
            (1, seq_length, self.num_kv_heads, head_dim),
            device=self.device,
            requires_grad=True,
            dtype=self.dtype,
        ).transpose(1, 2)
        dq, dk = torch.randn_like(
            q, device=self.device, dtype=self.dtype
        ), torch.randn_like(k, device=self.device)
        pos_ids = torch.arange(
            seq_length, device=self.device, dtype=torch.long
        ).unsqueeze(0)
        cos, sin = rotary_emb(k, pos_ids)
        # save q,k to self for grad_to_none
        self.q = q
        self.k = k
        # save dq,dk to self for backward
        self.dq = dq
        self.dk = dk
        return q, k, cos, sin, pos_ids

    @register_benchmark(baseline=True)
    def apply_rotary_pos_emb(self, hidden_size, seq_length) -> Callable:
        q, k, cos, sin, pos_ids = self.prepare_input(hidden_size, seq_length)
        return lambda: apply_rotary_pos_emb(q, k, cos, sin, pos_ids)

    @register_benchmark()
    def liger_rotary_pos_emb(self, hidden_size, seq_length) -> Callable:
        q, k, cos, sin, pos_ids = self.prepare_input(hidden_size, seq_length)
        return lambda: liger_rotary_pos_emb(q, k, cos, sin, pos_ids)

    @register_benchmark()
    def inductor_rotary_pos_emb_full_op(self, hidden_size, seq_length) -> Callable:
        q, k, cos, sin, pos_ids = self.prepare_input(hidden_size, seq_length)
        head_dim = hidden_size // self.num_q_heads
        compiled = torch.compile(
            LlamaRotaryEmbedding(head_dim, device=self.device), dynamic=False
        )
        cos, sin = compiled(k, pos_ids)
        compiled_func = torch.compile(apply_rotary_pos_emb, dynamic=False)
        return lambda: compiled_func(q, k, cos, sin, pos_ids)

    @register_x_val(label="(H, T)")
    def get_x_val(self, example_inputs) -> Tuple[int, int]:
        return (example_inputs[0], example_inputs[1])

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        q_out, k_out = fwd_fn()
        return lambda: torch.autograd.grad(
            (q_out, k_out),
            (self.q, self.k),
            (self.dq, self.dk),
            allow_unused=True,
            retain_graph=True,
        )

    def get_grad_to_none(self, args) -> List[torch.Tensor]:
        return [self.q, self.k]
