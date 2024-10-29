import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_x_val,
)

try:
    from liger_kernel.transformers.fused_linear_jsd import LigerFusedLinearJSD
except ModuleNotFoundError:
    LigerFusedLinearJSD = None

# Reference: https://github.com/linkedin/Liger-Kernel/
# blob/main/benchmark/scripts/benchmark_fused_linear_jsd.py


class TorchJSD(torch.nn.Module):
    def __init__(
        self,
        beta: float = 0.5,
        ignore_index: int = -100,
        dtype: torch.dtype = torch.float,
    ):
        super(TorchJSD, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction="none", log_target=True)
        self.beta = beta
        self.ignore_index = ignore_index
        self.dtype = dtype

    def forward(
        self,
        log_q: torch.Tensor,  # input
        log_p: torch.Tensor,  # target
        label=None,
    ):
        log_p, log_q = log_p.to(torch.float), log_q.to(torch.float)
        log_p, log_q = log_p.view(-1, log_p.size(-1)), log_q.view(-1, log_q.size(-1))
        m = torch.lerp(torch.exp(log_q), torch.exp(log_p), self.beta)
        loss = self.beta * self.kl(torch.log(m), log_p).sum(dim=-1) + (
            1 - self.beta
        ) * self.kl(torch.log(m), log_q).sum(dim=-1)

        if label is not None:
            loss = torch.where(label != self.ignore_index, loss, 0.0)
            n_non_ignore = (label != self.ignore_index).sum().item()
            if n_non_ignore == 0:
                loss = 0.0
            else:
                loss = (loss / n_non_ignore).sum()
        else:
            loss = (loss / log_q.shape[0]).sum()
        return loss.to(self.dtype)


class TorchLMHeadJSD(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based jsd loss.

    :param H: hidden size
    :param V: vocab size
    :param temperature: softmax temperature
    :param beta: jsd beta
    """

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.teacher_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.jsd = TorchJSD(beta=beta, ignore_index=ignore_index, dtype=dtype)
        self.temperature = temperature

    def forward(self, student_input, teacher_input, label=None):
        student_logits = self.student_lin(student_input)
        teacher_logits = self.teacher_lin(teacher_input)
        student_prob = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_prob = torch.log_softmax(teacher_logits / self.temperature, dim=-1)

        return self.jsd(student_prob, teacher_prob, label)


class LigerLMHeadJSD(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.teacher_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.fused_jsd = LigerFusedLinearJSD(
            jsd_beta=beta, ignore_index=ignore_index, temperature=temperature
        )

    def forward(self, student_input, teacher_input, label=None):
        return self.fused_jsd(
            student_input,
            self.student_lin.weight,
            teacher_input,
            self.teacher_lin.weight,
            label,
        )


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.H = 4096
        self.V = 128256
        self.baseline_op = TorchLMHeadJSD(
            H=self.H, V=self.V, dtype=self.dtype, device=self.device
        )
        self.liger_op = LigerLMHeadJSD(
            H=self.H, V=self.V, dtype=self.dtype, device=self.device
        )
        self.baseline_op.student_lin.weight.data = (
            self.liger_op.student_lin.weight.data
        ) = torch.rand(self.V, self.H, device=self.device, dtype=self.dtype)
        self.baseline_op.teacher_lin.weight.data = (
            self.liger_op.teacher_lin.weight.data
        ) = torch.rand(self.V, self.H, device=self.device, dtype=self.dtype)

        self.use_cuda_graphs = False

    def get_input_iter(self) -> Generator:
        for BT in [2**i for i in range(10, 14)]:
            student_input = torch.rand(
                BT, self.H, requires_grad=True, dtype=self.dtype, device=self.device
            )
            teacher_input = torch.rand(BT, self.H, dtype=self.dtype, device=self.device)
            yield student_input, teacher_input

    @register_benchmark(baseline=True)
    def torch_lm_head_jsd(self, student_input, teacher_input) -> Callable:
        return lambda: self.baseline_op(student_input, teacher_input)

    @register_benchmark()
    def liger_lm_head_jsd(self, student_input, teacher_input) -> Callable:
        return lambda: self.liger_op(student_input, teacher_input)

    @register_benchmark()
    def inductor_lm_head_jsd(self, student_input, teacher_input) -> Callable:
        compiled = torch.compile(self.baseline_op, dynamic=False)
        return lambda: compiled(student_input, teacher_input)

    @register_x_val(label="(B*T, H)")
    def get_x_val(self, example_inputs) -> Tuple[int, int]:
        return (example_inputs[0].size(0), example_inputs[0].size(1))

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        y = fwd_fn()
        return lambda: y.backward(retain_graph=True)

    def get_grad_to_none(self, args) -> List[torch.Tensor]:
        student_input = args[0]
        return [
            student_input,
            self.baseline_op.student_lin.weight,
            self.baseline_op.teacher_lin.weight,
        ]
