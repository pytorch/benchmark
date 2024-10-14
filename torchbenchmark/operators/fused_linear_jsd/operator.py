import argparse
from typing import Callable, Generator, List, Optional

import torch


from torchbenchmark.util.triton_op import BenchmarkOperator, register_benchmark

try:
    from liger_kernel.transformers.fused_linear_jsd import LigerFusedLinearJSD
except ModuleNotFoundError:
    LigerCrossEntropyLoss = None

# Reference: https://github.com/linkedin/Liger-Kernel/
# blob/main/benchmark/scripts/benchmark_fused_linear_jsd.py


class TorchJSD(torch.nn.Module):
    def __init__(self, beta: float = 0.5, dtype: torch.dtype = torch.float):
        super(TorchJSD, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.beta = beta
        self.dtype = dtype

    def forward(
        self,
        log_q: torch.tensor,  # input
        log_p: torch.tensor,  # target
    ):
        log_p, log_q = log_p.to(torch.float), log_q.to(torch.float)
        log_p, log_q = log_p.view(-1, log_p.size(-1)), log_q.view(-1, log_q.size(-1))
        m = torch.lerp(torch.exp(log_p), torch.exp(log_q), self.beta)
        loss = self.beta * self.kl(torch.log(m), log_p) + (1 - self.beta) * self.kl(
            torch.log(m), log_q
        )
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
        temperature: float = 1.0,
        beta: float = 0.5,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.teacher_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.jsd = TorchJSD(beta, dtype=dtype)
        self.temperature = temperature

    def forward(self, student_input, teacher_input):
        student_logits = self.student_lin(student_input)
        teacher_logits = self.teacher_lin(teacher_input)
        student_prob = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_prob = torch.log_softmax(teacher_logits / self.temperature, dim=-1)

        return self.jsd(student_prob, teacher_prob)


class LigerLMHeadJSD(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
        temperature: float = 1.0,
        beta: float = 0.5,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.teacher_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.fused_jsd = LigerFusedLinearJSD(beta, temperature)

    def forward(self, student_input, teacher_input):
        return self.fused_jsd(
            student_input,
            self.student_lin.weight,
            teacher_input,
            self.teacher_lin.weight,
        )


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        # H": 4096, "V": 128256,
        self.H = 4096
        self.V = 128256
        self.baseline_model = TorchLMHeadJSD(
            self.H, self.V, dtype=self.dtype, device=self.device
        ).to(self.device)
        self.liger_model = LigerLMHeadJSD(
            self.H, self.V, dtype=self.dtype, device=self.device
        ).to(self.device)
        self.use_cuda_graphs = False
        self.baseline_model.student_lin.weight.data = (
            self.liger_model.student_lin.weight.data
        ) = torch.rand(self.V, self.H, device=self.device, dtype=self.dtype)
        self.baseline_model.teacher_lin.weight.data = (
            self.liger_model.teacher_lin.weight.data
        ) = torch.rand(self.V, self.H, device=self.device, dtype=self.dtype)

    def get_input_iter(self) -> Generator:
        for BT in [2**i for i in range(10, 14)]:
            student_input = torch.rand(
                BT, self.H, requires_grad=True, dtype=self.dtype, device=self.device
            )
            teacher_input = torch.rand(BT, self.H, dtype=self.dtype, device=self.device)
            yield student_input, teacher_input

    @register_benchmark(baseline=True)
    def fused_linear_jsd(self, input, target) -> Callable:
        return lambda: self.baseline_model(input, target)

    @register_benchmark()
    def liger_fused_linear_jsd(self, input, target) -> Callable:
        return lambda: self.liger_model(input, target)

    @register_benchmark()
    def inductor_fused_linear_jsd(self, input, target) -> Callable:
        compiled = torch.compile(self.baseline_model, dynamic=False)
        return lambda: compiled(input, target)

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        y = fwd_fn()
        # TODO: how to pass grad_to_none=[_input]?
        return lambda: y.backward(retain_graph=True)
