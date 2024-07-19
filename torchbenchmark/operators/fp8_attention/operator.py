"""
Adding FP8 to FlashAttention-2
https://research.colfax-intl.com/adding-fp8-to-flashattention/
"""

import argparse
import math

from typing import Callable, Generator, List, Optional, Any, Tuple
from torchbenchmark.util.kernels.triton_fused_attention import attention as triton_attention

import torch
from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode as BenchmarkMode,
    register_benchmark,
    register_metric,
)

try:
    # colfax Flash Attention V2 on FP8 for Hopper
    torch.ops.load_library(
        "//ai_codesign/gen_ai/cutlass-kernels:fmha_forward_lib_pipeline_h128"
    )
    colfax_fmha_pipeline = torch.ops.cutlass.fmha_forward_pipeline
except (ImportError, IOError, AttributeError):
    colfax_fmha_pipeline = None


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--embedding-dim", type=int, default=2048, help="specify embedding dim, embedding dim = n_heads * head_dim")
    parser.add_argument("--d-head", type=int, default=64, help="specify head dimension")
    parser.add_argument("--causal", action="store_true", help="enable causal")
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "tflops"]
    DEFAULT_PRECISION = "fp8"

    def __init__(self, tb_args: argparse.Namespace, extra_args: Optional[List[str]]=None):
        super().__init__(tb_args, extra_args=extra_args)
        args = parse_op_args(self.extra_args)
        self.BATCH = args.batch
        self.embedding_dim = args.embedding_dim
        self.D_HEAD = args.d_head
        self.causal = args.causal
        self.sm_scale = 1.3

    def colfax_preprocess(self, q, k, v):
        # colfax expects q,k: BATCH, N_CTX, H, D_HEAD and v: BATCH, D_HEAD, H, N_CTX
        # passed-in: BATCH, H, N_CTX, D_HEAD
        q = q.permute(0,2,1,3).contiguous()
        k = k.permute(0,2,1,3).contiguous()
        v = v.permute(0,3,1,2).contiguous()
        q = q.to(torch.float8_e4m3fn)
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
        return (
            q,
            k,
            v,
        )

    @register_benchmark(enabled=bool(colfax_fmha_pipeline))
    def colfax_fmha(
        self,
        q: torch.Tensor,  # // [b, seqlen, num_heads, head_dim]
        k: torch.Tensor,  # // [b, seqlen, num_heads, head_dim]
        v: torch.Tensor,  # // [b, seqlen, num_heads, head_dim]
    ) -> Callable:
        kLog2e = float(1.4426950408889634074); # log_2(e) = M_LOG2E
        softmax_scale = 1.0 / math.sqrt(float(self.D_HEAD))
        scale = softmax_scale * kLog2e
        colfax_q, colfax_k, colfax_v = self.colfax_preprocess(q, k, v)
        return lambda: colfax_fmha_pipeline(self.N_CTX, self.N_CTX, self.BATCH, colfax_q, colfax_k, colfax_v, scale)

    def triton_preprocess(self, q, k, v):
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.permute(0, 1, 3, 2)
        v = v.to(torch.float8_e5m2)
        return (
            q,
            k,
            v,
        )

    @register_benchmark()
    def triton_flash_v2(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        triton_q, triton_k, triton_v = self.triton_preprocess(q, k, v)
        # full fp8 will be enabled if type of q,k,v is fp8
        return lambda: triton_attention(triton_q, triton_k, triton_v, False, self.sm_scale)

    def get_x_val(self, _example_inputs) -> Tuple[int, int, int, int]:
        H = self.embedding_dim // self.D_HEAD
        return (self.BATCH, self.N_CTX, H, self.D_HEAD)

    def get_input_iter(self) -> Generator:
        # The non-fp8 FA varies N_CTX and fixes other variables. Let's do the same for fp8.
        # The autotune config only depends on N_CTX in OSS Triton tutorial.
        head_dims = [64, 128, 256]
        BATCH = self.BATCH
        D_HEAD = self.D_HEAD
        requires_grad = True
        for N_CTX in [2**i for i in range(7, 15)]:
            self.N_CTX = N_CTX
            H = self.embedding_dim // D_HEAD
            
            # colfax expects q,k: BATCH, N_CTX, H, D_HEAD and v: BATCH, D_HEAD, H, N_CTX
            q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=torch.float16, device=self.device, requires_grad=True)
            k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=torch.float16, device=self.device, requires_grad=True)
            v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=torch.float16, device=self.device, requires_grad=True)
            yield (q, k, v)

    @register_metric()
    def tflops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        H = self.embedding_dim // self.D_HEAD
        flops_per_matmul = (
            2.0 * self.BATCH * H * self.N_CTX * self.N_CTX * self.D_HEAD
        )
        tflops = 2 * flops_per_matmul
        return tflops / metrics.latency * 1e-9
