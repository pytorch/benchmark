# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This benchmark script is based on the benchmark code from:
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

It benchmarks the following FMHA kernels:

* Triton-Flash-V2: the triton version of FA-V2:

  https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

* SDPA: the torch.nn.attention version of FA-V2

* [optional] Flash-V2: the FA-V2 from //ai_codesign/gen_ai/flash_attention_v2:flash_attention_v2,
  which was imported from https://github.com/Dao-AILab/flash-attention

* [optional] Xformers: the memory-efficient attention from xformers:

  https://fburl.com/code/cuorcm9h

* [optional] Xformers-Splitk: the triton-splitk FMHA kernel from xformers:

  https://fburl.com/code/awt36vjj
  Disabled by default because it failed with some configs. Note that
  the relevant benchmark only works with causal = False at the moment.
  Known to work with "--batch=8 --n-heads=8 --xformers-splitk"
"""

import argparse
import math
import os

import torch
import triton  # @manual=//triton:triton
from torchbenchmark import add_path, SUBMODULE_PATH

try:
    with add_path(SUBMODULE_PATH.joinpath("kernels")):
        from kernels.flash_attention import attention as triton_op_FA2
    HAS_KERNELS = True
except BaseException:
    HAS_KERNELS = False

from typing import Callable, Optional

from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention as sdpa
from torchbenchmark import add_ld_library_path
from torchbenchmark.util.kernels.triton_fused_attention import (
    attention as triton_tutorial_FA2,
    attention_tma as triton_tutorial_FA2_tma,
)

# [Optional] flash_attn v2
try:
    from flash_attn.flash_attn_interface import (
        flash_attn_qkvpacked_func as flash_attn_func,
    )

    from .test_fmha_utils import make_packed_qkv
except (ImportError, IOError, AttributeError):
    pass

HAS_CUDA_124 = torch.cuda.is_available() and torch.version.cuda >= "12.4"

# [Optional] flash_attn v3
HAS_FLASH_V3 = True
try:
    torch_lib_path = os.path.join(os.path.dirname(__file__), "lib")
    with add_ld_library_path(torch_lib_path):
        from flash_attn_interface import flash_attn_func as flash_attn_v3
except (ImportError, IOError, AttributeError):
    try:
        from ai_codesign.gen_ai.flash_attention_v2.hopper.flash_attn_interface import (
            flash_attn_func as flash_attn_v3,
        )
    except (ImportError, IOError, AttributeError):
        HAS_FLASH_V3 = False
        pass

# [Optional] xformers backend
try:
    import xformers  # @manual=//fair/xformers:xformers
    import xformers.ops.fmha as xformers_fmha  # @manual=//fair/xformers:xformers

    from .test_fmha_utils import permute_qkv
except (ImportError, IOError, AttributeError):
    pass

# [Optional] colfax cutlass backend
try:
    if not hasattr(torch.version, "git_version"):
        # colfax Flash Attention V2 for Hopper
        torch.ops.load_library("//ai_codesign/gen_ai/cutlass-kernels:fmha_forward_lib")
    else:
        from userbenchmark.triton.loader import load_library

        load_library("cutlass_kernels/fmha_forward_lib.so")
    colfax_cutlass_fmha = torch.ops.cutlass.fmha_forward
except (ImportError, IOError, AttributeError):
    colfax_cutlass_fmha = None

# [Optional] ThunderKittens backend
try:
    if not hasattr(torch.version, "git_version"):
        import h100_fwd as tk_fwd
        import h100_fwd_causal as tk_fwd_causal
    else:
        # causal is not supported right now
        from userbenchmark.triton.loader import load_library

        load_library("tk/tk_attn_h100_fwd.so")
        tk_fwd = torch.ops.tk
except (ImportError, IOError, AttributeError):
    tk_fwd = None
    tk_fwd_causal = None

from typing import Any, Generator, List

from torchbenchmark.util.input import input_filter

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode as BenchmarkMode,
    register_benchmark,
    register_metric,
    register_x_val,
)


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--n-heads", type=int, default=48, help="Number of heads")
    parser.add_argument("--d-head", type=int, default=64, help="specify head dimension")
    parser.add_argument("--causal", action="store_true", help="enable causal")
    parser.add_argument(
        "--xformers-splitk", action="store_true", help="benchmark xformers-split impl"
    )
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.use_cuda_graphs = False
        args = parse_op_args(self.extra_args)
        self.use_cuda_graphs = False
        self.BATCH = args.batch
        self.H = args.n_heads
        self.D_HEAD = args.d_head
        self.N_CTX = None
        self.causal = args.causal
        self.sm_scale = 1.3
        self.xformers_splitk = args.xformers_splitk

    @register_benchmark()
    def aten(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        def _inner():
            M = torch.tril(torch.ones((self.N_CTX, self.N_CTX), device=self.device))
            p = torch.matmul(q, k.transpose(2, 3)) * self.sm_scale
            if self.causal:
                p[:, :, M == 0] = float("-inf")
            p = torch.softmax(p.float(), dim=-1).to(q.dtype)
            # p = torch.exp(p)
            ref_out = torch.matmul(p, v)
            return ref_out

        return _inner

    @register_benchmark(baseline=True)
    def sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        def sdpa_flash_attention(q, k, v):
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                return sdpa(
                    q,
                    k,
                    v,
                    is_causal=self.causal,
                    scale=self.sm_scale,
                )

        return lambda: sdpa_flash_attention(
            q,
            k,
            v,
        )

    @register_benchmark()
    def flash_v2(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        qkv = make_packed_qkv(q, k, v)
        fn = lambda: flash_attn_func(
            qkv, softmax_scale=self.sm_scale, causal=self.causal
        )
        return fn

    @register_benchmark(enabled=HAS_FLASH_V3)
    def flash_v3(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        # [B, H, S, D] -> [B, S, H, D]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        fn = lambda: flash_attn_v3(q, k, v, self.sm_scale, self.causal)
        return fn

    @register_benchmark()
    def triton_tutorial_flash_v2(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        return lambda: triton_tutorial_FA2(q, k, v, self.causal, self.sm_scale)

    @register_benchmark(enabled=HAS_CUDA_124)
    def triton_tutorial_flash_v2_tma(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        return lambda: triton_tutorial_FA2_tma(q, k, v, self.causal, self.sm_scale)

    @register_benchmark(enabled=HAS_KERNELS)
    def triton_op_flash_v2(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        return lambda: triton_op_FA2(q, k, v, self.causal, self.sm_scale)

    # Note that we hit "CUDA error: an illegal memory access was encountered"
    # for quite a few configs. It was known to work with, e.g.
    # --batch 1 --n-heads 4 --d-head 64
    def triton_op_flash_seq_v2(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        sequence_parallel = True
        return lambda: triton_op_FA2(
            q, k, v, self.causal, self.sm_scale, sequence_parallel
        )

    def xformers_preprocess(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        q_1, k_1, v_1 = permute_qkv(q, k, v, perm=(0, 2, 1, 3))
        attn_bias = xformers.ops.LowerTriangularMask() if self.causal else None
        fhma_input = xformers_fmha.Inputs(
            query=q_1, key=k_1, value=v_1, attn_bias=attn_bias, scale=self.sm_scale
        )
        return fhma_input

    @register_benchmark(enabled=False)
    def xformers(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        fhma_input = self.xformers_preprocess(q, k, v)
        xformers_cutlass_fhma = xformers.ops.fmha.cutlass.FwOp
        return lambda: xformers_cutlass_fhma().apply(fhma_input, needs_gradient=False)

    @register_benchmark(enabled=False)
    def xformers_splitk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        fhma_input = self.xformers_preprocess(q, k, v)
        xformers_splitk_fhma = xformers_fmha.triton_splitk.FwOp
        return lambda: xformers_splitk_fhma().apply(fhma_input, needs_gradient=False)

    def colfax_cutlass_preprocess(self, q, k, v):
        return (
            torch.transpose(q, 1, 2),
            torch.transpose(k, 1, 2),
            torch.transpose(v, 1, 2),
        )

    @register_benchmark(enabled=False)
    def colfax_cutlass(self, q, k, v):
        default_scale = 1.0 / math.sqrt(float(self.D_HEAD))
        colfax_q, colfax_k, colfax_v = self.colfax_cutlass_preprocess(q, k, v)
        return lambda: colfax_cutlass_fmha(
            self.N_CTX,
            self.N_CTX,
            self.BATCH,
            colfax_q,
            colfax_k,
            colfax_v,
            default_scale,
        )

    @register_benchmark(enabled=False)
    def tk(self, q, k, v):
        o = torch.zeros_like(v)

        def tk_dispatcher():
            if self.causal:
                tk_fwd_causal.attention_forward_causal(q, k, v, o)
            else:
                tk_fwd.attention_forward(q, k, v, o)
            return o

        return tk_dispatcher

    @register_benchmark(enabled=False, label=f"cudnn_{torch.backends.cudnn.version()}")
    def cudnn(self, q, k, v):
        os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

        def sdpa_flash_attention(q, k, v):
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                return sdpa(
                    q,
                    k,
                    v,
                    is_causal=self.causal,
                    scale=self.sm_scale,
                )

        return lambda: sdpa_flash_attention(
            q,
            k,
            v,
        )

    @register_benchmark()
    def flex_attention(self, q, k, v):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        flex_attention = torch.compile(flex_attention, dynamic=False)

        if self.causal:
            B, H, S, D = q.shape
            block_mask = create_block_mask(
                causal_mask, B=None, H=None, Q_LEN=S, KV_LEN=S
            )
        else:
            block_mask = None

        return lambda: flex_attention(q, k, v, block_mask=block_mask)

    @register_metric()
    def tflops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        q, k, v = example_inputs
        BATCH, H, N_CTX, D_HEAD = q.shape
        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
        tflops = 2 * flops_per_matmul
        if self.causal:
            tflops *= 0.5
        if self.mode == BenchmarkMode.BWD:
            tflops *= 2.5  # 2.0(bwd) + 0.5(recompute)
        elif self.mode == BenchmarkMode.FWD_BWD:
            tflops *= 3.5  # 1.0(fwd) + 2.0(bwd) + 0.5(recompute)
        return tflops / metrics.latency * 1e-9

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        o = fwd_fn()
        o_tensor = input_filter(
            lambda x: isinstance(x, torch.Tensor),
            o,
        )
        do = torch.rand_like(o_tensor)
        fn = lambda: o_tensor.backward(do, retain_graph=True)
        return fn

    def get_input_iter(self) -> Generator:
        D_HEAD = self.D_HEAD
        ctx_vals = [2**i for i in range(9, 15)]
        requires_grad = True
        for N_CTX in ctx_vals:
            BATCH = 16384 // N_CTX
            H = 2048 // D_HEAD
            q = torch.randn(
                (BATCH, H, N_CTX, D_HEAD),
                dtype=self.dtype,
                device=self.device,
                requires_grad=requires_grad,
            )
            k = torch.randn(
                (BATCH, H, N_CTX, D_HEAD),
                dtype=self.dtype,
                device=self.device,
                requires_grad=requires_grad,
            )
            v = torch.randn(
                (BATCH, H, N_CTX, D_HEAD),
                dtype=self.dtype,
                device=self.device,
                requires_grad=requires_grad,
            )
            self.N_CTX = N_CTX
            yield (q, k, v)
        for q, k, v in self.__llama_example_input(
            self.device, self.dtype, requires_grad
        ):
            yield (q, k, v)

    def __llama_example_input(self, device, dtype, requires_grad):
        shapes = [
            (4, 32, 19, 128),
            (4, 32, 1, 128),
            # currently we are only able to use the same shape for q, k, v but in prod q shape is (4, 32, 1, 128) here
            (4, 32, 511, 128),
        ]
        for shape in shapes:
            yield (
                torch.randn(
                    shape,
                    dtype=dtype,
                    device=device,
                    requires_grad=requires_grad,
                ),
                torch.randn(
                    shape,
                    dtype=dtype,
                    device=device,
                    requires_grad=requires_grad,
                ),
                torch.randn(
                    shape,
                    dtype=dtype,
                    device=device,
                    requires_grad=requires_grad,
                ),
            )

    @register_x_val(label="(Batch, Heads, SeqLen, Dhead)")
    def get_x_val(self, example_inputs) -> float:
        q, k, v = example_inputs
        B, H, S, D = q.shape
        return (B, H, S, D)

    def plot(self):
        y_metric_name = "tflops"

        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["N_CTX"],  # argument names to use as an x-axis for the plot
                x_vals=self.output.x_vals,  # different possible values for `x_name`
                line_arg="provider",  # argument name whose value corresponds to a different line in the plot
                line_vals=[
                    "aten",
                    "sdpa",
                    "flash_v2",
                    "triton_tutorial_flash_v2",
                    "triton_op_flash_v2",
                    # FIXME: cuda illegal meory failure with default config
                    "triton_op_flash_seq_v2",
                    "xformers",
                    "hw_roofline",
                ],  # possible values for `line_arg``
                line_names=[
                    "ATen",
                    "SDPA",
                    "Flash V2",
                    "Triton Tutorial Flash V2",
                    "Triton Op Flash V2",
                    # FIXME: cuda illegal meory failure with default config
                    # "Triton Op Flash (Seq Parallel) V2",
                    "XFormers",
                    "Hardware Roofline",
                ],  # label name for the lines
                styles=[
                    ("blue", "-"),
                    ("yellow", "-"),
                    ("green", "-"),
                    ("red", "-"),
                    ("brown", "-"),
                    # FIXME: for "Triton Op Flash (Seq Parallel) V2", which had
                    # cuda illegal meory failure with default config
                    # ("orange", "-"),
                    ("purple", "-"),
                    ("black", "dashed"),
                ],  # line styles
                ylabel=y_metric_name,  # label name for the y-axis
                plot_name="flashattention-tflops",  # name for the plot. Used also as a file name for saving the plot.
                args={},  # values for function arguments not in `x_names` and `y_name`
            )
        )
        def _plot(N_CTX, provider):
            tflops = self.output.get_y_vals(N_CTX, provider, y_metric_name)
            return tflops

        _plot.run(
            show_plots=True, print_data=False, save_path="/tmp/test_flashattention"
        )
