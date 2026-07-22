# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import ast
import copy
import functools
import linecache
import os
import sys
import tempfile
from typing import Any, Dict, List

import torch

import triton
import triton.language as tl


def get_cuda_autotune_config():
    return [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=2
        ),
        # triton.Config(
        #     {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=4, num_warps=4
        # ),
        # triton.Config(
        #     {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=4
        # ),
        # triton.Config(
        #     {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=4
        # ),
        # triton.Config(
        #     {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=4
        # ),
        # triton.Config(
        #     {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=4
        # ),
        # triton.Config(
        #     {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=5, num_warps=2
        # ),
        # triton.Config(
        #     {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=5, num_warps=2
        # ),
    ]


def get_autotune_config():
    return get_cuda_autotune_config()


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "D", "H_D"],
)
@triton.jit
def fused_ffn_kernel(
    X_ptr,
    W13_ptr,
    W2_ptr,
    Y_ptr,
    P_out_ptr,  # Output for intermediate results
    M,
    D,
    H_D,  # Note: P is not needed as a parameter since P == D
    stride_xm,
    stride_xd,
    stride_w13a,
    stride_w13b,
    stride_w2n,
    stride_w2d,  # Changed from stride_w2p to stride_w2d
    stride_ym,
    stride_yd,  # Changed from stride_yp to stride_yd
    stride_poutm,
    stride_poutn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # This will be used for both D and P dimensions
    BLOCK_K_D: tl.constexpr,  # This will be used for D dimension only
):
    # Program IDs for M dimension
    pid_m = tl.program_id(0)

    # Offsets for M
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Initialize accumulator with float32 precision
    acc = tl.zeros((BLOCK_M, BLOCK_K_D), dtype=tl.float32)

    # Loop over H_D in BLOCK_N chunks
    for start_n in range(0, H_D, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < H_D

        # Initialize partial results
        p1_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        p2_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Block pointers for W13 (for p1 and p2)
        w1t_bptr = tl.make_block_ptr(
            base=W13_ptr,
            shape=(D, H_D),
            strides=(stride_w13b, stride_w13a),
            offsets=(0, start_n),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
        w3t_bptr = tl.make_block_ptr(
            base=W13_ptr,
            shape=(D, H_D),
            strides=(stride_w13b, stride_w13a),
            offsets=(0, H_D + start_n),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )

        # Loop over K (which is equal to D) in BLOCK_K chunks
        for k in range(0, D, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D

            # Load X block
            x_bptr = tl.make_block_ptr(
                base=X_ptr,
                shape=(M, D),
                strides=(stride_xm, stride_xd),
                offsets=(pid_m * BLOCK_M, k),
                block_shape=(BLOCK_M, BLOCK_K),
                order=(1, 0),
            )
            X_block = tl.load(x_bptr, boundary_check=(0, 1), padding_option="zero")
            # X_block = tl.where(mask_m[:, None] & mask_k[None, :], X_block, 0.0).to(
            #     tl.float16
            # )

            # Load W1 and W3 blocks
            W1_block = tl.load(w1t_bptr)
            W3_block = tl.load(w3t_bptr)

            # Perform GEMM operations
            p1_block += tl.dot(X_block, W1_block)
            p2_block += tl.dot(X_block, W3_block)

            # Advance the block pointers
            w1t_bptr = tl.advance(w1t_bptr, (BLOCK_K, 0))
            w3t_bptr = tl.advance(w3t_bptr, (BLOCK_K, 0))

        # Apply SiLU activation to p1 and multiply with p2
        p_out_block = p1_block * tl.sigmoid(p1_block) * p2_block
        # p_out_block = tl.where(mask_m[:, None] & mask_n[None, :], p_out_block, 0.0)

        # Store P_out
        P_out_offs = P_out_ptr + (
            offs_m[:, None] * stride_poutm + offs_n[None, :] * stride_poutn
        )
        tl.store(
            P_out_offs,
            p_out_block.to(tl.float16),
            mask=mask_m[:, None] & mask_n[None, :],
        )

        w2_bptr = tl.make_block_ptr(
            base=W2_ptr,
            shape=(H_D, D),
            strides=(stride_w2n, stride_w2d),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, BLOCK_K_D),
            order=(0, 1),
        )
        W2_block = tl.load(w2_bptr, boundary_check=(0, 1), padding_option="zero")

        # Perform the second GEMM
        acc += tl.dot(p_out_block.to(tl.float16), W2_block)

    offs_d = tl.arange(0, BLOCK_K_D)
    mask_d = offs_d < D
    y_offs = Y_ptr + offs_m[:, None] * stride_ym + offs_d[None, :] * stride_yd
    tl.store(y_offs, acc.to(tl.float16), mask=mask_m[:, None] & mask_d[None, :])


def fused_ffn(
    x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, has_p: bool = False
):
    # x: [B_T, D]
    # w13: [H_D*2, D]
    # D = K
    # out1: [B_T, H_D]
    # w2: [H_D, P]
    # P = K
    # output: [B_T, P]
    B_T, D = x.shape
    H_D_2, D = w13.shape
    P, H_D = w2.shape
    assert D == P, f"D and P must be equal but got {D=} and {P=}"
    assert H_D_2 == 2 * H_D, f"H_D_2 must be 2 times of H_D but got {H_D_2=} and {H_D=}"

    def grid(META):
        return (triton.cdiv(B_T, META["BLOCK_M"]),)  # triton.cdiv(P, META["BLOCK_P"]))

    output = torch.empty((B_T, P), dtype=x.dtype, device=x.device)
    if has_p:
        p_out = torch.empty((B_T, H_D), dtype=x.dtype, device=x.device)
    else:
        p_out = torch.empty(1, dtype=x.dtype, device=x.device)  # Dummy tensor

    w2_t = w2.t().contiguous()

    BLOCK_K_D = D

    fused_ffn_kernel[grid](
        x,
        w13,
        w2_t,
        output,
        p_out,
        B_T,
        D,
        H_D,
        x.stride(0),
        x.stride(1),
        w13.stride(0),
        w13.stride(1),
        w2_t.stride(0),
        w2_t.stride(1),
        output.stride(0),
        output.stride(1),
        p_out.stride(0) if has_p else 0,
        p_out.stride(1) if has_p else 0,
        BLOCK_K_D=BLOCK_K_D,
    )

    return output, p_out if has_p else None


def eager_ffn(x, w13, w2):
    p = torch.matmul(x, w13.t())
    H_D_2, D = w13.shape
    H_D = H_D_2 // 2
    p1 = p[:, :H_D]  # B_T, H_D
    p2 = p[:, H_D:]  # B_T, H_D
    p_out = p1 * torch.sigmoid(p1) * p2
    out = torch.matmul(p_out, w2.t())
    return out, p_out


def nunerics_check(shape):
    B_T, H_D, D = shape
    print(f"Running numeric check for {shape}")
    x = torch.randn((B_T, D), dtype=torch.float16, device="cuda")
    w13 = torch.randn((H_D * 2, D), dtype=torch.float16, device="cuda") * 0.1
    w2 = torch.randn((D, H_D), dtype=torch.float16, device="cuda") * 0.1
    triton_out, triton_p = fused_ffn(x, w13, w2, has_p=True)
    eager_out, eager_p = eager_ffn(x, w13, w2)

    if not torch.allclose(triton_p, eager_p, atol=1e-2, rtol=1e-2):
        print("P numeric check failed")
        print(f"triton output: {triton_p.flatten()[0:10]}")
        print(f"eager output: {eager_p.flatten()[0:10]}")
    else:
        print("P numeric check passed")
    if not torch.allclose(triton_out, eager_out, atol=1e-2, rtol=1e-2):
        print("Y numeric check failed")
        print(f"triton output: {triton_out.flatten()[0:10]}")
        print(f"eager output: {eager_out.flatten()[0:10]}")
    else:
        print("Y numeric check passed")

    torch.testing.assert_close(triton_out, eager_out, atol=1e-2, rtol=1e-2)


def do_benchmark():

    D = 2048
    H_D = 8192

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=[
                "B_T",
                "H_D",
                "D",
            ],  # Argument names to use as an x-axis for the plot
            x_vals=[
                (i, H_D, D)
                for H_D, D in [(5325, 4096)]
                for i in [1024, 2048, 4096, 8192, 16384]
            ],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["eager", "fused"],
            line_names=["Eager", "Fused"],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="Latency(ms)",  # Label name for the y-axis
            plot_name="fused_ffn-benchmark",
            args={},
        )
    )

    @triton.testing.perf_report(configs)
    def benchmark(B_T, H_D, D, provider):
        # breakpoint()
        x = torch.randn((B_T, D), dtype=torch.float16, device="cuda")
        w13 = torch.randn((H_D * 2, D), dtype=torch.float16, device="cuda")
        w2 = torch.randn((D, H_D), dtype=torch.float16, device="cuda")
        quantiles = [0.5, 0.2, 0.8]
        if provider == "eager":
            return triton.testing.do_bench(
                lambda: eager_ffn(x, w13, w2), quantiles=quantiles
            )
        if provider == "fused":
            return triton.testing.do_bench(
                lambda: fused_ffn(x, w13, w2), quantiles=quantiles
            )

    benchmark.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    # B_T, H_D, D
    torch.manual_seed(0)
    nunerics_check((1024, 1024, 128))

    # do_benchmark()
