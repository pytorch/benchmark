# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch


def generate_qkv(
    BATCH: int,
    H: int,
    N_CTX: int,
    D_HEAD: int,
    dtype: torch.dtype,
    device: str = "cuda",
    requires_grad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(20)
    q = torch.randn(
        (BATCH, H, N_CTX, D_HEAD),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    k = torch.randn(
        (BATCH, H, N_CTX, D_HEAD),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    v = torch.randn(
        (BATCH, H, N_CTX, D_HEAD),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    return (q, k, v)


def permute_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    perm: Tuple[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_1 = torch.permute(q, perm)
    k_1 = torch.permute(k, perm)
    v_1 = torch.permute(v, perm)
    return (q_1, k_1, v_1)


def make_packed_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Make a packed qkv tensor for flash_attention:
    from 3 * (batch, num_head, seq, head_dim) -> (batch, seq, 3, num_head, head_dim)
    """
    assert (
        q.size() == k.size() == v.size()
    ), f"{q.size()=}, {k.size()=}, {v.size()=} must be equal!"
    (BATCH, H, N_CTX, D_HEAD) = q.size()
    (q_1, k_1, v_1) = permute_qkv(q, k, v, perm=(0, 2, 1, 3))
    qkv = torch.cat([q_1, k_1, v_1], dim=2)
    return torch.reshape(qkv, (BATCH, N_CTX, 3, H, D_HEAD))
