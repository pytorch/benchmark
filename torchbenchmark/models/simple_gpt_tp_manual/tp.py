from typing import List

import torch
from torch import nn
from torch.distributed import _functional_collectives as funcol
from .model import LLaMA, CausalSelfAttention, MLP


LOCAL_RANK = None
LOCAL_WORLD_SIZE = None


def _get_rank() -> int:
    return LOCAL_RANK


def _get_world_size() -> int:
    return LOCAL_WORLD_SIZE


def _apply_tp_linear(
    linear: nn.Linear, style: str, weight_splits: List[int] = []
) -> None:
    rank = _get_rank()
    world_size = _get_world_size()

    # Linear's weight matrix is transposed, and is of shape
    # (linear.out_features, linear.in_features)
    dim_lookup = {"colwise": (0, "out_features"), "rowwise": (1, "in_features")}
    assert style in dim_lookup
    shard_dim, size_attr = dim_lookup[style]

    # ensure we can shard evenly
    assert getattr(linear, size_attr) % world_size == 0

    def shard(x, dim):
        assert x.size(dim=dim) % world_size == 0
        return torch.tensor_split(x, world_size, dim=dim)[rank]

    def shard_qkv(qkv, dim):
        q, k, v = qkv.split(weight_splits, dim=dim)
        q = shard(q, dim)
        k = shard(k, dim)
        v = shard(v, dim)
        return torch.cat((q, k, v))

    # shard
    if weight_splits:
        # attention
        assert len(weight_splits) == 3

        sharded_weight = shard_qkv(linear.weight, shard_dim)
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard_qkv(linear.scales, 0)
    else:
        sharded_weight = shard(linear.weight, shard_dim)
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard(linear.scales, 0)

    # overwrite
    linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
    setattr(linear, size_attr, getattr(linear, size_attr) // world_size)

    # shape info should still be synced
    assert linear.weight.shape == (linear.out_features, linear.in_features)


def _apply_tp_mlp(mlp: MLP) -> None:
    assert hasattr(mlp, "fc_1")
    assert hasattr(mlp, "fc_2")
    assert hasattr(mlp, "proj")

    _apply_tp_linear(mlp.fc_1, "colwise")
    _apply_tp_linear(mlp.fc_2, "colwise")
    _apply_tp_linear(mlp.proj, "rowwise")

    world_size = _get_world_size()
    mlp.register_forward_hook(
        lambda _module, _input, output: funcol.all_reduce(
            output, "sum", list(range(world_size))
        )
    )


def _apply_tp_attn(attn: CausalSelfAttention) -> None:
    assert hasattr(attn, "attn")
    assert hasattr(attn, "proj")

    kv_size = attn.n_query_groups * attn.head_dim
    _apply_tp_linear(attn.attn, "colwise", [attn.n_embd, kv_size, kv_size])
    _apply_tp_linear(attn.proj, "rowwise")

    # overwrite
    world_size = _get_world_size()
    attn.n_head = attn.n_head // world_size
    attn.n_embd = attn.n_embd // world_size
    attn.head_dim = attn.n_embd // attn.n_head
    attn.n_query_groups = attn.n_query_groups // world_size

    attn.register_forward_hook(
        lambda _module, _input, output: (
            funcol.all_reduce(output[0], "sum", list(range(world_size))),
            output[1],
        )
    )


def _apply_tp_llama(llama: LLaMA) -> None:
    # overwrite config before LLaMA.setup_cache is called
    world_size = _get_world_size()
    llama.config.n_head = llama.config.n_head // world_size
    llama.config.n_embd = llama.config.n_embd // world_size
    llama.config.n_query_groups = llama.config.n_query_groups // world_size


def apply_tp(model: LLaMA, rank: int, world_size: int) -> None:
    global LOCAL_RANK, LOCAL_WORLD_SIZE
    LOCAL_RANK = rank
    LOCAL_WORLD_SIZE = world_size
    assert LOCAL_RANK >= 0 and LOCAL_RANK < 8
    assert LOCAL_WORLD_SIZE > 1 and LOCAL_WORLD_SIZE <= 8

    _apply_tp_llama(model)
    for block in model.transformer.h:
        # Apply to MLP
        _apply_tp_mlp(block.mlp)
        _apply_tp_attn(block.attn)
