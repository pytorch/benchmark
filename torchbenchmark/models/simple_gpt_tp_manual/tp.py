import os
from typing import Optional, List

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol
from .model import LLaMA, CausalSelfAttention, MLP


def _get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

def maybe_init_dist() -> Optional[int]:
    try:
        # provided by torchrun
        rank = _get_rank()
        world_size = _get_world_size()

        if world_size < 2:
            # too few gpus to parallelize, tp is no-op
            return None
    except KeyError:
        # not run via torchrun, no-op
        return None

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank


def _apply_tp_linear(linear: nn.Linear, style: str, weight_splits: List[int] = []) -> None:
    rank = _get_rank()
    world_size = _get_world_size()

    # Linear's weight matrix is transposed, and is of shape
    # (linear.out_features, linear.in_features)
    dim_lookup = {
        "colwise": (0, "out_features"),
        "rowwise": (1, "in_features")
    }
    assert style in dim_lookup
    shard_dim, size_attr = dim_lookup[style]

    # ensure we can shard evenly
    assert getattr(linear, size_attr) % world_size == 0

    # shard
    if weight_splits:
        # attention
        assert len(weight_splits) == 3

        q, k, v = linear.weight.split(weight_splits, dim=shard_dim)

        assert q.size(dim=shard_dim) % world_size == 0
        q = torch.tensor_split(q, world_size, dim=shard_dim)[rank]
        assert k.size(dim=shard_dim) % world_size == 0
        k = torch.tensor_split(k, world_size, dim=shard_dim)[rank]
        assert v.size(dim=shard_dim) % world_size == 0
        v = torch.tensor_split(v, world_size, dim=shard_dim)[rank]

        sharded_weight = torch.cat((q,k,v))
    else:
        sharded_weight = torch.tensor_split(linear.weight, world_size, dim=shard_dim)[rank]

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
    mlp.register_forward_hook(lambda _module, _input, output: funcol.all_reduce(
        output, "sum", list(range(world_size))))


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

    attn.register_forward_hook(lambda _module, _input, output: (funcol.all_reduce(
        output[0], "sum", list(range(world_size))), output[1]))


def _apply_tp_llama(llama: LLaMA) -> None:
    # overwrite config before LLaMA.setup_cache is called
    world_size = _get_world_size()
    llama.config.n_head = llama.config.n_head // world_size
    llama.config.n_embd = llama.config.n_embd // world_size
    llama.config.n_query_groups = llama.config.n_query_groups // world_size
    

def apply_tp(model: LLaMA) -> None:
    _apply_tp_llama(model)
    for block in model.transformer.h:
        # Apply to MLP
        _apply_tp_mlp(block.mlp)
        _apply_tp_attn(block.attn)
