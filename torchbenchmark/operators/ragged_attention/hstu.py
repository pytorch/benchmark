import torch
import triton
from torchbenchmark import add_path, SUBMODULE_PATH

try:
    # Internal Import
    from hammer.oss.generative_recommenders.ops.triton.triton_ragged_hstu_attention import (
        _ragged_hstu_attn_fwd,
        _ragged_hstu_attn_fwd_persistent,
    )
except ModuleNotFoundError:
    # OSS Import
    import importlib

    with add_path(str(SUBMODULE_PATH)):
        triton_ragged_hstu_attention = importlib.import_module(
            "generative-recommenders.ops.triton.triton_ragged_hstu_attention"
        )
        _ragged_hstu_attn_fwd = triton_ragged_hstu_attention._ragged_hstu_attn_fwd
        _ragged_hstu_attn_fwd_persistent = (
            triton_ragged_hstu_attention._ragged_hstu_attn_fwd_persistent
        )

from typing import Tuple


class RaggedHSTUAttn(torch.nn.Module):
    def __init__(
        self,
        batch_size,
        num_heads,
        max_seq_len,
        num_buckets,
        persistent_kernel: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        super().__init__()
        self.all_ts_weights = torch.nn.Parameter(
            torch.randn(
                (self.num_buckets + 1,),
                dtype=torch.bfloat16,
            ).cuda()
        )
        self.all_pos_weights = torch.nn.Parameter(
            torch.randn(
                (2 * self.max_seq_len - 1,),
                dtype=torch.bfloat16,
            ).cuda()
        )
        self.persistent_kernel = persistent_kernel

    def forward(
        self, qkv: torch.Tensor, seq_offsets: torch.Tensor, timestamps: torch.Tensor
    ) -> torch.Tensor:
        torch._check(timestamps.size(0) + 1 == seq_offsets.size(0))

        q = qkv[:, :, :128]
        k = qkv[:, :, 128:256]
        v = qkv[:, :, 256:384]
        out = torch.zeros_like(v)

        Z = timestamps.size(0)
        N = timestamps.size(1) - 1
        _, H, DimQ = q.shape
        _, _, DimV = v.shape

        kwargs = {
            "Q": q,
            "K": k,
            "V": v,
            "seq_offsets": seq_offsets,
            "delta_x_offsets": None,
            "TS": timestamps,
            "TW": self.all_ts_weights,
            "PW": self.all_pos_weights,
            "Bias": None,
            "seq2_offsets": None,
            "num_targets": None,
            "Scale": None,
            "Out": out,
            "stride_qm": q.stride(0),
            "stride_qh": q.stride(1),
            "stride_kn": k.stride(0),
            "stride_kh": k.stride(1),
            "stride_vn": v.stride(0),
            "stride_vh": v.stride(1),
            "stride_sz": None,
            "stride_sm": None,
            "stride_ts": timestamps.stride(0),
            "stride_om": out.stride(0),
            "stride_oh": out.stride(1),
            "alpha": 0.08838834764831843,
            "Z": Z,
            "H": H,
            "MAX_SEQ_LEN": N,
            "DimQ": DimQ,
            "DimV": DimV,
            "DeltaSize": None,
            "num_buckets": self.num_buckets,
            "max_pos_ind": None,
            "time_bucket_incr": 60.0,
            "time_bucket_div": 1.0,
            "time_delta": 0.0,
            "INVALID_MASK_TYPE": "lower_triangular",
            "CAUSAL": True,
            "BUCKET_FN": "sqrt",
            "ATTN_BIAS_TYPE": "fused",
            "USE_TIME_BIAS": False,
            "USE_POS_BIAS": False,
            "HAS_MAX_POS_IND": False,
            "HAS_MULTIPLE_TARGETS": False,
            "HAS_ATTN_SCALE": False,
            "IS_DELTA_Q": False,
            "ALLOW_TF32": True,
            "BLOCK_D_Q": DimQ,
            "BLOCK_D_V": DimV,
            "max_attn_len": 0,
            "HAS_MAX_ATTN_LEN": False,
            "sort_by_length_indices": False,
            "AUTOTUNE_MAX_SEQ_LEN": N,
            "contextual_seq_len": 0,
            "HAS_CONTEXTUAL_SEQ_LEN": False,
            "HAS_SORT_BY_LENGTH_INDICES": False,
        }
        if self.persistent_kernel:
            grid = (1216,)
            # pyre-fixme[16]: Module `triton_ragged_hstu_attention` has no attribute
            _ragged_hstu_attn_fwd_persistent[grid](**kwargs)
        else:
            grid = lambda meta: (  # noqa E731
                triton.cdiv(N, meta["BLOCK_M"]),
                Z * H,
            )
            # pyre-fixme[16]: Module `triton_ragged_hstu_attention` has no attribute
            _ragged_hstu_attn_fwd[grid](**kwargs)

        return out


def get_test_inputs(
    batch_size, num_heads, max_seq_len
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    timestamp_deltas: torch.Tensor = (
        torch.randint(
            86400,
            size=(batch_size, max_seq_len + 1),
        )
        .requires_grad_(False)
        .cuda()
    )
    timestamps = timestamp_deltas.cumsum(dim=1)

    lengths = (
        torch.randint(
            max_seq_len + 1,
            size=(batch_size,),
        )
        .requires_grad_(False)
        .cuda()
    )
    seq_offsets = (
        torch.zeros(
            (batch_size + 1,),
            dtype=torch.int64,
        )
        .requires_grad_(False)
        .cuda()
    )
    seq_offsets[1:] = torch.cumsum(
        lengths,
        dim=0,
    )
    L = int(seq_offsets[-1].item())

    qkv = (
        torch.randn(
            (L, num_heads, 512),
            dtype=torch.bfloat16,
        )
        .requires_grad_(False)
        .cuda()
    )
    return qkv, seq_offsets, timestamps
