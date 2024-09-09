"""Full definition of a LLaMA Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
"""

# mypy: ignore-errors
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self


MaskCache = torch.Tensor
RoPECache = torch.Tensor
KVCache = Tuple[torch.Tensor, torch.Tensor]


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class LLaMAConfig:
    block_size: int = 2048
    vocab_size: int = 32000
    padded_vocab_size: Optional[int] = None
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    intermediate_size: int = None
    n_query_groups: int = -1
    head_dim: int = 64
    multiple_of: int = 64
    rope_base: float = 10000

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.multiple_of)
        if self.n_query_groups == -1:
            self.n_query_groups = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.n_embd
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.n_embd // self.n_head

    @classmethod
    def from_name(cls, name: str) -> Self:
        if name in llama2_configs:
            return cls(**llama2_configs[name])
        # fuzzy search
        config = [
            config
            for config in llama2_configs
            if config in str(name).upper() or config in str(name)
        ]
        assert len(config) == 1, name
        return cls(**llama2_configs[config[0]])


llama2_configs = {
    "1.1B": dict(
        n_layer=22, n_head=32, n_embd=2048, n_query_groups=4, intermediate_size=5632
    ),  # TinyLLaMA
    "CodeLlama-7b-Python-hf": dict(
        block_size=16384,
        vocab_size=32000,
        n_layer=32,
        n_embd=4096,
        multiple_of=16,
        rope_base=1000000,
    ),
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "34B": dict(
        n_layer=48,
        n_head=64,
        n_embd=8192,
        vocab_size=32000,
        n_query_groups=8,
        intermediate_size=22016,
        rope_base=1000000,
    ),  # CodeLLama-34B-Python-hf
    "70B": dict(
        n_layer=80, n_head=64, n_embd=8192, n_query_groups=8, intermediate_size=28672
    ),
}


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.k_cache = torch.nn.Parameter(torch.zeros(cache_shape, dtype=dtype))
        self.v_cache = torch.nn.Parameter(torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        self.k_cache[:, :, input_pos] = k_val
        self.v_cache[:, :, input_pos] = v_val

        return self.k_cache, self.v_cache


class KVCacheAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.kv_caches = nn.ModuleList([])

    def initialize(
        self,
        layers,
        max_batch_size,
        max_seq_length,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
    ):
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.kv_caches = nn.ModuleList(
            [
                KVCache(max_batch_size, max_seq_length, n_heads, head_dim)
                for _ in range(layers)
            ]
        )

    def __getitem__(self, idx):
        return self.kv_caches[idx]

    def clear(self):
        self.kv_caches = nn.ParameterList([])


class LLaMA(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=RMSNorm(config.n_embd),
            )
        )

        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[MaskCache] = None
        self.kv_caches = KVCacheAggregator()
        self.max_batch_size = None
        self.max_seq_length = None

    def setup_caches(self, max_batch_size, max_seq_length, dtype=torch.bfloat16):
        if (
            self.max_seq_length == max_seq_length
            and self.max_batch_size == max_batch_size
        ):
            return
        head_dim = self.config.n_embd // self.config.n_head
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        self.kv_caches.initialize(
            layers=self.config.n_layer,
            max_batch_size=max_batch_size,
            max_seq_length=max_seq_length,
            n_heads=self.config.n_query_groups,
            head_dim=head_dim,
        )

        self.rope_cache = build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=self.config.n_embd // self.config.n_head,
            dtype=dtype,
            base=self.config.rope_base,
        )
        ones = torch.ones(
            (self.config.block_size, self.config.block_size), dtype=torch.bool
        )
        self.mask_cache = torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )

    @torch.no_grad()
    def forward(
        self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[KVCache]]]:
        B, T = idx.size()
        assert self.rope_cache is not None, "Caches must be initialized first"

        block_size = self.config.block_size
        max_seq_length = self.max_seq_length
        if max_seq_length is None:
            max_seq_length = block_size

        assert (
            T <= max_seq_length
        ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert (
            max_seq_length <= block_size
        ), f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert (
            T <= block_size
        ), f"Cannot forward sequence of length {T}, block size is only {block_size}"

        rope = self.rope_cache.index_select(0, input_pos)
        mask = self.mask_cache.index_select(2, input_pos)
        mask = mask[:, :, :, :max_seq_length]

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        for i, block in enumerate(self.transformer.h):
            x, new_kv_cache = block(
                x, rope, mask, max_seq_length, input_pos, self.kv_caches[i]
            )

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(LLaMAConfig.from_name(name))


class Block(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        h, new_kv_cache = self.attn(
            self.norm_1(x), rope, mask, max_seq_length, input_pos, kv_cache
        )
        x = x + h
        x = x + self.mlp(self.norm_2(x))
        return x, new_kv_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_query_groups) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, total_head_dim, bias=False)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_query_groups = config.n_query_groups
        self.n_embd = config.n_embd

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, _ = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        kv_size = self.n_query_groups * self.head_dim
        q, k, v = self.attn(x).split([self.n_embd, kv_size, kv_size], dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_query_groups, self.head_dim)
        v = v.view(B, T, self.n_query_groups, self.head_dim)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            k, v = kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_query_groups, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_query_groups, dim=1)
        # todo(chilli): currently doesn't work with GQA
        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y, kv_cache


class MLP(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()

        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc_1(x)) * self.fc_2(x)
        x = self.proj(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, base: int = 10000
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache


def apply_rope(x: torch.Tensor, rope_cache: RoPECache) -> torch.Tensor:
    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because the reference does
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
