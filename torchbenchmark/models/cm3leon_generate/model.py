# Copyright (c) Meta Platforms, Inc. and affiliates.
# Portions of this code are derived from https://github.com/facebookresearch/metaseq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from torch import Tensor
from typing import Optional, Dict, Any
from tqdm import tqdm


# torch.set_float32_matmul_precision("high")


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def make_positions(tensor, padding_idx: int):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (positions is None) or (
            self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."

        # we cannot use incremental state here because we must be aware of
        # padding.

        if positions is None and self.padding_idx is not None:
            positions = make_positions(input, self.padding_idx)

        assert positions is not None

        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


def PositionalEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int,
    learned: bool = False,
    learned_sinusoidal: bool = False,
    full_megatron_init=False,
    pos_init_scalar=1.0,
    megatron_init_sigma=None,
    truncate_init=False,
):
    def _init_emb(tensor, sigma):
        if sigma <= 1e-8:  # effectively 0
            return nn.init.zeros_(tensor)
        if truncate_init:
            return nn.init.trunc_normal_(
                tensor, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
            )
        else:
            return nn.init.normal_(tensor, mean=0.0, std=sigma)

    if learned:
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        # TODO: The right place for this offset would be inside
        # LearnedPositionalEmbedding. Move this there for a cleaner implementation.
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        if full_megatron_init:
            _init_emb(m.weight, megatron_init_sigma * pos_init_scalar)
        else:
            _init_emb(m.weight, embedding_dim**-0.5 * pos_init_scalar)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    elif learned_sinusoidal:
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        with torch.no_grad():
            m.weight.copy_(
                SinusoidalPositionalEmbedding.get_embedding(
                    num_embeddings,
                    embedding_dim,
                    padding_idx,
                )
            )
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim,
            padding_idx,
            init_size=num_embeddings + padding_idx + 1,
        )
    return m


from typing import Tuple
from torch.nn import Parameter, init
import math
import uuid


def softmax(x, dim: int):
    return F.softmax(x, dim=dim, dtype=torch.float32)


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class Linear(nn.Module):
    """
    Exact same as pytorch nn.Linear but with option to initialize weight and bias directly on GPU
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        initialize_params_on_gpu: bool = False,
        dtype: torch.dtype = None,
    ) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        device = torch.cuda.current_device() if initialize_params_on_gpu else None
        if dtype is None:
            dtype = torch.float
        self.weight = Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class Dropout(nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def extra_repr(self) -> str:
        return "p={}".format(self.p)

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def init_incremental_state(self):
        self._incremental_state_id = "5"  # str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        initialize_params_on_gpu=False,
        dtype: Optional[torch.dtype] = None,
    ):
        self.init_incremental_state()
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout_module = Dropout(dropout, module_name=self.__class__.__name__)
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.self_attention = self_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )
        random_state = torch.get_rng_state()
        # random_state_cuda = torch.cuda.get_rng_state()
        self.k_proj = Linear(
            self.kdim,
            embed_dim,
            bias=bias,
            initialize_params_on_gpu=initialize_params_on_gpu,
            dtype=dtype,
        )
        self.v_proj = Linear(
            self.vdim,
            embed_dim,
            bias=bias,
            initialize_params_on_gpu=initialize_params_on_gpu,
            dtype=dtype,
        )
        self.q_proj = Linear(
            embed_dim,
            embed_dim,
            bias=bias,
            initialize_params_on_gpu=initialize_params_on_gpu,
            dtype=dtype,
        )
        self.out_proj = Linear(
            embed_dim,
            embed_dim,
            bias=bias,
            initialize_params_on_gpu=initialize_params_on_gpu,
            dtype=dtype,
        )
        torch.set_rng_state(random_state)
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if (
            incremental_state is None
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
        ):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training or self.dropout_module.apply_during_inference,
                key_padding_mask,
                False,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            # Replace any non-finite values with finite equivalents, since otherwise
            # we may get NaN when adding attn_mask or computing softmax.
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        return attn, None  # To match return type of F.multi_head_attention_forward

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)


from typing import Callable, List


class ActivationFn(nn.Module):
    def __init__(self, name, embed_dim, ffn_dim):
        super().__init__()
        self.fn = self.__get_fn(name)

    def forward(self, fc1_in, fc1_out, model_parallel: bool):
        return self.fn(fc1_out)

    def __get_fn(self, name: str) -> Callable:
        """Returns the activation function corresponding to the arg passed in the run"""

        if name == "relu":
            return F.relu
        elif name == "relu_squared":
            return relu_squared
        elif name == "gelu":
            return gelu
        elif name == "tanh":
            return torch.tanh
        elif name == "linear":
            return lambda x: x
        else:
            raise RuntimeError("--activation-fn {} not supported".format(name))


class TransformerDecoderLayer(nn.Module):
    """Pre-norm Decoder layer block.

    Note that we have found model training to require pre-norm to remain stable.

    Args:
        embed_dim (int): dimension of the model embedding
        decoder_embed_dim (int): dimension of the decoder embedding
        dropout (float): dropout probability
        decoder_attention_heads (int): number of decoder attention heads
        attention_dropout (float): dropout probability for attention weights
        decoder_ffn_embed_dim (int): dimension of the decoder feedforward network embedding
        activation_fn (str): activation function name
        add_bias_kv (bool): whether to add bias to the key and value projections
        add_zero_attn (bool): whether to add a zero attention vector for padding tokens
        disable_affine_ln (bool): whether to disable affine layer normalization
        disable_bias (bool): whether to disable bias in linear layers
        tensor_parallel_init_model_on_gpu (bool): whether to initialize model on GPU for tensor parallelism
        full_megatron_init (bool): whether to use full Megatron initialization
        megatron_init_sigma (float): sigma value for Megatron initialization
        truncate_init (bool): whether to truncate the initialization values
    """

    def __init__(
        self,
        embed_dim,
        decoder_embed_dim,
        dropout=0.1,
        decoder_attention_heads=8,
        attention_dropout=0.1,
        decoder_ffn_embed_dim=2048,
        activation_fn="relu",
        add_bias_kv=False,
        add_zero_attn=False,
        disable_affine_ln=False,
        disable_bias=False,
        tensor_parallel_init_model_on_gpu=False,
        full_megatron_init=False,
        megatron_init_sigma=0.006,
        truncate_init=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_module = Dropout(dropout, module_name=self.__class__.__name__)
        self.self_attn = self.build_self_attention(
            decoder_embed_dim,
            decoder_attention_heads,
            attention_dropout,
            add_bias_kv,
            add_zero_attn,
            tensor_parallel_init_model_on_gpu,
            disable_bias,
            megatron_init_sigma,
            truncate_init,
        )

        self.nh = decoder_attention_heads
        self.head_dim = int(decoder_embed_dim / self.nh)
        affine_ln = not disable_affine_ln

        self.self_attn_layer_norm = LayerNorm(
            decoder_embed_dim, elementwise_affine=affine_ln
        )

        self.fc1 = self.build_fc1(
            decoder_embed_dim,
            decoder_ffn_embed_dim,
            tensor_parallel_init_model_on_gpu,
            full_megatron_init,
            megatron_init_sigma,
            truncate_init,
            disable_bias,
        )

        self.activation_fn = ActivationFn(
            activation_fn,
            decoder_embed_dim,
            decoder_ffn_embed_dim,
        )

        self.fc2 = self.build_fc2(
            decoder_ffn_embed_dim,
            decoder_embed_dim,
            tensor_parallel_init_model_on_gpu,
            full_megatron_init,
            megatron_init_sigma,
            truncate_init,
            disable_bias,
        )

        self.final_layer_norm = LayerNorm(
            decoder_embed_dim, elementwise_affine=affine_ln
        )

    def build_fc1(
        self,
        input_dim,
        output_dim,
        initialize_params_on_gpu=False,
        full_megatron_init=False,
        megatron_init_sigma=0.006,
        truncate_init=False,
        disable_bias=False,
    ):
        return Linear(
            input_dim,
            output_dim,
            initialize_params_on_gpu=initialize_params_on_gpu,
            bias=not disable_bias,
        )

    def build_fc2(
        self,
        input_dim,
        output_dim,
        initialize_params_on_gpu=False,
        full_megatron_init=False,
        megatron_init_sigma=0.006,
        truncate_init=False,
        disable_bias=False,
    ):
        return Linear(
            input_dim,
            output_dim,
            initialize_params_on_gpu=initialize_params_on_gpu,
            bias=not disable_bias,
        )

    def build_self_attention(
        self,
        embed_dim,
        decoder_attention_heads,
        attention_dropout,
        add_bias_kv,
        add_zero_attn,
        tensor_parallel_init_model_on_gpu,
        disable_bias,
        megatron_init_sigma,
        truncate_init,
    ):
        return MultiheadAttention(
            embed_dim,
            decoder_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
            initialize_params_on_gpu=tensor_parallel_init_model_on_gpu,
            bias=not disable_bias,
        )

    def forward_attention(
        self,
        query,
        key,
        value,
        residual,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        x, _ = self.self_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            incremental_state=incremental_state,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        return x

    def forward(
        self,
        x,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        self_attn_mask: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.forward_attention(
            query=x,
            key=x,
            value=x,
            residual=residual,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            attn_mask=self_attn_mask,
        )
        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_fn(x, self.fc1(x), model_parallel=False)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_tokens,
        decoder_attention_heads,
        decoder_ffn_embed_dim,
        activation_fn="relu",
        dropout=0.1,
        attention_dropout=0.1,
        no_emb_dropout=False,
        share_decoder_input_output_embed=False,
        embed_dim=512,
        max_target_positions=1024,
        no_scale_embedding=False,
        decoder_learned_pos=False,
        decoder_learned_sinusoidal=False,
        full_megatron_init=False,
        pos_init_scalar=1.0,
        megatron_init_sigma=0.006,
        truncate_init=False,
        decoder_layers=6,
        self_attn_doc_sep=-1,
        initialize_params_on_gpu=False,
        dtype=torch.float32,
        add_bias_kv=False,
        add_zero_attn=False,
        disable_affine_ln=False,
        disable_bias=False,
        tensor_parallel_init_model_on_gpu=False,
    ):
        super().__init__()
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.tensor_parallel_init_model_on_gpu = tensor_parallel_init_model_on_gpu
        self.megatron_init_sigma = megatron_init_sigma
        self.full_megatron_init = full_megatron_init
        self.activation_fn = activation_fn
        self.attention_dropout = attention_dropout
        self.dropout_module = Dropout(dropout, module_name=self.__class__.__name__)
        self.dropout = dropout
        self.truncate_init = truncate_init
        if no_emb_dropout:
            self.dropout_module = None

        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.disable_affine_ln = disable_affine_ln
        self.disable_bias = disable_bias
        self.decoder_attention_heads = decoder_attention_heads
        self.share_input_output_embed = share_decoder_input_output_embed
        self.embed_dim = embed_dim
        self.padding_idx: int = embed_tokens.padding_idx
        assert self.padding_idx is not None
        self.max_target_positions = max_target_positions
        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if no_scale_embedding else math.sqrt(self.embed_dim)
        self.decoder_ffn_embed_dim = decoder_ffn_embed_dim
        # default value
        device = torch.cuda.current_device() if initialize_params_on_gpu else None
        # default value

        self.self_attn_doc_sep = self_attn_doc_sep

        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                self.embed_dim,
                self.padding_idx,
                learned=decoder_learned_pos,
                learned_sinusoidal=decoder_learned_sinusoidal,
                full_megatron_init=full_megatron_init,
                pos_init_scalar=pos_init_scalar,
                megatron_init_sigma=megatron_init_sigma,
                truncate_init=truncate_init,
            )
            if decoder_learned_pos
            else None
        )
        self.embed_positions.to(device).to(dtype)

        self.layers = nn.ModuleList([])
        layers = []
        for i in range(decoder_layers):
            layers.append(self.build_decoder_layer())

        self.layers = nn.ModuleList(layers)

        self.num_layers = len(self.layers)

        self.layer_norm = LayerNorm(
            self.embed_dim,
            elementwise_affine=not disable_affine_ln,
        )
        self.layer_norm.to(device).to(dtype)

        self.output_projection = None
        if self.share_input_output_embed:
            self.output_projection = Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
                initialize_params_on_gpu=initialize_params_on_gpu,
                dtype=dtype,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = Linear(
                self.embed_dim,
                len(dictionary),
                bias=False,
                initialize_params_on_gpu=initialize_params_on_gpu,
                dtype=dtype,
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.embed_dim**-0.5
            )

    def build_base_decoder_layer(self):
        return TransformerDecoderLayer(
            self.embed_dim,
            self.embed_dim,
            self.dropout,
            self.decoder_attention_heads,
            self.attention_dropout,
            self.decoder_ffn_embed_dim,
            self.activation_fn,
            self.add_bias_kv,
            self.add_zero_attn,
            self.disable_affine_ln,
            self.disable_bias,
            self.tensor_parallel_init_model_on_gpu,
            self.full_megatron_init,
            self.megatron_init_sigma,
            self.truncate_init,
        )

    def build_decoder_layer(self):
        layer = self.build_base_decoder_layer()
        return layer

    def forward_embedding(
        self,
        tokens,
        token_embedding: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        # embed tokens and positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                tokens, incremental_state=incremental_state, positions=positions
            )
        # see BaseDecoder for important information about
        # incremental state
        if incremental_state is not None:
            tokens = tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        if token_embedding is None:
            token_embedding = self.embed_tokens(tokens)

        x = embed = self.embed_scale * token_embedding
        if positions is not None:
            x += positions

        if self.dropout_module is not None:
            x = self.dropout_module(x)

        # Returning in T x B x C format as that makes integrating sequence parallelism easier.
        x = x.transpose(0, 1).contiguous()
        return x, embed, positions

    # forward for TransformerDecoder
    def forward(
        self,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        src_lengths: Optional[Any] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings
            self_attn_padding_mask (torch.Tensor, optional): precomputed padding
                mask for self-attention (default None will recompute mask)

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        # see BaseDecoder for important information about
        # incremental state
        x = self.extract_features(
            prev_output_tokens,
            incremental_state=incremental_state,
            token_embeddings=token_embeddings,
            self_attn_padding_mask=self_attn_padding_mask,
        )
        if not features_only:
            x = self.output_layer(x)

        # Transposing back to B x T x C, so that the interface stays the same.
        x = x.transpose(0, 1).contiguous()
        return x

    def extract_features(
        self,
        prev_output_tokens: torch.Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
    ) -> torch.Tensor:
        # compute self-attention padding mask (involves device-to-host transfer,
        # so put it at the top of the forward)
        assert prev_output_tokens is not None
        assert self.padding_idx is not None
        if (
            self_attn_padding_mask is None
            and prev_output_tokens.eq(self.padding_idx).any()
        ):
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
        # assert self_attn_padding_mask is not None

        # embed tokens and positions
        # x is T x B x C
        x, tok, pos = self.forward_embedding(
            prev_output_tokens, token_embeddings, incremental_state
        )

        # see BaseDecoder for important information about
        # incremental state. Note that it may be an empty dictionary.
        if incremental_state is not None:
            self_attn_mask = self.buffered_future_mask(x, prev_output_tokens)
        else:
            self_attn_mask = None

        # decoder layers
        # store other representations for instrumentation in VocabParallelCrossEntCrit
        # Note: we are only storing the embeddings output and output of final transformer block
        # instead of all inner representations, as thats the only thing being logged and storing
        # all intermediate representation causes OOM for large models during validation.
        for idx, layer in enumerate(self.layers):
            x = layer(
                x,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # Returned x is T x B x C here, as sequence_parallel requires T to be first dim
        return x

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        return self.output_projection(features)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor, input_tokens=None) -> torch.Tensor:
        cur_seq_len, batch_size = tensor.size(0), tensor.size(1)
        max_seq_len = self.max_positions()
        need_to_make_new_mask = (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(1) < max_seq_len
            or (
                self._future_mask.size(0) != (batch_size * self.decoder_attention_heads)
            )
        )

        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if need_to_make_new_mask:
            self._future_mask = torch.triu(
                fill_with_neg_inf(
                    torch.zeros([max_seq_len, max_seq_len], device=tensor.device)
                ),
                1,
            )

        self._future_mask = self._future_mask.to(tensor)
        if self.self_attn_doc_sep != -1:
            return self._future_mask
        else:
            return self._future_mask[:cur_seq_len, :cur_seq_len]


def _sample_topp(temperature: float, sampling_topp: float, lprobs: torch.Tensor):
    if temperature == 0.0 or sampling_topp == 0.0:
        # greedy search
        return tuple(lprobs.max(dim=-1))

    probs = lprobs.exp()
    sprobs, sinds = probs.sort(dim=-1, descending=True)
    mask = (sprobs.cumsum(dim=-1) - sprobs) >= sampling_topp
    trunc_sprobs = sprobs.detach().clone()
    trunc_sprobs[mask] = 0
    trunc_sprobs.div_(trunc_sprobs.sum(dim=-1).unsqueeze(-1))
    choices = torch.multinomial(trunc_sprobs, 1)[:, 0]
    hyp_ids = torch.arange(lprobs.size(0)).to(lprobs.device)
    tok_ids = sinds[hyp_ids, choices]
    scores = sprobs[hyp_ids, choices].log()
    return scores, tok_ids


class SequenceGenerator(nn.Module):
    def __init__(
        self, model, beam_size: int, generate_size: int, use_incremental: bool = True
    ) -> None:
        super().__init__()
        self.model = model
        self.beam_size = beam_size
        self.generate_size = generate_size
        self.use_incremental = use_incremental

    def forward(self, src_tokens):
        with torch.no_grad():
            incremental_states = torch.jit.annotate(
                Dict[str, Dict[str, Optional[Tensor]]], {}
            )
            bsz, src_len = src_tokens.size()[:2]
            beam_size = self.beam_size

            max_len = src_len + self.generate_size
            new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
            new_order = new_order.to(src_tokens.device).long()

            tokens = (
                torch.zeros(bsz * beam_size, max_len).to(src_tokens).long().fill_(0)
            )

            start_step = src_tokens.shape[1]
            tokens[:, :start_step] = src_tokens.repeat_interleave(beam_size, 0)
            model_out = self.model(
                tokens[:, :start_step],
                incremental_state=incremental_states if self.use_incremental else None,
            )
            model_predictions = F.log_softmax(model_out.float()[:, -1, :])
            for step in range(start_step, max_len):
                tokens[:, step] = model_predictions.max(-1)[1]
                # forward through the next pass
                model_out = self.model(
                    tokens[:, : step + 1],
                    incremental_state=incremental_states
                    if self.use_incremental
                    else None,
                )
                # see above for why this must remain float
                model_predictions = F.log_softmax(model_out.float()[:, -1, :])
            return tokens


class SequenceGeneratorFixedSize(nn.Module):
    def __init__(self, model, beam_size: int, generate_size: int) -> None:
        super().__init__()
        self.model = model
        self.beam_size = beam_size
        self.generate_size = generate_size

    def forward(self, src_tokens):
        with torch.no_grad():
            bsz, src_len = src_tokens.size()[:2]
            beam_size = self.beam_size

            max_len = src_len + self.generate_size
            new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
            new_order = new_order.to(src_tokens.device).long()

            start_step = src_tokens.shape[1]
            tokens = (
                torch.zeros(bsz * beam_size, max_len).to(src_tokens).long().fill_(0)
            )
            tokens[:, :start_step] = src_tokens.repeat_interleave(beam_size, 0)

            model_out = self.model(tokens)
            model_predictions = F.log_softmax(model_out.float()[:, start_step, :])
            for step in range(start_step, max_len):
                tokens[:, step] = model_predictions.max(-1)[1]
                model_out = self.model(
                    tokens,
                )
                # see above for why this must remain float
                model_predictions = F.log_softmax(model_out.float()[:, step, :])
            return tokens


def create_model(embed_dim=1536):
    embed_tokens = torch.nn.Embedding(2048, embed_dim, padding_idx=-1)
    return (
        TransformerDecoder(
            embed_tokens,
            decoder_layers=24,
            decoder_attention_heads=16,
            max_target_positions=2048,
            embed_dim=embed_dim,
            decoder_ffn_embed_dim=embed_dim * 4,
            no_scale_embedding=True,
            share_decoder_input_output_embed=True,
            decoder_learned_pos=True,
            dropout=0.1,
        )
    )
