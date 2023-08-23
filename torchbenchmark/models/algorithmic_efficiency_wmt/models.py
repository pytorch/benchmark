import copy
import math
from typing import Any, Callable, Dict, Optional, Tuple, Union
import warnings

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.init import normal_
from torch.nn.init import xavier_uniform_


# Mask making utilities ported to PyTorch from
# https://github.com/google/flax/blob/main/flax/linen/attention.py.
def make_attention_mask(query_input: Tensor,
                        key_input: Tensor,
                        pairwise_fn: Callable[..., Any] = torch.mul,
                        dtype: torch.dtype = torch.float32) -> Tensor:
  """Mask-making helper for attention weights.

  Args:
    query_input: a batched, flat input of query_length size
    key_input: a batched, flat input of key_length size
    pairwise_fn: broadcasting elementwise comparison function
    dtype: mask return dtype

  Returns:
    A `[batch..., len_q, len_kv]` shaped attention mask.
  """
  mask = pairwise_fn(query_input.unsqueeze(-1), key_input.unsqueeze(-2))
  return mask.to(dtype)


def make_causal_mask(x: Tensor,
                     device: str = 'cuda:0',
                     dtype: torch.dtype = torch.float32) -> Tensor:
  """Make a causal mask for self-attention.

  Args:
    x: input array of shape `[batch..., len]`
    device: device to store the idxs
    dtype: mask return dtype

  Returns:
    A `[batch..., len, len]` shaped causal attention mask.
  """
  idxs = torch.broadcast_to(
      torch.arange(x.shape[-1], dtype=torch.int32, device=device), x.shape)
  return make_attention_mask(idxs, idxs, torch.greater_equal, dtype=dtype)


def make_src_mask(src, inputs_segmentation, nhead):
  """Utility for creating src mask and adjust it for PyTorch Transformer API."""
  src_mask = make_attention_mask(src > 0, src > 0)
  # Add segmentation block-diagonal attention mask if using segmented data.
  if inputs_segmentation is not None:
    src_mask = torch.logical_and(
        src_mask,
        make_attention_mask(inputs_segmentation, inputs_segmentation, torch.eq))
  # Flip values and ensure numerical stability.
  src_mask = torch.repeat_interleave(
      torch.logical_not(src_mask), repeats=nhead, dim=0)
  new_src_mask = torch.zeros_like(src_mask, dtype=torch.float32)
  new_src_mask.masked_fill_(src_mask, -1e10)
  return new_src_mask


def make_tgt_and_memory_mask(tgt,
                             src,
                             inputs_segmentation,
                             targets_segmentation,
                             decode,
                             nhead):
  """ Utility for creating target and memory mask and adjust them for PyTorch
  Transformer API."""
  if not decode:
    tgt_mask = torch.logical_and(
        make_attention_mask(tgt > 0, tgt > 0),
        make_causal_mask(tgt, device=tgt.device))
    memory_mask = make_attention_mask(tgt > 0, src > 0)
  else:
    tgt_mask = None
    memory_mask = make_attention_mask(torch.ones_like(tgt) > 0, src > 0)
  # Add segmentation block-diagonal attention masks if using segmented data.
  if inputs_segmentation is not None:
    tgt_mask = torch.logical_and(
        tgt_mask,
        make_attention_mask(targets_segmentation,
                            targets_segmentation,
                            torch.eq))
    memory_mask = torch.logical_and(
        memory_mask,
        make_attention_mask(targets_segmentation, inputs_segmentation,
                            torch.eq))
  # Flip values and ensure numerical stability.
  memory_mask = torch.repeat_interleave(
      torch.logical_not(memory_mask), repeats=nhead, dim=0)
  new_memory_mask = torch.zeros_like(memory_mask, dtype=torch.float32)
  new_memory_mask.masked_fill_(memory_mask, -1e10)
  if tgt_mask is not None:
    tgt_mask = torch.repeat_interleave(
        torch.logical_not(tgt_mask), repeats=nhead, dim=0)
    new_tgt_mask = torch.zeros_like(tgt_mask, dtype=torch.float32)
    new_tgt_mask.masked_fill_(tgt_mask, -1e10)
    tgt_mask = new_tgt_mask
  return tgt_mask, new_memory_mask


def shift_right(x, axis=1):
  """Shift the input to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  pad_widths = tuple(t for tup in reversed(pad_widths) for t in tup)
  padded = F.pad(x, pad_widths, mode='constant')
  return padded[:, :-1]


class Transformer(nn.Module):
  """Transformer architecture based on the model from the WMT Jax workload."""

  def __init__(self,
               ntoken: int = 32000,
               d_model: int = 1024,
               nhead: int = 16,
               d_hid: int = 1024,
               nlayers: int = 6,
               dropout_rate: Optional[float] = 0.1,
               attention_dropout_rate: Optional[float] = 0.1,
               layer_norm_eps: float = 1e-6):
    super().__init__()
    if dropout_rate is None:
      dropout_rate = 0.1
    if attention_dropout_rate is None:
      attention_dropout_rate = 0.1
    self.pos_encoder = PositionalEncoding(d_model, dropout_rate)
    self.shared_embedding = nn.Embedding(ntoken, d_model)
    self.encoder = Encoder(d_model,
                           nhead,
                           d_hid,
                           nlayers,
                           dropout_rate,
                           attention_dropout_rate,
                           layer_norm_eps)
    self.decoder = Decoder(d_model,
                           nhead,
                           d_hid,
                           nlayers,
                           dropout_rate,
                           attention_dropout_rate,
                           layer_norm_eps)
    # Share positional encoding and embedding between encoder and decoder.
    self.encoder.pos_encoder = self.pos_encoder
    self.encoder.shared_embedding = self.shared_embedding
    self.decoder.pos_encoder = self.pos_encoder
    self.decoder.shared_embedding = self.shared_embedding

    self._reset_parameters()

  def _reset_parameters(self):
    """Initiate parameters in the transformer model."""
    for module in self.modules():
      if isinstance(module, nn.Linear):
        xavier_uniform_(module.weight)
        if module.bias is not None:
          normal_(module.bias, std=1e-6)

  def forward(self,
              src: Tensor,
              tgt: Tensor,
              inputs_positions: Optional[Tensor] = None,
              targets_positions: Optional[Tensor] = None,
              inputs_segmentation: Optional[Tensor] = None,
              targets_segmentation: Optional[Tensor] = None,
              decode: bool = False) -> Tensor:
    """
    Args:
      src: Tensor, shape [batch_size, seq_len]
      tgt: Tensor, shape [batch_size, seq_len]
      inputs_positions: Optional[Tensor], shape [batch_size, seq_len]
      targets_positions: Optional[Tensor], shape [batch_size, seq_len]
      inputs_segmentation: Optional[Tensor], shape [batch_size, seq_len]
      targets_segmentation: Optional[Tensor], shape [batch_size, seq_len]
      decode: bool

    Returns:
      output Tensor of shape [batch_size, seq_len, ntoken]
    """
    if src.size(0) != tgt.size(0):
      raise RuntimeError('The batch size of src and tgt must be equal.')
    memory = self.encoder(
        src,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation)
    output = self.decoder(
        tgt,
        memory,
        src,  # just for calculating the padding mask
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        decode=decode)
    return output


class TransformerEncoder(nn.Module):
  r"""TransformerEncoder is a stack of N encoder layers. Users can build the
  BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

  Args:
      encoder_layer: an instance of the TransformerEncoderLayer() class.
      num_layers: the number of sub-encoder-layers in the encoder.
      norm: the layer normalization component (optional).
      enable_nested_tensor: if True, input will automatically convert to
        nested tensor (and convert back on output). This will improve
        the overall performance of TransformerEncoder when padding
        rate is high.

  Examples::
    >>> encoder_layer = nn.TransformerEncoderLayer(12, 8)
    >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, 6)
    >>> src = torch.rand(10, 32, 512)
    >>> out = transformer_encoder(src)
  """
  __constants__ = ['norm']

  def __init__(self,
               encoder_layer,
               num_layers,
               norm=None,
               enable_nested_tensor=True,
               mask_check=True):
    super().__init__()
    self.layers = nn.ModuleList(
        [copy.deepcopy(encoder_layer) for _ in range(num_layers)])
    self.num_layers = num_layers
    self.norm = norm
    self.enable_nested_tensor = enable_nested_tensor
    self.mask_check = mask_check

  def forward(self,
              src: Tensor,
              mask: Optional[Tensor] = None,
              src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
    """Pass the input through the encoder layers in turn.

    Args:
        src: the sequence to the encoder (required).
        mask: the mask for the src sequence (optional).
        src_key_padding_mask: the mask for the src keys per batch (optional).

    Shape:
        see the docs in Transformer class.
    """
    if src_key_padding_mask is not None:
      _skpm_dtype = src_key_padding_mask.dtype  # pylint: disable=invalid-name
      if _skpm_dtype != torch.bool and not torch.is_floating_point(
          src_key_padding_mask):
        raise AssertionError(
            'only bool and floating types of key_padding_mask are supported')
    output = src
    convert_to_nested = False
    src_key_padding_mask_for_layers = src_key_padding_mask

    for mod in self.layers:
      output = mod(
          output,
          src_mask=mask,
          src_key_padding_mask=src_key_padding_mask_for_layers)

    if convert_to_nested:
      output = output.to_padded_tensor(0.)

    if self.norm is not None:
      output = self.norm(output)

    return output


class Encoder(nn.Module):

  def __init__(self,
               d_model: int = 1024,
               nhead: int = 16,
               d_hid: int = 1024,
               nlayers: int = 6,
               dropout_rate: float = 0.1,
               attention_dropout_rate: float = 0.1,
               layer_norm_eps: float = 1e-6):
    super().__init__()
    self.nhead = nhead
    self.shared_embedding = None
    self.pos_encoder = None
    encoder_layer = TransformerEncoderLayer(
        d_model,
        nhead,
        d_hid,
        dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        layer_norm_eps=layer_norm_eps)
    encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
    self.encoder = TransformerEncoder(encoder_layer, nlayers, encoder_norm)

  def forward(self,
              src: Tensor,
              inputs_positions: Optional[Tensor] = None,
              inputs_segmentation: Optional[Tensor] = None) -> Tensor:
    src = src.to(torch.int)
    src_mask = make_src_mask(src, inputs_segmentation, self.nhead)
    src = self.shared_embedding(src)
    src = self.pos_encoder(src, inputs_positions)
    memory = self.encoder(src, mask=src_mask)
    return memory


class Decoder(nn.Module):

  def __init__(self,
               d_model: int = 1024,
               nhead: int = 16,
               d_hid: int = 1024,
               nlayers: int = 6,
               dropout_rate: float = 0.1,
               attention_dropout_rate: float = 0.1,
               layer_norm_eps: float = 1e-6):
    super().__init__()
    self.nhead = nhead
    self.shared_embedding = None
    self.pos_encoder = None
    self.decoder = TransformerDecoder(d_model,
                                      nhead,
                                      d_hid,
                                      dropout_rate,
                                      attention_dropout_rate,
                                      layer_norm_eps,
                                      nlayers)

  def forward(
      self,
      tgt: Tensor,
      memory: Tensor,
      src: Tensor,  # just for calculating the padding mask
      targets_positions: Optional[Tensor] = None,
      inputs_segmentation: Optional[Tensor] = None,
      targets_segmentation: Optional[Tensor] = None,
      decode: bool = False,
      max_len: Optional[int] = None,
      cache: Optional[dict] = None) -> Any:
    tgt = tgt.to(torch.int)
    tgt_mask, memory_mask = make_tgt_and_memory_mask(
        tgt, src, inputs_segmentation, targets_segmentation,
        decode, self.nhead)
    if not decode:
      tgt = shift_right(tgt)
    tgt = self.shared_embedding(tgt)
    tgt = self.pos_encoder(tgt, targets_positions, decode=decode, cache=cache)
    if decode:
      tgt, cache = tgt
    output = self.decoder(
        tgt,
        memory,
        tgt_mask=tgt_mask,
        memory_mask=memory_mask,
        decode=decode,
        max_len=max_len,
        cache=cache)
    if decode:
      output, cache = output
    normalize = math.sqrt(output.shape[-1])
    output = torch.matmul(output, self.shared_embedding.weight.T) / normalize
    if decode:
      return output, cache
    return output


class PositionalEncoding(nn.Module):

  def __init__(self,
               d_model: int,
               dropout_rate: float = 0.1,
               max_len: int = 256):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout_rate)

    position = torch.arange(max_len).unsqueeze(1)
    scale_factor = -math.log(10000.0) / (d_model // 2 - 1)
    div_term = torch.exp(torch.arange(d_model // 2) * scale_factor)
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, :d_model // 2] = torch.sin(position * div_term)
    pe[0, :, d_model // 2:2 * (d_model // 2)] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(
      self,
      x: Tensor,
      inputs_positions: Optional[Tensor] = None,
      decode: bool = False,
      cache: Optional[Dict[str, Dict[str, Tensor]]] = None
  ) -> Union[Tensor, Tuple[Tensor, Dict[str, Dict[str, Tensor]]]]:
    """
    Args:
      x: Tensor (shape [batch_size, seq_len, embedding_dim])
      inputs_positions: Tensor (shape [batch_size, seq_len]) or None
      decode: bool
      cache: Dict[str, Dict[str, Tensor]] or None
    Returns:
      Tensor or Tuple[Tensor, Dict[str, Dict[str, Tensor]]]
    """
    # We use a cache position index for tracking decoding position.
    if decode:
      name = self._get_name()
      if cache is None:
        cache = {
            name: {
                'cache_index':
                    torch.tensor(0, dtype=torch.long, device=self.pe.device),
            },
        }
      pe = self.pe[0, cache[name]['cache_index'], :]
      cache[name]['cache_index'] += 1
      return self.dropout(x + pe), cache
    if inputs_positions is None:
      # normal unpacked case:
      pe = self.pe[:, :x.size(1), :]
    else:
      # for packed data we need to use known position indices:
      pe = self.pe[0, inputs_positions, :]
    return self.dropout(x + pe)


# TransformerEncoderLayer and TransformerDecoderLayer are taken from:
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
# Only difference is using custom MultiheadAttention modules without bias and
# '_qkv_same_embed_dim' always set to 'False'.
class TransformerEncoderLayer(nn.Module):
  r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
  This standard encoder layer is based on the paper "Attention Is All You Need".
  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
  Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all
  you need. In Advances in Neural Information Processing Systems,
  pages 6000-6010. Users may modify or implement in a different way during
  application.
  Args:
    d_model: the number of expected features in the input (default=1024).
    nhead: the number of heads in the multiheadattention models (default=16).
    dim_feedforward: the dimension of the feedforward network model
        (default=1024).
    dropout_rate: the dropout_rate value (default=0.1).
    activation: the activation function of the intermediate layer, can be a
       string ("relu" or "gelu") or a unary callable (default=F.relu).
    layer_norm_eps: the eps value in layer normalization components
        (default=1e-6).
    batch_first: If ``True``, then the input and output tensors are provided
        as (batch, seq, feature). Default: ``True`` (batch, seq, feature).
    norm_first: if ``True``, layer norm is done prior to attention and
        feedforward operations, respectivaly. Otherwise it's done after.
        Default: ``True``.
  Examples::
    >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    >>> src = torch.rand(10, 32, 512)
    >>> out = encoder_layer(src)
  Alternatively, when ``batch_first`` is ``True``:
    >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8,
        batch_first=True)
    >>> src = torch.rand(32, 10, 512)
    >>> out = encoder_layer(src)
  """
  __constants__ = ['batch_first', 'norm_first']

  def __init__(self,
               d_model: int = 1024,
               nhead: int = 16,
               dim_feedforward: int = 1024,
               dropout_rate: float = 0.1,
               attention_dropout_rate: float = 0.1,
               activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
               layer_norm_eps: float = 1e-6,
               batch_first: bool = True,
               norm_first: bool = True,
               device=None,
               dtype=None) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.self_attn = MultiheadAttention(
        d_model,
        nhead,
        dropout_rate=attention_dropout_rate,
        batch_first=batch_first,
        bias=False,
        **factory_kwargs)

    # Implementation of Feedforward model.
    self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
    self.dropout = nn.Dropout(dropout_rate)
    self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

    self.norm_first = norm_first
    self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    self.dropout1 = nn.Dropout(dropout_rate)
    self.dropout2 = nn.Dropout(dropout_rate)

    # We can't test self.activation in forward() in TorchScript,
    # so stash some information about it instead.
    if activation is F.relu or isinstance(activation, torch.nn.ReLU):
      self.activation_relu_or_gelu = 1
    elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
      self.activation_relu_or_gelu = 2
    else:
      self.activation_relu_or_gelu = 0
    self.activation = activation

  def forward(self,
              src: Tensor,
              src_mask: Optional[Tensor] = None,
              src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
    r"""Pass the input through the encoder layer.

    Args:
        src: the sequence to the encoder layer (required).
        src_mask: the mask for the src sequence (optional).
        src_key_padding_mask: the mask for the src keys per batch (optional).

    Shape:
        see the docs in Transformer class.
    """
    if src_key_padding_mask is not None:
      _skpm_dtype = src_key_padding_mask.dtype  # pylint: disable=invalid-name
      if _skpm_dtype != torch.bool and not torch.is_floating_point(
          src_key_padding_mask):
        raise AssertionError(
            'Only bool and floating types of key_padding_mask are supported')
    x = src
    if self.norm_first:
      x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
      x = x + self._ff_block(self.norm2(x))
    else:
      x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
      x = self.norm2(x + self._ff_block(x))

    return x

  # Self-attention block:
  def _sa_block(self,
                x: Tensor,
                attn_mask: Optional[Tensor],
                key_padding_mask: Optional[Tensor]) -> Tensor:
    x = self.self_attn(
        x,
        x,
        x,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=False)[0]
    return self.dropout1(x)

  # Feed forward block:
  def _ff_block(self, x: Tensor) -> Tensor:
    x = self.linear2(self.dropout(self.activation(self.linear1(x))))
    return self.dropout2(x)


# Modified to use cache for autoregressive decoding.
class TransformerDecoder(nn.Module):
  r"""TransformerDecoder is a stack of N decoder layers
  Args:
    d_model: the number of expected features in the input (default=1024)
    nhead: the number of heads in the multiheadattention models (default=16)
    d_hid: the dimension of the feedforward network model
        (default=1024)
    dropout_rate: the dropout_rate value (default=0.1)
    layer_norm_eps: the eps value in layer normalization components
        (default=1e-6).
    decoder_layer: an instance of the TransformerDecoderLayer() class
    num_layers: the number of sub-decoder-layers in the decoder
  Examples::
    >>> decoder_layer = nn.TransformerDecoderLayer(12, 8)
    >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, 6)
    >>> memory = torch.rand(10, 32, 512)
    >>> tgt = torch.rand(20, 32, 512)
    >>> out = transformer_decoder(tgt, memory)
  """
  __constants__ = ['norm']

  def __init__(self,
               d_model,
               nhead,
               d_hid,
               dropout_rate,
               attention_dropout_rate,
               layer_norm_eps,
               num_layers):
    super().__init__()
    self.layers = nn.ModuleList([
        TransformerDecoderLayer(
            d_model,
            nhead,
            d_hid,
            dropout_rate,
            attention_dropout_rate,
            layer_norm_eps=layer_norm_eps) for _ in range(num_layers)
    ])
    self.num_layers = num_layers
    self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

  def forward(self,
              tgt: Tensor,
              memory: Tensor,
              tgt_mask: Optional[Tensor] = None,
              memory_mask: Optional[Tensor] = None,
              decode: bool = False,
              max_len: Optional[int] = None,
              cache: Optional[dict] = None) -> Any:
    r"""Pass the inputs (and mask) through the decoder layer in turn.
    Args:
      tgt: the sequence to the decoder (required).
      memory: the sequence from the last layer of the encoder (required).
      tgt_mask: the mask for the tgt sequence (optional).
      memory_mask: the mask for the memory sequence (optional).
      decode: wether to use cache for autoregressive decoding or not.
      max_len: maximum sequence length, necessary for decoding cache.
    Shape:
      see the docs in Transformer class.
    """
    output = tgt

    for idx, mod in enumerate(self.layers):
      output, cache = mod(
          output,
          memory,
          tgt_mask=tgt_mask,
          memory_mask=memory_mask,
          decode=decode,
          max_len=max_len,
          cache=cache,
          index=idx)

    if self.norm is not None:
      output = self.norm(output)

    if decode:
      return output, cache
    return output


# Modified to use cache for autoregressive decoding.
class TransformerDecoderLayer(nn.Module):
  r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and
  feedforward network.
  This standard decoder layer is based on the paper "Attention Is All You Need".
  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
  Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all
  you need. In Advances in Neural Information Processing Systems,
  pages 6000-6010. Users may modify or implement in a different way during
  application.
  Args:
    d_model: the number of expected features in the input (default=1024).
    nhead: the number of heads in the multiheadattention models (default=16).
    dim_feedforward: the dimension of the feedforward network model
        (default=1024).
    dropout_rate: the dropout_rate value (default=0.1).
    activation: the activation function of the intermediate layer, can be a
        string ("relu" or "gelu") or a unary callable (default=F.relu).
    layer_norm_eps: the eps value in layer normalization components
        (default=1e-6).
    batch_first: If ``True``, then the input and output tensors are provided
        as (batch, seq, feature). Default: ``True`` (batch, seq, feature).
    norm_first: if ``True``, layer norm is done prior to self attention,
        multihead attention and feedforward operations, respectivaly.
        Otherwise it's done after. Default: ``True``.
  Examples::
    >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    >>> memory = torch.rand(10, 32, 512)
    >>> tgt = torch.rand(20, 32, 512)
    >>> out = decoder_layer(tgt, memory)
  Alternatively, when ``batch_first`` is ``True``:
    >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8,
        batch_first=True)
    >>> memory = torch.rand(32, 10, 512)
    >>> tgt = torch.rand(32, 20, 512)
    >>> out = decoder_layer(tgt, memory)
  """
  __constants__ = ['batch_first', 'norm_first']

  def __init__(self,
               d_model: int = 1024,
               nhead: int = 16,
               dim_feedforward: int = 1024,
               dropout_rate: float = 0.1,
               attention_dropout_rate: float = 0.1,
               activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
               layer_norm_eps: float = 1e-6,
               batch_first: bool = True,
               norm_first: bool = True,
               device=None,
               dtype=None) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.self_attn = MultiheadAttention(
        d_model,
        nhead,
        dropout_rate=attention_dropout_rate,
        batch_first=batch_first,
        bias=False,
        **factory_kwargs)
    self.multihead_attn = MultiheadAttention(
        d_model,
        nhead,
        dropout_rate=attention_dropout_rate,
        batch_first=batch_first,
        bias=False,
        **factory_kwargs)

    # Implementation of Feedforward model.
    self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
    self.dropout = nn.Dropout(dropout_rate)
    self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

    self.norm_first = norm_first
    self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    self.dropout1 = nn.Dropout(dropout_rate)
    self.dropout2 = nn.Dropout(dropout_rate)
    self.dropout3 = nn.Dropout(dropout_rate)

    self.activation = activation

  def forward(  # pylint: disable=arguments-renamed
      self,
      tgt: Tensor,
      memory: Tensor,
      tgt_mask: Optional[Tensor] = None,
      memory_mask: Optional[Tensor] = None,
      decode: bool = False,
      max_len: Optional[int] = None,
      cache: Optional[dict] = None,
      index: Optional[int] = None) -> Any:
    r"""Pass the inputs (and mask) through the decoder layer.
    Args:
      tgt: the sequence to the decoder layer (required).
      memory: the sequence from the last layer of the encoder (required).
      tgt_mask: the mask for the tgt sequence (optional).
      memory_mask: the mask for the memory sequence (optional).
      decode: wether to use cache for autoregressive decoding or not.
      max_len: maximum sequence length, necessary for decoding cache.
    Shape:
      see the docs in Transformer class.
    """
    # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

    x = tgt
    if self.norm_first:
      sa_out, cache = self._sa_block(
          self.norm1(x),
          tgt_mask,
          decode=decode,
          max_len=max_len,
          cache=cache,
          index=index)
      x = x + sa_out
      x = x + self._mha_block(self.norm2(x), memory, memory_mask, None)
      x = x + self._ff_block(self.norm3(x))
    else:
      sa_out, cache = self._sa_block(
          x,
          tgt_mask,
          decode=decode,
          max_len=max_len,
          cache=cache,
          index=index)
      x = self.norm1(x + sa_out)
      x = self.norm2(x + self._mha_block(x, memory, memory_mask, None))
      x = self.norm3(x + self._ff_block(x))

    return x, cache

  # Self-attention block:
  def _sa_block(  # pylint: disable=arguments-renamed
      self,
      x: Tensor,
      attn_mask: Optional[Tensor],
      decode: bool = False,
      max_len: Optional[int] = None,
      cache: Optional[dict] = None,
      index: Optional[int] = None) -> Any:
    x, _, cache = self.self_attn(
        x,
        x,
        x,
        attn_mask=attn_mask,
        need_weights=False,
        decode=decode,
        max_len=max_len,
        cache=cache,
        index=index)
    return self.dropout1(x), cache

  # Multihead attention block:
  def _mha_block(self,
                 x: Tensor,
                 mem: Tensor,
                 attn_mask: Optional[Tensor],
                 key_padding_mask: Optional[Tensor]) -> Tensor:
    x = self.multihead_attn(
        x,
        mem,
        mem,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=False)[0]
    return self.dropout2(x)

  # Feed forward block.
  def _ff_block(self, x: Tensor) -> Tensor:
    x = self.linear2(self.dropout(self.activation(self.linear1(x))))
    return self.dropout3(x)


# Only difference to standard PyTorch class is that 'self._qkv_same_embed_dim'
# is always set to 'False' and the use of a cache registered as a buffer for
# autoregressive decoding.
class MultiheadAttention(nn.MultiheadAttention):
  r"""Allows the model to jointly attend to information
  from different representation subspaces.
  See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.
  .. math::
      \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
  where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.
  Args:
    embed_dim: Total dimension of the model.
    num_heads: Number of parallel attention heads. Note that ``embed_dim`` will
        be split across ``num_heads`` (i.e. each head will have dimension
        ``embed_dim // num_heads``).
    dropout_rate: Dropout probability on ``attn_output_weights``.
        Default: ``0.0`` (no dropout_rate).
    bias: If specified, adds bias to input / output projection layers.
       Default: ``True``.
    add_bias_kv: If specified, adds bias to the key and value sequences at
        dim=0. Default: ``False``.
    add_zero_attn: If specified, adds a new batch of zeros to the key and value
        sequences at dim=1. Default: ``False``.
    kdim: Total number of features for keys. Default: ``None``
        (uses ``kdim=embed_dim``).
    vdim: Total number of features for values. Default: ``None``
        (uses ``vdim=embed_dim``).
    batch_first: If ``True``, then the input and output tensors are provided
        as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
  Examples::
    >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
    >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
  """

  def __init__(self,
               embed_dim,
               num_heads,
               dropout_rate=0.,
               bias=True,
               add_bias_kv=False,
               add_zero_attn=False,
               kdim=None,
               vdim=None,
               batch_first=True,
               device=None,
               dtype=None) -> None:
    super().__init__(
        embed_dim,
        num_heads,
        dropout=dropout_rate,
        bias=bias,
        add_bias_kv=add_bias_kv,
        add_zero_attn=add_zero_attn,
        kdim=kdim,
        vdim=vdim,
        batch_first=batch_first,
        device=device,
        dtype=dtype)
    # This is set to 'True' for kdim == vdim == embed_dim in the standard
    # PyTorch class.
    self._qkv_same_embed_dim = False

    factory_kwargs = {'device': device, 'dtype': dtype}
    self.q_proj_weight = nn.Parameter(
        torch.empty((embed_dim, embed_dim), **factory_kwargs))
    self.k_proj_weight = nn.Parameter(
        torch.empty((embed_dim, self.kdim), **factory_kwargs))
    self.v_proj_weight = nn.Parameter(
        torch.empty((embed_dim, self.vdim), **factory_kwargs))
    self.register_parameter('in_proj_weight', None)

    self._reset_parameters()

  def _reset_parameters(self):
    if self._qkv_same_embed_dim:
      xavier_uniform_(self.in_proj_weight)
    else:
      xavier_uniform_(self.q_proj_weight)
      xavier_uniform_(self.k_proj_weight)
      xavier_uniform_(self.v_proj_weight)

    if self.in_proj_bias is not None:
      normal_(self.in_proj_bias, std=1e-6)
      normal_(self.out_proj.bias, std=1e-6)
    if self.bias_k is not None:
      normal_(self.bias_k, std=1e-6)
    if self.bias_v is not None:
      normal_(self.bias_v, std=1e-6)

  def forward(self,
              query: Tensor,
              key: Tensor,
              value: Tensor,
              key_padding_mask: Optional[Tensor] = None,
              need_weights: bool = True,
              attn_mask: Optional[Tensor] = None,
              average_attn_weights: bool = True,
              decode: bool = False,
              max_len: Optional[int] = None,
              cache: Optional[dict] = None,
              index: Optional[int] = None) -> Any:
    r"""
    Args:
      query: Query embeddings of shape :math:`(L, E_q)` for unbatched input,
          :math:`(L, N, E_q)` when ``batch_first=False`` or :math:`(N, L, E_q)`
          when ``batch_first=True``, where :math:`L` is the target sequence
          length, :math:`N` is the batch size, and :math:`E_q` is the query
          embedding dimension ``embed_dim``.
          Queries are compared against key-value pairs to produce the output.
          See "Attention Is All You Need" for more details.
      key: Key embeddings of shape :math:`(S, E_k)` for unbatched input,
          :math:`(S, N, E_k)` when ``batch_first=False`` or :math:`(N, S, E_k)`
          when ``batch_first=True``, where :math:`S` is the source sequence
          length, :math:`N` is the batch size, and :math:`E_k` is the key
          embedding dimension ``kdim``.
          See "Attention Is All You Need" for more details.
      value: Value embeddings of shape :math:`(S, E_v)` for unbatched input,
          :math:`(S, N, E_v)` when ``batch_first=False`` or :math:`(N, S, E_v)`
          when ``batch_first=True``, where :math:`S` is the source
          sequence length, :math:`N` is the batch size, and :math:`E_v` is the
          value embedding dimension ``vdim``.
          See "Attention Is All You Need" for more details.
      key_padding_mask: Dummy argument to make MultiheadAttention compatible
          with standard PyTorch TransformerEncoder implementation.
      need_weights: If specified, returns ``attn_output_weights`` in addition
          to ``attn_outputs``.Default: ``True``.
      attn_mask: If specified, a 2D or 3D mask preventing attention to certain
          positions. Must be of shape :math:`(L, S)` or
          :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the
          batch size, :math:`L` is the target sequence length, and :math:`S`
          is the source sequence length. A 2D mask will be broadcasted across
          the batch while a 3D mask allows for a different mask for each entry
          in the batch. Binary, byte, and float masks are supported.
          For a binary mask, a ``True`` value indicates that the
          corresponding position is not allowed to attend. For a byte mask,
          a non-zero value indicates that the corresponding position is not
          allowed to attend. For a float mask, the mask values will be added to
          the attention weight.
      average_attn_weights: If true, indicates that the returned
          ``attn_weights`` should be averaged across heads. Otherwise,
          ``attn_weights`` are provided separately per head. Note that this
          flag only has an effect when ``need_weights=True``. Default:
          ``True`` (i.e. average weights across heads)
      decode: wether to use cache for autoregressive decoding or not.
      max_len: maximum sequence length, necessary for decoding cache.
    Outputs:
      - **attn_output** - Attention outputs of shape :math:`(L, E)` when input
        is unbatched, :math:`(L, N, E)` when ``batch_first=False`` or
        :math:`(N, L, E)` when ``batch_first=True``,
        where :math:`L` is the target sequence length, :math:`N` is the batch
        size, and :math:`E` is the embedding dimension ``embed_dim``.
      - **attn_output_weights** - Only returned when ``need_weights=True``.
        If ``average_attn_weights=True``, returns attention weights averaged
        across heads of shape :math:`(L, S)` when input is unbatched or
        :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the
        target sequence length, and :math:`S` is the source sequence length.
        If ``average_weights=False``, returns attention weights per
        head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched
        or :math:`(N, \text{num\_heads}, L, S)`.
      .. note::
          `batch_first` argument is ignored for unbatched inputs.
    """
    del key_padding_mask
    is_batched = query.dim() == 3
    if self.batch_first and is_batched:
      # make sure that the transpose op does not affect the "is" property
      if key is value:
        if query is key:
          query = key = value = query.transpose(1, 0)
        else:
          query, key = [x.transpose(1, 0) for x in (query, key)]
          value = key
      else:
        query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

    name = f'decoder.layers.{index}.self_attn'
    loc_cache = cache[name] if decode and name in cache else None

    attn_output, attn_output_weights, loc_cache = multi_head_attention_forward(
        query, key, value, self.embed_dim, self.num_heads,
        self.in_proj_bias, self.bias_k, self.bias_v,
        self.dropout, self.out_proj.weight, self.out_proj.bias,
        training=self.training, need_weights=need_weights, attn_mask=attn_mask,
        q_proj_weight=self.q_proj_weight,
        k_proj_weight=self.k_proj_weight,
        v_proj_weight=self.v_proj_weight,
        average_attn_weights=average_attn_weights,
        decode=decode, cache=loc_cache, max_len=max_len)

    if decode:
      cache[name] = loc_cache

    if self.batch_first and is_batched:
      return attn_output.transpose(1, 0), attn_output_weights, cache
    else:
      return attn_output, attn_output_weights, cache


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
  r"""Performs the in-projection step of the attention operation. This is simply
  a triple of linear projections, with shape constraints on the weights which
  ensure embedding dimension uniformity in the projected outputs.
  Output is a triple containing projection tensors for query, key and value.
  """
  eq, ek = q.size(-1), k.size(-1)
  assert w_q.shape == (eq, eq), \
    f'Expecting query weights shape of {(eq, eq)}, but got {w_q.shape}'
  assert w_k.shape == (eq, ek), \
    f'Expecting key weights shape of {(eq, ek)}, but got {w_k.shape}'
  assert w_v.shape == (eq, ek), \
    f'Expecting value weights shape of {(eq, ek)}, but got {w_v.shape}'
  assert b_q is None or b_q.shape == (eq,), \
    f'Expecting query bias shape of {(eq,)}, but got {b_q.shape}'
  assert b_k is None or b_k.shape == (eq,), \
    f'Expecting key bias shape of {(eq,)}, but got {b_k.shape}'
  assert b_v is None or b_v.shape == (eq,), \
    f'Expecting value bias shape of {(eq,)}, but got {b_v.shape}'
  return torch.nn.functional.linear(q, w_q, b_q), \
    torch.nn.functional.linear(k, w_k, b_k), \
    torch.nn.functional.linear(v, w_v, b_v)


# Modified to create cache for autoregressive decoding.
def multi_head_attention_forward(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 embed_dim_to_check: int,
                                 num_heads: int,
                                 in_proj_bias: Optional[Tensor],
                                 bias_k: Optional[Tensor],
                                 bias_v: Optional[Tensor],
                                 dropout_rate: float,
                                 out_proj_weight: Tensor,
                                 out_proj_bias: Optional[Tensor],
                                 training: bool = True,
                                 need_weights: bool = True,
                                 attn_mask: Optional[Tensor] = None,
                                 q_proj_weight: Optional[Tensor] = None,
                                 k_proj_weight: Optional[Tensor] = None,
                                 v_proj_weight: Optional[Tensor] = None,
                                 average_attn_weights: bool = True,
                                 decode: bool = False,
                                 cache: Optional[dict] = None,
                                 max_len: Optional[int] = None) -> Any:
  r"""
  Args:
    query, key, value: map a query and a set of key-value pairs to an output.
        See "Attention Is All You Need" for more details.
    embed_dim_to_check: total dimension of the model.
    num_heads: parallel attention heads.
    in_proj_bias: input projection bias.
    bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
    dropout_rate: probability of an element to be zeroed.
    out_proj_weight, out_proj_bias: the output projection weight and bias.
    training: apply dropout_rate if is ``True``.
    need_weights: output attn_output_weights.
    attn_mask: 2D or 3D mask that prevents attention to certain positions.
    A 2D mask will be broadcasted for all
        the batches while a 3D mask allows to specify a different mask for the
        entries of each batch.
    q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias:
        input projection weight and bias.
    average_attn_weights: If true, indicates that the returned ``attn_weights``
        should be averaged across heads.
        Otherwise, ``attn_weights`` are provided separately per head.
        Note that this flag only has an effect when ``need_weights=True.``.
        Default: True
    decode: wether to use cache for autoregressive decoding or not.
    cache: dict which contains cache for decoding for the current
        MulitheadAttention module.
    max_len: maximum sequence length, necessary for decoding cache.
  Shape:
    Inputs:
    - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence
      length, N is the batch size, E is the embedding dimension.
    - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence
      length, N is the batch size, E is the embedding dimension.
    - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence
      length, N is the batch size, E is the embedding dimension.
    - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length,
      S is the source sequence length. 3D mask :math:`(N*num_heads, L, S)`
      where N is the batch size, L is the target sequence length,
      S is the source sequence length. attn_mask ensures that position i is
      allowed to attend the unmasked positions. If a ByteTensor is provided,
      the non-zero positions are not allowed to attend while the zero positions
      will be unchanged. If a BoolTensor is provided, positions with ``True``
      are not allowed to attend while ``False`` values will be unchanged.
      If a FloatTensor is provided, it will be added to the attention weight.
  Outputs:
    - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target
      sequence length, N is the batch size, E is the embedding dimension.
    - attn_output_weights: Only returned when ``need_weights=True``.
      If ``average_attn_weights=True``, returns
      attention weights averaged across heads of shape :math:`(L, S)` when input
      is unbatched or :math:`(N, L, S)`, where :math:`N` is the batch size,
      :math:`L` is the target sequence length, and :math:`S` is the source
      sequence length. If ``average_weights=False``, returns attention weights
      per head of shape :math:`(num_heads, L, S)` when input is unbatched or
      :math:`(N, num_heads, L, S)`.
  """
  # Set up shape variables.
  tgt_len, bsz, embed_dim = query.shape
  src_len, _, _ = key.shape
  assert embed_dim == embed_dim_to_check, \
      f'was expecting dimension of {embed_dim_to_check}, but got {embed_dim}'
  if isinstance(embed_dim, torch.Tensor):
    # `embed_dim` can be a tensor when JIT tracing.
    head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
  else:
    head_dim = embed_dim // num_heads
  assert head_dim * num_heads == embed_dim, \
      f'embed_dim {embed_dim} not divisible by num_heads {num_heads}'
  # Allow MHA to have different embedding dimensions when separate projection
  # weights are used.
  assert key.shape[:2] == value.shape[:2], \
      (f"key's sequence and batch dims {key.shape[:2]} do not match value's "
       f'{value.shape[:2]}')

  # Compute in-projection.
  assert q_proj_weight is not None, \
      'use_separate_proj_weight is True but q_proj_weight is None'
  assert k_proj_weight is not None, \
      'use_separate_proj_weight is True but k_proj_weight is None'
  assert v_proj_weight is not None, \
      'use_separate_proj_weight is True but v_proj_weight is None'
  if in_proj_bias is None:
    b_q = b_k = b_v = None
  else:
    b_q, b_k, b_v = in_proj_bias.chunk(3)
  q, k, v = _in_projection(
      query, key, value, q_proj_weight, k_proj_weight,
      v_proj_weight, b_q, b_k, b_v)

  # During fast autoregressive decoding, we feed one position at a time,
  # and cache the keys and values step by step.
  if decode:
    if cache is None:
      cache = {
          'cached_key':
              torch.zeros((bsz, max_len, embed_dim),
                          dtype=k.dtype,
                          device=k.device),
          'cached_value':
              torch.zeros((bsz, max_len, embed_dim),
                          dtype=v.dtype,
                          device=v.device),
          'cache_index':
              torch.tensor(0, dtype=torch.long, device=k.device),
      }
    cached_key = cache['cached_key']
    cached_value = cache['cached_value']
    cache_index = cache['cache_index']
    batch_size, max_length, num_features = cached_key.shape
    assert batch_size == bsz, f'{batch_size} != {bsz}'
    assert max_length == max_len, f'{max_length} != {max_len}'
    assert num_features == embed_dim, f'{num_features} != {embed_dim}'
    # Shape check of cached keys against query input.
    expected_shape = (1, batch_size, num_features)
    if expected_shape != query.shape:
      raise ValueError('Autoregressive cache shape error, expected query shape '
                       f'{expected_shape} instead got {query.shape}.')
    # Update key, value caches with our new 1d spatial slices.
    cached_key[:, cache_index:cache_index + 1, :] = k.transpose(dim0=0, dim1=1)
    cached_value[:, cache_index:cache_index + 1, :] = v.transpose(
        dim0=0, dim1=1)
    k = cached_key.transpose(dim0=0, dim1=1)
    v = cached_value.transpose(dim0=0, dim1=1)
    cache_index += 1
    # Causal mask for cached decoder self-attention:
    # our single query position should only attend to those key
    # positions that have already been generated and cached,
    # not the remaining zero elements.
    if attn_mask is not None:
      raise ValueError('Attention mask has to be None for decode == True.')
    attn_mask = (torch.arange(max_length, device=k.device) >=
                 cache_index).reshape(1, max_length)

  # Prepare attention mask.
  if not decode and attn_mask is not None:
    if attn_mask.dtype == torch.uint8:
      warnings.warn(
          'Byte tensor for attn_mask in nn.MultiheadAttention is deprecated.'
          'Use bool tensor instead.')
      attn_mask = attn_mask.to(torch.bool)
    else:
      assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
          f'float, byte, and bool types are supported, not {attn_mask.dtype}'
    # ensure attn_mask's dim is 3
    if attn_mask.dim() == 2:
      correct_2d_size = (tgt_len, src_len)
      if attn_mask.shape != correct_2d_size:
        raise RuntimeError(
            f'The shape of the 2D attn_mask is {attn_mask.shape}, '
            f'but should be {correct_2d_size}.')
      attn_mask = attn_mask.unsqueeze(0)
    elif attn_mask.dim() == 3:
      correct_3d_size = (bsz * num_heads, tgt_len, src_len)
      if attn_mask.shape != correct_3d_size:
        raise RuntimeError(f'The shape of attn_mask is {attn_mask.shape}, '
                           f'should be {correct_3d_size}.')
    else:
      raise RuntimeError(
          f"attn_mask's dimension {attn_mask.dim()} is not supported")

  # Add bias along batch dimension (currently second).
  if bias_k is not None and bias_v is not None:
    k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
    v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
    if attn_mask is not None:
      attn_mask = F.pad(attn_mask, (0, 1))
  else:
    assert bias_k is None
    assert bias_v is None

  # Reshape q, k, v for multihead attention and make em batch first.
  q = \
    q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
  k = \
    k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
  v = \
    v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

  # Update source sequence length after adjustments.
  src_len = k.size(1)

  # Convert mask to float.
  if attn_mask is not None and attn_mask.dtype == torch.bool:
    new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
    new_attn_mask.masked_fill_(attn_mask, -1e10)
    attn_mask = new_attn_mask

  # Adjust dropout_rate probability.
  if not training:
    dropout_rate = 0.0

  # Calculate attention and out projection.
  attn_output = torch.nn.functional.scaled_dot_product_attention(
      q, k, v, attn_mask, dropout_rate)
  attn_output = attn_output.transpose(0, 1).contiguous().view(
      tgt_len * bsz, embed_dim)
  attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
  attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

  if need_weights:
    q_scaled = q / math.sqrt(q.shape[-1])

    if attn_mask is not None:
      attn_output_weights = torch.baddbmm(attn_mask,
                                          q_scaled,
                                          k.transpose(-2, -1))
    else:
      attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))

    # Optionally average attention weights over heads.
    attn_output_weights = attn_output_weights.view(bsz,
                                                   num_heads,
                                                   tgt_len,
                                                   src_len)
    if average_attn_weights:
      attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
    return attn_output, attn_output_weights, cache
  else:
    return attn_output, None, cache
