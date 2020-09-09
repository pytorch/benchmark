import torch
import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward
from .utils.tensor2tensor import TensorToTensor


class LambdaModule(torch.nn.Module):
    def __init__(self, att):
        super().__init__()
        self.attention = att
        self.mask = torch.zeros((4))

    @torch.jit.export
    def set_mask(self, mask: torch.Tensor):
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention.forward(x, x, x, mask=self.mask)


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.lambda_module = LambdaModule(self.attention)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        self.lambda_module.set_mask(mask)
        x = self.input_sublayer(x, self.lambda_module)
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
