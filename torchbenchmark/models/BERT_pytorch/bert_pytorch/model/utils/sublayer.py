import torch
import torch.nn as nn
#from .layer_norm import LayerNorm
from .tensor2tensor import TensorToTensor

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer: TensorToTensor):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer.forward(self.norm(x)))
