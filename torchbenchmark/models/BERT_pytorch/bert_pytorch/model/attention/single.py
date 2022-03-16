import torch.nn as nn
import torch.nn.functional as F
import torch

import math
from ..utils.tensor2tensor import TensorToTensor
from typing import Optional

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, dropout: TensorToTensor, mask: Optional[torch.Tensor]=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            if scores.dtype == torch.float16:
                min_mask = -65504.0
            else:
                min_mask = -1e9
            scores = scores.masked_fill(mask == 0, min_mask)

        p_attn = F.softmax(scores, dim=-1)

        p_attn = dropout.forward(p_attn)

        return torch.matmul(p_attn, value), p_attn
