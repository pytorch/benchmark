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
                """
                -1e9 is overflow in fp16. It needs to be set a min.
                Theoretically, the mask for empty token needs to be set as -inf. Check https://arxiv.org/pdf/1706.03762.pdf
                """
                min_mask = -65504.0 #torch.finfo(torch.float16).min == -65504.0. jit scripting could handle finfo
            else:
                min_mask = -1e9
            scores = scores.masked_fill(mask == 0, min_mask)

        p_attn = F.softmax(scores, dim=-1)

        p_attn = dropout.forward(p_attn)

        return torch.matmul(p_attn, value), p_attn
