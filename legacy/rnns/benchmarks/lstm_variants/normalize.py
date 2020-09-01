#!/usr/bin/env python

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V

"""
Implementation of various normalization techniques. Also only works on instances
where batch size = 1.

"""


class LayerNorm(nn.Module):
    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, input_size, learnable=True, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.alpha = T(1, input_size).fill_(0)
        self.beta = T(1, input_size).fill_(0)
        self.epsilon = epsilon
        # Wrap as parameters if necessary
        if learnable:
            W = P
        else:
            W = V
        self.alpha = W(self.alpha)
        self.beta = W(self.beta)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x):
        size = x.size()
        x = x.view(x.size(0), -1)
        x = (x - th.mean(x, 1).unsqueeze(1)) / th.sqrt(th.var(x, 1).unsqueeze(1) + self.epsilon)
        if self.learnable:
            x = self.alpha.expand_as(x) * x + self.beta.expand_as(x)
        return x.view(size)
