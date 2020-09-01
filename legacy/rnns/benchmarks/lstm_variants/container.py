#!/usr/bin/env python

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V

"""
A helper class to contruct multi-layered LSTMs.
"""


class MultiLayerLSTM(nn.Module):

    """
    MultiLayer LSTM of any type.

    Note: Dropout is deactivated on the last layer.
    """

    def __init__(self, input_size, layer_type, layer_sizes=(64, 64), *args, **kwargs):
        super(MultiLayerLSTM, self).__init__()
        rnn = layer_type
        layers = []
        prev_size = input_size
        for size in layer_sizes[:-1]:
            layer = rnn(input_size=prev_size, hidden_size=size, *args, **kwargs)
            layers.append(layer)
            prev_size = size
        if 'dropout' in kwargs:
            del kwargs['dropout']
        layer = rnn(input_size=prev_size, hidden_size=size, dropout=0.0,
                    *args, **kwargs)
        layers.append(layer)
        self.layers = layers
        self.layer_sizes = layer_sizes
        self.input_size = input_size
        self.params = nn.ModuleList(layers)

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()

    def create_hiddens(self, bsz=1):
        # Uses Xavier init here.
        hiddens = []
        for l in self.layers:
            std = math.sqrt(2.0 / (l.input_size + l.hidden_size))
            hiddens.append([V(T(1, bsz, l.hidden_size).normal_(0, std)),
                            V(T(1, bsz, l.hidden_size).normal_(0, std))])
        return hiddens

    def sample_mask(self):
        for l in self.layers:
            l.sample_mask()

    def forward(self, x, hiddens):
        new_hiddens = []
        for l, h in zip(self.layers, hiddens):
            print('asdf')
            x, new_h = l(x, h)
            new_hiddens.append(new_h)
        return x, new_hiddens
