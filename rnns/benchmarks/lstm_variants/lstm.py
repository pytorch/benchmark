#!/usr/bin/env python

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V

from .normalize import LayerNorm

"""
Implementation of LSTM variants.

For now, they only support a sequence size of 1, and are ideal for RL use-cases.
Besides that, they are a stripped-down version of PyTorch's RNN layers.
(no bidirectional, no num_layers, no batch_first)
"""


@th.jit.script
def slowlstm_cell(x, h, c, w_xi, w_hi, b_i,
                  w_xf, w_hf, b_f, w_xo, w_ho, b_o,
                  w_xc, w_hc, b_c):
    h = h.view((h.size(1), -1))
    c = c.view((c.size(1), -1))
    x = x.view((x.size(1), -1))
    # Linear mappings
    i_t = th.mm(x, w_xi) + th.mm(h, w_hi) + b_i
    f_t = th.mm(x, w_xf) + th.mm(h, w_hf) + b_f
    o_t = th.mm(x, w_xo) + th.mm(h, w_ho) + b_o
    # activations
    i_t = i_t.sigmoid()
    f_t = f_t.sigmoid()
    o_t = o_t.sigmoid()
    # cell computations
    c_t = th.mm(x, w_xc) + th.mm(h, w_hc) + b_c
    c_t = c_t.tanh()
    c_t = th.mul(c, f_t) + th.mul(i_t, c_t)
    h_t = th.mul(o_t, th.tanh(c_t))
    # Reshape for compatibility
    h_t = h_t.view((1, h_t.size(0), -1))
    c_t = c_t.view((1, c_t.size(0), -1))
    return h_t, c_t


class SlowLSTM(nn.Module):

    """
    A pedagogic implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0,
                 jit=False):
        super(SlowLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        # input to hidden weights
        self.w_xi = P(T(hidden_size, input_size))
        self.w_xf = P(T(hidden_size, input_size))
        self.w_xo = P(T(hidden_size, input_size))
        self.w_xc = P(T(hidden_size, input_size))
        # hidden to hidden weights
        self.w_hi = P(T(hidden_size, hidden_size))
        self.w_hf = P(T(hidden_size, hidden_size))
        self.w_ho = P(T(hidden_size, hidden_size))
        self.w_hc = P(T(hidden_size, hidden_size))
        # bias terms
        self.b_i = T(hidden_size).fill_(0)
        self.b_f = T(hidden_size).fill_(0)
        self.b_o = T(hidden_size).fill_(0)
        self.b_c = T(hidden_size).fill_(0)

        # Wrap biases as parameters if desired, else as variables without gradients
        if bias:
            W = P
        else:
            W = V
        self.b_i = W(self.b_i)
        self.b_f = W(self.b_f)
        self.b_o = W(self.b_o)
        self.b_c = W(self.b_c)
        self.reset_parameters()

        self.jit = jit

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):

        if self.jit:
            h, c = hidden
            h_t, c_t = slowlstm_cell(x, h, c,
                                     self.w_xi, self.w_hi, self.b_i,
                                     self.w_xf, self.w_hf, self.b_f,
                                     self.w_xo, self.w_ho, self.b_o,
                                     self.w_xc, self.w_hc, self.b_c)
            if self.dropout > 0.0:
                F.dropout(h_t, p=self.dropout, training=self.training,
                          inplace=True)
            return h_t, (h_t, c_t)

        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)
        # Linear mappings
        i_t = th.mm(x, self.w_xi) + th.mm(h, self.w_hi) + self.b_i
        f_t = th.mm(x, self.w_xf) + th.mm(h, self.w_hf) + self.b_f
        o_t = th.mm(x, self.w_xo) + th.mm(h, self.w_ho) + self.b_o
        # activations
        i_t.sigmoid_()
        f_t.sigmoid_()
        o_t.sigmoid_()
        # cell computations
        c_t = th.mm(x, self.w_xc) + th.mm(h, self.w_hc) + self.b_c
        c_t.tanh_()
        c_t = th.mul(c, f_t) + th.mul(i_t, c_t)
        h_t = th.mul(o_t, th.tanh(c_t))
        # Reshape for compatibility
        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        if self.dropout > 0.0:
            F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
        return h_t, (h_t, c_t)

    def sample_mask(self):
        pass


class LSTM(nn.Module):

    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf

    Special args:

    dropout_method: one of
            * pytorch: default dropout implementation
            * gal: uses GalLSTM's dropout
            * moon: uses MoonLSTM's dropout
            * semeniuta: uses SemeniutaLSTM's dropout
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0,
                 dropout_method='pytorch', jit=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()
        assert(dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta'])
        self.dropout_method = dropout_method

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)

        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        if do_dropout and self.dropout_method == 'semeniuta':
            g_t = F.dropout(g_t, p=self.dropout, training=self.training)

        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)

        if do_dropout and self.dropout_method == 'moon':
            c_t.data.set_(th.mul(c_t, self.mask).data)
            c_t.data *= 1.0 / (1.0 - self.dropout)

        h_t = th.mul(o_t, c_t.tanh())

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == 'pytorch':
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == 'gal':
                h_t.data.set_(th.mul(h_t, self.mask).data)
                h_t.data *= 1.0 / (1.0 - self.dropout)

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, (h_t, c_t)


class GalLSTM(LSTM):

    """
    Implementation of Gal & Ghahramami:
    'A Theoretically Grounded Application of Dropout in Recurrent Neural Networks'
    http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
    """

    def __init__(self, *args, **kwargs):
        super(GalLSTM, self).__init__(*args, **kwargs)
        self.dropout_method = 'gal'
        self.sample_mask()


class MoonLSTM(LSTM):

    """
    Implementation of Moon & al.:
    'RNNDrop: A Novel Dropout for RNNs in ASR'
    https://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf
    """

    def __init__(self, *args, **kwargs):
        super(MoonLSTM, self).__init__(*args, **kwargs)
        self.dropout_method = 'moon'
        self.sample_mask()


class SemeniutaLSTM(LSTM):
    """
    Implementation of Semeniuta & al.:
    'Recurrent Dropout without Memory Loss'
    https://arxiv.org/pdf/1603.05118.pdf
    """

    def __init__(self, *args, **kwargs):
        super(SemeniutaLSTM, self).__init__(*args, **kwargs)
        self.dropout_method = 'semeniuta'


class LayerNormLSTM(LSTM):

    """
    Layer Normalization LSTM, based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf

    Special args:
        ln_preact: whether to Layer Normalize the pre-activations.
        learnable: whether the LN alpha & gamma should be used.
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0,
                 dropout_method='pytorch', ln_preact=True, learnable=True,
                 jit=False):
        super(LayerNormLSTM, self).__init__(input_size=input_size,
                                            hidden_size=hidden_size,
                                            bias=bias,
                                            dropout=dropout,
                                            dropout_method=dropout_method)
        if ln_preact:
            self.ln_i2h = LayerNorm(4 * hidden_size, learnable=learnable)
            self.ln_h2h = LayerNorm(4 * hidden_size, learnable=learnable)
        self.ln_preact = ln_preact
        self.ln_cell = LayerNorm(hidden_size, learnable=learnable)

    def forward(self, x, hidden):
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)
        if self.ln_preact:
            i2h = self.ln_i2h(i2h)
            h2h = self.ln_h2h(h2h)
        preact = i2h + h2h

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        if do_dropout and self.dropout_method == 'semeniuta':
            g_t = F.dropout(g_t, p=self.dropout, training=self.training)

        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)

        if do_dropout and self.dropout_method == 'moon':
            c_t.data.set_(th.mul(c_t, self.mask).data)
            c_t.data *= 1.0 / (1.0 - self.dropout)

        c_t = self.ln_cell(c_t)
        h_t = th.mul(o_t, c_t.tanh())

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == 'pytorch':
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == 'gal':
                h_t.data.set_(th.mul(h_t, self.mask).data)
                h_t.data *= 1.0 / (1.0 - self.dropout)

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, (h_t, c_t)


class LayerNormGalLSTM(LayerNormLSTM):

    """
    Mixes GalLSTM's Dropout with Layer Normalization
    """

    def __init__(self, *args, **kwargs):
        super(LayerNormGalLSTM, self).__init__(*args, **kwargs)
        self.dropout_method = 'gal'
        self.sample_mask()


class LayerNormMoonLSTM(LayerNormLSTM):

    """
    Mixes MoonLSTM's Dropout with Layer Normalization
    """

    def __init__(self, *args, **kwargs):
        super(LayerNormMoonLSTM, self).__init__(*args, **kwargs)
        self.dropout_method = 'moon'
        self.sample_mask()


class LayerNormSemeniutaLSTM(LayerNormLSTM):

    """
    Mixes SemeniutaLSTM's Dropout with Layer Normalization
    """

    def __init__(self, *args, **kwargs):
        super(LayerNormSemeniutaLSTM, self).__init__(*args, **kwargs)
        self.dropout_method = 'semeniuta'
